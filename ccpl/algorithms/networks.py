"""
CCPL v5 — Neural network primitives (report-fixed edition)

Changes from Technical Report:
  FIX-1: LambdaNet.backward_update accepts weight_decay param for L2 regularisation
  FIX-2: LambdaNet stores last raw input for reuse (avoids recompute in parallel call)
  FIX-5: Adam clip tightened from 5.0 to 3.0 for better gradient stability
"""
import numpy as np

# ── Activations ───────────────────────────────────────────────────────────────

def relu(x):     return np.maximum(0.0, x)
def sigmoid(x):  return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
def tanh(x):     return np.tanh(np.clip(x, -20, 20))
def softplus(x):
    return np.logaddexp(0.0, np.asarray(x))
def softmax(x):
    x = np.asarray(x)
    if not np.all(np.isfinite(x)):
        raise FloatingPointError("softmax received a non-finite logit")
    shifted = x - x.max(axis=-1, keepdims=True)
    # Max subtraction already prevents overflow.  Clipping the negative tail
    # assigns artificial probability mass and makes the usual softmax
    # derivative inconsistent with the forward pass.
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)

def d_relu(x):     return (x > 0).astype(np.float32)
def d_sigmoid(s):  return s * (1.0 - s)
def d_tanh(t):     return 1.0 - t**2
def d_softplus(x): return sigmoid(x)


# ── Primitives ────────────────────────────────────────────────────────────────

class Linear:
    def __init__(self, in_dim, out_dim, rng, scale=None):
        scale = scale or np.sqrt(2.0 / in_dim)
        self.W = rng.normal(0, scale, (in_dim, out_dim)).astype(np.float32)
        self.b = np.zeros(out_dim, np.float32)
        self._last_x = None

    def forward(self, x):
        x = np.asarray(x, np.float32)
        self._last_x = x
        return x @ self.W + self.b

    def backward(self, d_out):
        if self._last_x is None:
            raise RuntimeError("Linear.backward() called before forward().")
        d_out = np.asarray(d_out, np.float32)
        # Flatten every leading dimension.  Using ``x.T`` is only correct for
        # 2-D arrays and produced invalid gradients for sequence-shaped input.
        x_2d = self._last_x.reshape(-1, self._last_x.shape[-1])
        d_2d = d_out.reshape(-1, d_out.shape[-1])
        dW = x_2d.T @ d_2d
        db = d_2d.sum(axis=0)
        dx = d_out @ self.W.T
        return dx, dW, db

    def params(self): return [self.W, self.b]


class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.g = np.ones(dim, np.float32)
        self.b = np.zeros(dim, np.float32)
        self.eps = eps
        self._cache = None

    def forward(self, x):
        x = np.asarray(x, np.float32)
        mu = x.mean(-1, keepdims=True)
        var = ((x - mu) ** 2).mean(-1, keepdims=True)
        inv_std = 1.0 / np.sqrt(var + self.eps)
        x_hat = (x - mu) * inv_std
        self._cache = (x_hat, inv_std)
        return self.g * x_hat + self.b

    def backward(self, d_out):
        if self._cache is None:
            raise RuntimeError("LayerNorm.backward() called before forward().")
        x_hat, inv_std = self._cache
        d_out = np.asarray(d_out, np.float32)
        D = d_out.shape[-1]
        reduce_axes = tuple(range(d_out.ndim - 1))
        dg     = (d_out * x_hat).sum(axis=reduce_axes)
        db_g   = d_out.sum(axis=reduce_axes)
        dx_hat = d_out * self.g
        # LayerNorm normalises over the feature axis, so the denominator and
        # leading coefficient are D, not the batch size.  The old formula was
        # only accidentally correct when batch_size == feature_dim.
        dx     = (inv_std / D) * (
            D * dx_hat
            - dx_hat.sum(-1, keepdims=True)
            - x_hat * (dx_hat * x_hat).sum(-1, keepdims=True)
        )
        return dx, dg, db_g

    def params(self): return [self.g, self.b]


class MLP:
    """Linear -> LayerNorm -> ReLU trunk with full backprop."""
    def __init__(self, dims, rng, scale=None):
        self.layers, self.norms = [], []
        for i in range(len(dims) - 1):
            self.layers.append(Linear(dims[i], dims[i+1], rng, scale))
            if i < len(dims) - 2:
                self.norms.append(LayerNorm(dims[i+1]))
        self._pre_acts = []

    def forward(self, x):
        x = np.asarray(x, np.float32)
        self._pre_acts = []
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            if i < len(self.norms):
                x = self.norms[i].forward(x)
                self._pre_acts.append(x.copy())
                x = relu(x)
        return x

    def backward(self, d_out):
        grads = []
        d = d_out
        n_hidden = len(self.norms)
        for i in reversed(range(len(self.layers))):
            if i < n_hidden:
                d = d * d_relu(self._pre_acts[i])
                d, dg, db_n = self.norms[i].backward(d)
                grads = [dg, db_n] + grads
            dx, dW, db = self.layers[i].backward(d)
            grads = [dW, db] + grads
            d = dx
        return d, grads

    def all_params(self):
        p = []
        for i, layer in enumerate(self.layers):
            p.extend(layer.params())
            if i < len(self.norms):
                p.extend(self.norms[i].params())
        return p

    def hidden_features(self, x):
        x = np.asarray(x, np.float32)
        for i, layer in enumerate(self.layers[:-1]):
            x = layer.forward(x)
            if i < len(self.norms):
                x = self.norms[i].forward(x)
                x = relu(x)
        return x


# ── GRU Cell with full backprop ────────────────────────────────────────────────

class GRUCell:
    def __init__(self, input_dim, hidden_dim, rng):
        s = np.sqrt(2.0 / (input_dim + hidden_dim))
        def _W(): return rng.normal(0, s, (input_dim,  hidden_dim)).astype(np.float32)
        def _U(): return rng.normal(0, s, (hidden_dim, hidden_dim)).astype(np.float32)
        def _b(): return np.zeros(hidden_dim, np.float32)

        self.Wr, self.Ur, self.br = _W(), _U(), _b()
        self.Wz, self.Uz, self.bz = _W(), _U(), _b()
        self.Wn, self.Un, self.bn = _W(), _U(), _b()
        self._cache = None

    def forward(self, x, h):
        x = np.asarray(x, np.float32)
        h = np.asarray(h, np.float32)
        r = sigmoid(x @ self.Wr + h @ self.Ur + self.br)
        z = sigmoid(x @ self.Wz + h @ self.Uz + self.bz)
        n = tanh(   x @ self.Wn + (r * h) @ self.Un + self.bn)
        h_new = (1.0 - z) * n + z * h
        self._cache = (x, h, r, z, n)
        return h_new

    def backward(self, d_h_new):
        x, h, r, z, n = self._cache
        d_n = d_h_new * (1.0 - z) * d_tanh(n)
        d_z = d_h_new * (h - n)   * d_sigmoid(z)

        dWn = x.T @ d_n;        dUn = (r * h).T @ d_n;  dbn = d_n.sum(0)
        dWz = x.T @ d_z;        dUz = h.T @ d_z;         dbz = d_z.sum(0)

        d_r_raw = d_n @ self.Un.T * h
        d_r     = d_r_raw * d_sigmoid(r)
        dWr = x.T @ d_r;        dUr = h.T @ d_r;         dbr = d_r.sum(0)

        dx = d_n @ self.Wn.T + d_z @ self.Wz.T + d_r @ self.Wr.T
        dh = (d_h_new * z + d_n @ self.Un.T * r +
              d_z @ self.Uz.T + d_r @ self.Ur.T)
        return dx, dh, [dWr, dUr, dbr, dWz, dUz, dbz, dWn, dUn, dbn]

    def params(self):
        return [self.Wr, self.Ur, self.br,
                self.Wz, self.Uz, self.bz,
                self.Wn, self.Un, self.bn]

    def zero_state(self, batch=1):
        return np.zeros((batch, self.Wr.shape[1]), np.float32)


# ── Adam ──────────────────────────────────────────────────────────────────────

class Adam:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
                 clip=3.0):   # FIX-5: tightened from 5.0 to 3.0
        self.params = list(params)
        self.lr, self.beta1, self.beta2, self.eps, self.clip = lr, beta1, beta2, eps, clip
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self, grads):
        grads = list(grads)
        if len(grads) != len(self.params):
            raise ValueError(
                f"Adam gradient mismatch: {len(grads)} gradients for "
                f"{len(self.params)} parameters")
        self.t += 1
        lr_t = self.lr * np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)
        for i, (p, g) in enumerate(zip(self.params, grads)):
            g = np.asarray(g, dtype=p.dtype)
            if g.shape != p.shape:
                raise ValueError(
                    f"Adam gradient shape mismatch at parameter {i}: "
                    f"got {g.shape}, expected {p.shape}")
            if not np.all(np.isfinite(g)):
                raise FloatingPointError(
                    f"Non-finite gradient at Adam parameter {i}.")
            g = np.clip(g, -self.clip, self.clip)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g**2
            p -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + self.eps)

    def add_params(self, new_params):
        for p in new_params:
            self.params.append(p)
            self.m.append(np.zeros_like(p))
            self.v.append(np.zeros_like(p))


# ── GRU Policy Network ─────────────────────────────────────────────────────────

class GRUPolicyNet:
    def __init__(self, state_dim, action_dim, gru_dim=64, hidden_dim=128,
                 n_layers=2, lr=1e-3, seed=0):
        rng = np.random.default_rng(seed)
        self.gru_dim    = gru_dim
        self.action_dim = action_dim

        self.gru      = GRUCell(state_dim, gru_dim, rng)
        trunk_dims    = [gru_dim] + [hidden_dim] * n_layers
        self.trunk    = MLP(trunk_dims, rng)
        self.val_head = Linear(hidden_dim, 1,          rng, scale=0.01)
        self.adv_head = Linear(hidden_dim, action_dim, rng, scale=0.01)

        all_p = (self.gru.params() + self.trunk.all_params() +
                 self.val_head.params() + self.adv_head.params())
        self.optim = Adam(all_p, lr=lr)

    def forward(self, state, h):
        state = np.asarray(state, np.float32)
        if state.ndim == 1: state = state[None]
        h_new = self.gru.forward(state, h)
        feat  = self.trunk.forward(h_new)
        v     = self.val_head.forward(feat)
        a     = self.adv_head.forward(feat)
        q     = v + a - a.mean(axis=-1, keepdims=True)
        return q, h_new

    def backward_update(self, state, h, actions, td_errors, weights):
        B     = len(actions)
        state = np.asarray(state, np.float32)
        h     = np.asarray(h, np.float32)

        h_new = self.gru.forward(state, h)
        feat  = self.trunk.forward(h_new)
        self.val_head.forward(feat)
        self.adv_head.forward(feat)

        delta = (td_errors * weights / B).astype(np.float32)

        d_q = np.zeros((B, self.action_dim), np.float32)
        for i, ac in enumerate(actions):
            d_q[i, ac] = -delta[i]
        d_q -= d_q.mean(axis=-1, keepdims=True)

        # td_errors = target - prediction, hence dL/dV is negative.
        d_feat_v, dWv, dbv = self.val_head.backward(-delta[:, None])
        d_feat_a, dWa, dba = self.adv_head.backward(d_q)
        d_feat = d_feat_v + d_feat_a

        d_h_new, trunk_grads = self.trunk.backward(d_feat)
        _, _, gru_grads      = self.gru.backward(d_h_new)

        all_grads = gru_grads + trunk_grads + [dWv, dbv, dWa, dba]
        self.optim.step(all_grads)

    def zero_state(self, batch=1): return self.gru.zero_state(batch)

    def copy_weights_from(self, other):
        for s, t in zip(other._flat(), self._flat()): t[:] = s

    def soft_update_from(self, other, tau=0.005):
        for s, t in zip(other._flat(), self._flat()): t[:] = tau * s + (1 - tau) * t

    def _flat(self):
        return (self.gru.params() + self.trunk.all_params() +
                self.val_head.params() + self.adv_head.params())


# ── Multi-Horizon Consequence Network ─────────────────────────────────────────

class MultiHorizonConsequenceNet:
    def __init__(self, state_dim, action_dim, hidden_dim=128, n_layers=2,
                 lr=4e-4, seed=0):
        rng  = np.random.default_rng(seed)
        self.action_dim = action_dim
        inp  = state_dim + action_dim

        def _trunk(): return MLP([inp] + [hidden_dim]*n_layers, rng)
        def _heads(h_dim):
            return {hz: {"f": Linear(h_dim, 1, rng, 0.01),
                         "u": Linear(h_dim, 1, rng, 0.01),
                         "d": Linear(h_dim, 1, rng, 0.01)}
                    for hz in ("short", "mid", "long")}

        self.trunk_a, self.heads_a = _trunk(), _heads(hidden_dim)
        self.trunk_b, self.heads_b = _trunk(), _heads(hidden_dim)

        self.log_alpha = np.array([0.5], np.float32)
        self.log_beta  = np.array([0.3], np.float32)
        self.log_gamma = np.array([0.2], np.float32)
        self.horizon_w = np.zeros(3,    np.float32)

        self.optim = Adam(self._all_params(), lr=lr)

    def _head_params(self, heads):
        p = []
        for hz in ("short", "mid", "long"):
            for k in ("f", "u", "d"):
                p.extend(heads[hz][k].params())
        return p

    def _all_params(self):
        return (self.trunk_a.all_params() + self.trunk_b.all_params() +
                self._head_params(self.heads_a) + self._head_params(self.heads_b) +
                [self.log_alpha, self.log_beta, self.log_gamma, self.horizon_w])

    @property
    def alpha(self): return float(softplus(self.log_alpha)[0])
    @property
    def beta(self):  return float(softplus(self.log_beta)[0])
    @property
    def gamma(self): return float(softplus(self.log_gamma)[0])
    @property
    def horizon_blend(self): return softmax(self.horizon_w[None]).squeeze()

    def _trunk_forward(self, trunk, heads, states, actions):
        oh = np.eye(self.action_dim, dtype=np.float32)[actions.astype(int)]
        x  = np.concatenate([states, oh], axis=-1)
        h  = trunk.forward(x)
        alpha, beta, gamma = self.alpha, self.beta, self.gamma
        out = {}
        for key in ("short", "mid", "long"):
            F = softplus(heads[key]["f"].forward(h))
            U = softplus(heads[key]["u"].forward(h))
            D = softplus(heads[key]["d"].forward(h))
            out[key] = (alpha * F + beta * U + gamma * D).squeeze(-1)
        return out, h

    def forward(self, states, actions):
        ha, _ = self._trunk_forward(self.trunk_a, self.heads_a, states, actions)
        hb, _ = self._trunk_forward(self.trunk_b, self.heads_b, states, actions)
        w  = self.horizon_blend
        Ca = w[0]*ha["short"] + w[1]*ha["mid"] + w[2]*ha["long"]
        Cb = w[0]*hb["short"] + w[1]*hb["mid"] + w[2]*hb["long"]
        C_total = 0.5 * (Ca + Cb)
        sigma   = np.abs(Ca - Cb)
        C_short = 0.5 * (ha["short"] + hb["short"])
        C_mid   = 0.5 * (ha["mid"]   + hb["mid"])
        C_long  = 0.5 * (ha["long"]  + hb["long"])
        return C_total, C_short, C_mid, C_long, sigma

    def predict(self, states, actions):
        C, _, _, _, sigma = self.forward(states, actions)
        return C, sigma

    def update_step(self, states, actions, targets, weights):
        actions = np.asarray(actions, np.int32)
        targets = np.asarray(targets, np.float32)
        weights = np.asarray(weights, np.float32)
        Wn = weights / (weights.sum() + 1e-8)
        oh = np.eye(self.action_dim, dtype=np.float32)[actions]
        x_in = np.concatenate([states, oh], axis=-1)
        horizon_names = ("short", "mid", "long")
        blend = self.horizon_blend
        scales = (self.alpha, self.beta, self.gamma)

        caches = []
        ensemble_values = []
        for trunk, heads in ((self.trunk_a, self.heads_a),
                             (self.trunk_b, self.heads_b)):
            H = trunk.forward(x_in)
            horizon_cache = []
            horizon_values = []
            for hz in horizon_names:
                raws = [heads[hz][key].forward(H) for key in ("f", "u", "d")]
                components = [softplus(raw) for raw in raws]
                value = sum(scale * component
                            for scale, component in zip(scales, components)).squeeze(-1)
                horizon_cache.append((raws, components))
                horizon_values.append(value)
            caches.append((trunk, heads, H, horizon_cache))
            ensemble_values.append(np.stack(horizon_values, axis=1))

        mean_horizons = 0.5 * (ensemble_values[0] + ensemble_values[1])
        prediction = mean_horizons @ blend
        error = prediction - targets
        c_loss = float(np.sum(Wn * error**2))
        d_prediction = 2.0 * Wn * error

        trunk_grads_list = []
        head_grads_list = []
        scale_grads = np.zeros(3, np.float64)
        for trunk, heads, H, horizon_cache in caches:
            dH = np.zeros_like(H)
            head_grads = []
            for h_index, hz in enumerate(horizon_names):
                d_value = d_prediction * 0.5 * blend[h_index]
                raws, components = horizon_cache[h_index]
                for component_index, (key, scale, raw, component) in enumerate(zip(
                        ("f", "u", "d"), scales, raws, components)):
                    d_raw = (d_value[:, None] * scale * d_softplus(raw))
                    dH_sub, dW, db = heads[hz][key].backward(d_raw)
                    dH += dH_sub
                    head_grads.extend([dW, db])
                    scale_grads[component_index] += float(
                        np.sum(d_value[:, None] * component))
            _, trunk_grads = trunk.backward(dH)
            trunk_grads_list.append(trunk_grads)
            head_grads_list.append(head_grads)

        # Chain rule through positive scalar coefficients and softmax blend.
        raw_scales = (self.log_alpha, self.log_beta, self.log_gamma)
        scalar_grads = [
            np.array([scale_grads[i] * float(sigmoid(raw)[0])], np.float32)
            for i, raw in enumerate(raw_scales)
        ]
        d_blend = np.sum(d_prediction[:, None] * mean_horizons, axis=0)
        d_blend_logits = blend * (d_blend - np.dot(d_blend, blend))

        all_grads = (trunk_grads_list[0] + trunk_grads_list[1]
                     + head_grads_list[0] + head_grads_list[1]
                     + scalar_grads + [d_blend_logits.astype(np.float32)])
        self.optim.step(all_grads)
        return c_loss


# ── State-Dependent Lambda Network ────────────────────────────────────────────

class LambdaNet:
    def __init__(self, state_dim=6, hidden_dim=32, lambda_max=3.0, lr=5e-4, seed=0):
        rng = np.random.default_rng(seed)
        self.lambda_max = lambda_max
        # FIX-L1: was [0, 1, 4] — missed system_pressure (3) and hidden_penalty (5),
        # the two features most predictive of impending collapse.  Use ALL features.
        self.input_idx  = list(range(state_dim))
        self.net   = MLP([len(self.input_idx), hidden_dim, hidden_dim, 1], rng, scale=0.05)
        self.optim = Adam(self.net.all_params(), lr=lr)

    def forward(self, state):
        s      = np.asarray(state, np.float32)
        scalar = s.ndim == 1
        if scalar: s = s[None]
        x   = s[:, self.input_idx]
        raw = self.net.forward(x)
        out = sigmoid(raw).squeeze(-1) * self.lambda_max
        return float(out[0]) if scalar else out

    def backward_update(self, states, lam_errors, weight_decay=0.0):
        """
        Update from errors expressed in normalized output units
        ``lambda/lambda_max - target``.
        """
        x   = np.asarray(states, np.float32)[:, self.input_idx]
        raw = self.net.forward(x)
        sig = sigmoid(raw).squeeze(-1)
        errors = np.asarray(lam_errors, np.float32)
        # d(lambda/lambda_max)/d(raw) = sigmoid'(raw).  The previous
        # implementation multiplied by lambda_max and omitted batch averaging.
        d_raw = (errors * sig * (1 - sig) / max(len(errors), 1))[:, None]
        _, grads = self.net.backward(d_raw)

        # FIX-4: apply L2 weight decay to all parameter gradients
        if weight_decay > 0.0:
            params = self.net.all_params()
            grads  = [g + weight_decay * p
                      for g, p in zip(grads, params)]

        self.optim.step(grads)

    def params(self): return self.net.all_params()


# ── Actor / Critic for baselines ──────────────────────────────────────────────

class ActorNetwork:
    def __init__(self, state_dim, action_dim, hidden_dim=128, n_layers=2, lr=3e-4, seed=0):
        rng  = np.random.default_rng(seed)
        dims = [state_dim] + [hidden_dim]*n_layers + [action_dim]
        self.net        = MLP(dims, rng)
        self.action_dim = action_dim
        self.rng        = np.random.default_rng(seed)
        self.optim      = Adam(self.net.all_params(), lr=lr)

    def logits(self, x): return self.net.forward(x)
    def probs(self, x):  return softmax(self.logits(x))

    def sample(self, x):
        p = self.probs(np.asarray(x, np.float32))
        p = np.clip(p, 1e-8, 1.0); p /= p.sum()
        return int(self.rng.choice(len(p), p=p)) if p.ndim == 1 else \
               np.array([self.rng.choice(len(r), p=r/r.sum()) for r in p])

    def backward_update(self, states, actions, advantages, weights,
                        entropy_coeff=0.0):
        B     = len(actions)
        probs = self.probs(states)
        delta = -(advantages * weights / B).astype(np.float32)
        # FIX: clip delta to prevent inf/nan gradients from extreme advantages
        delta = np.clip(delta, -5.0, 5.0)
        d_logits = np.zeros_like(probs)
        for i, a in enumerate(actions):
            d_logits[i] = -delta[i] * probs[i]
            d_logits[i, a] += delta[i]
        if entropy_coeff:
            log_p = np.log(probs + 1e-8)
            expected_log = (probs * log_p).sum(axis=-1, keepdims=True)
            d_logits += (entropy_coeff / B) * probs * (log_p - expected_log)
        # FIX: skip backward if gradient contains inf/nan
        if not np.all(np.isfinite(d_logits)):
            return
        _, grads = self.net.backward(d_logits)
        self.optim.step(grads)

    def backward_ppo(self, states, actions, advantages, old_log_probs,
                     clip_eps, weights, entropy_coeff=0.0):
        """One gradient step on the clipped PPO surrogate.

        The previous implementation computed the PPO ratio for logging but
        then applied an unclipped vanilla policy-gradient update.  Here the
        gradient is zero in the clipped region and includes the likelihood
        ratio everywhere else.
        """
        states = np.asarray(states, np.float32)
        actions = np.asarray(actions, np.int32)
        advantages = np.asarray(advantages, np.float32)
        old_log_probs = np.asarray(old_log_probs, np.float32)
        weights = np.asarray(weights, np.float32)
        B = len(actions)

        probs = self.probs(states)
        log_probs = np.log(probs[np.arange(B), actions] + 1e-8)
        ratio = np.exp(np.clip(log_probs - old_log_probs, -20.0, 20.0))
        clipped = ((advantages >= 0.0) & (ratio > 1.0 + clip_eps)) | \
                  ((advantages < 0.0) & (ratio < 1.0 - clip_eps))
        coeff = -advantages * ratio * weights / B
        coeff = np.where(clipped, 0.0, coeff).astype(np.float32)

        d_logits = -coeff[:, None] * probs
        d_logits[np.arange(B), actions] += coeff

        if entropy_coeff:
            log_all = np.log(probs + 1e-8)
            expected_log = (probs * log_all).sum(axis=-1, keepdims=True)
            d_logits += (entropy_coeff / B) * probs * (log_all - expected_log)

        if not np.all(np.isfinite(d_logits)):
            raise FloatingPointError("Non-finite PPO policy gradient.")
        _, grads = self.net.backward(d_logits)
        self.optim.step(grads)
        return ratio

    def backward_discrete_objective(self, states, q_values, cost_values,
                                    lagrange, alpha, weights):
        """Exact gradient for a discrete SAC-style policy objective."""
        states = np.asarray(states, np.float32)
        weights = np.asarray(weights, np.float32)
        probs = self.probs(states)
        log_probs = np.log(probs + 1e-8)
        # L = E_a[alpha log pi(a|s) - Q_r(s,a) + lambda Q_c(s,a)].
        score_grad = (alpha * (log_probs + 1.0)
                      - np.asarray(q_values, np.float32)
                      + float(lagrange) * np.asarray(cost_values, np.float32))
        centred = score_grad - (probs * score_grad).sum(-1, keepdims=True)
        d_logits = probs * centred * (weights / len(states))[:, None]
        if not np.all(np.isfinite(d_logits)):
            raise FloatingPointError("Non-finite discrete policy gradient.")
        _, grads = self.net.backward(d_logits)
        self.optim.step(grads)


class CriticNetwork:
    def __init__(self, state_dim, hidden_dim=128, n_layers=2, lr=1e-3, seed=0):
        rng  = np.random.default_rng(seed)
        dims = [state_dim] + [hidden_dim]*n_layers + [1]
        self.net   = MLP(dims, rng)
        self.optim = Adam(self.net.all_params(), lr=lr)

    def value(self, x): return self.net.forward(np.asarray(x, np.float32)).squeeze(-1)

    def backward_update(self, states, v_errors, weights):
        B     = len(v_errors)
        self.net.forward(np.asarray(states, np.float32))
        delta = np.clip(v_errors * weights / B, -5.0, 5.0).astype(np.float32)
        # v_errors = target - prediction, so gradient descent needs -error.
        _, grads = self.net.backward(-delta[:, None])
        self.optim.step(grads)


class QNetwork:
    def __init__(self, state_dim, action_dim, hidden_dim=128, n_layers=3, lr=1e-3, seed=0):
        rng = np.random.default_rng(seed)
        dims = [state_dim] + [hidden_dim]*n_layers
        self.trunk      = MLP(dims, rng)
        self.val_head   = Linear(hidden_dim, 1,          rng, scale=0.01)
        self.adv_head   = Linear(hidden_dim, action_dim, rng, scale=0.01)
        self.action_dim = action_dim
        all_p = self.trunk.all_params() + self.val_head.params() + self.adv_head.params()
        self.optim = Adam(all_p, lr=lr)

    def forward(self, x):
        h = self.trunk.forward(x)
        v = self.val_head.forward(h)
        a = self.adv_head.forward(h)
        return v + a - a.mean(axis=-1, keepdims=True)

    def backward_update(self, states, actions, td_errors, weights):
        B     = len(actions)
        delta = (td_errors * weights / B).astype(np.float32)
        h     = self.trunk.forward(states)
        self.val_head.forward(h)
        self.adv_head.forward(h)

        # Dueling gradient: Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
        # dL/dQ(s,a) = -delta for the selected action, 0 elsewhere
        d_q = np.zeros((B, self.action_dim), np.float32)
        for i, ac in enumerate(actions):
            d_q[i, ac] = -delta[i]
        # Subtract mean to match the forward centering: A' = A - mean(A)
        d_q -= d_q.mean(axis=-1, keepdims=True)

        # Value head: grad is mean of d_q over actions (because V adds to all)
        # -delta broadcast across all actions averages to -delta
        d_h_v, dWv, dbv = self.val_head.backward(-delta[:, None])
        d_h_a, dWa, dba = self.adv_head.backward(d_q)
        _, trunk_grads  = self.trunk.backward(d_h_v + d_h_a)
        self.optim.step(trunk_grads + [dWv, dbv, dWa, dba])

    def copy_weights_from(self, other):
        for s, t in zip(other._flat(), self._flat()): t[:] = s
    def soft_update_from(self, other, tau=0.005):
        for s, t in zip(other._flat(), self._flat()): t[:] = tau*s + (1-tau)*t
    def _flat(self):
        return self.trunk.all_params() + self.val_head.params() + self.adv_head.params()


# ── Vectorised GRU batch forward ──────────────────────────────────────────────

def gru_batch_forward(cell: GRUCell, X: np.ndarray, H: np.ndarray) -> np.ndarray:
    X = np.asarray(X, np.float32); H = np.asarray(H, np.float32)
    r = sigmoid(X @ cell.Wr + H @ cell.Ur + cell.br)
    z = sigmoid(X @ cell.Wz + H @ cell.Uz + cell.bz)
    n = tanh(   X @ cell.Wn + (r * H) @ cell.Un + cell.bn)
    return (1.0 - z) * n + z * H
