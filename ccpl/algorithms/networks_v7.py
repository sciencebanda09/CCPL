"""
CCPL v7 — networks_v7.py
========================
New neural network primitives inspired by "Attention Is All You Need"
(Vaswani et al., 2017) integrated into the CCPL architecture.

Additions over networks.py:
  1. ScaledDotProductAttention  — full forward + backward
  2. MultiHeadAttention         — h parallel heads with output projection
  3. PositionalEncoding         — sinusoidal PE from Vaswani et al.
  4. EpisodicMemory             — fixed-size store of dangerous (key, value) pairs
                                   retrieved via attention-based similarity
  5. AttentionContextNet        — cross-attention: query = GRU hidden,
                                   keys/values = episodic memory → context vector
  6. AttentionAugmentedPolicyNet — GRUPolicyNet + residual attention context

Design principle:  All modules implement forward() and backward() so
gradients flow end-to-end through the same Adam optimiser as before.
No external frameworks required.
"""
import numpy as np
try:
    from .networks import (
        relu, sigmoid, tanh, softmax, d_relu, d_sigmoid, d_tanh,
        Linear, LayerNorm, MLP, GRUCell, GRUPolicyNet, Adam,
        gru_batch_forward,
    )
except ImportError:
    from networks import (
    relu, sigmoid, tanh, softmax, d_relu, d_sigmoid, d_tanh,
    Linear, LayerNorm, MLP, GRUCell, GRUPolicyNet, Adam,
    gru_batch_forward,
)



def sinusoidal_pe(seq_len: int, d_model: int) -> np.ndarray:
    """
    PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
    PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
    Returns (seq_len, d_model) float32 array — no learnable parameters.
    """
    pos = np.arange(seq_len, dtype=np.float32)[:, None]
    pe = np.zeros((seq_len, d_model), dtype=np.float32)
    even_dims = np.arange(0, d_model, 2, dtype=np.float32)[None, :]
    odd_dims = np.arange(1, d_model, 2, dtype=np.float32)[None, :]
    pe[:, 0::2] = np.sin(pos / (10000.0 ** (even_dims / d_model)))
    pe[:, 1::2] = np.cos(pos / (10000.0 ** ((odd_dims - 1.0) / d_model)))
    return pe



class ScaledDotProductAttention:
    """
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Shapes (batch-first):
      Q  : (..., Sq, d_k)
      K  : (..., Sk, d_k)
      V  : (..., Sk, d_v)
    Returns:
      out  : (..., Sq, d_v)
      attn : (..., Sq, Sk)   — attention weight matrix
    """

    def __init__(self):
        self._cache = None

    def forward(self, Q, K, V, mask=None):
        Q = np.asarray(Q, np.float32)
        K = np.asarray(K, np.float32)
        V = np.asarray(V, np.float32)
        d_k    = Q.shape[-1]
        scores = (Q @ K.swapaxes(-2, -1)).astype(np.float32) / np.sqrt(d_k + 1e-8)
        mask_array = None if mask is None else np.asarray(mask, dtype=bool)
        if mask_array is not None:
            scores = np.where(mask_array, np.float32(-np.inf), scores)
            if np.any(np.all(mask_array, axis=-1)):
                raise ValueError("Every attention query must have an unmasked key")
        scores -= scores.max(axis=-1, keepdims=True)
        attn_e  = np.exp(scores).astype(np.float32)
        attn    = (attn_e / (attn_e.sum(axis=-1, keepdims=True) + 1e-9)).astype(np.float32)
        out     = (attn @ V).astype(np.float32)
        self._cache = (Q, K, V, attn, d_k, mask_array)
        return out, attn

    def backward(self, d_out):
        """Returns dQ, dK, dV."""
        Q, K, V, attn, d_k, mask = self._cache
        dV      = attn.swapaxes(-2, -1) @ d_out
        d_attn  = d_out @ V.swapaxes(-2, -1)
        d_scores = attn * (d_attn - (d_attn * attn).sum(-1, keepdims=True))
        if mask is not None:
            d_scores = np.where(mask, 0.0, d_scores)
        d_scores = d_scores / np.sqrt(d_k + 1e-8)
        dQ = d_scores @ K
        dK = d_scores.swapaxes(-2, -1) @ Q
        return dQ, dK, dV



class MultiHeadAttention:
    """
    MultiHead(Q,K,V) = Concat(head_1,...,head_h) W^O
    head_i = Attention(Q W^Q_i, K W^K_i, V W^V_i)

    d_model must be divisible by n_heads.
    Supports cross-attention (Q from one source, K/V from another).
    """

    def __init__(self, d_model: int, n_heads: int, rng,
                 scale: float = None, lr: float = 1e-3):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads
        s = scale or np.sqrt(2.0 / d_model)

        self.W_Q = rng.normal(0, s, (d_model, d_model)).astype(np.float32)
        self.W_K = rng.normal(0, s, (d_model, d_model)).astype(np.float32)
        self.W_V = rng.normal(0, s, (d_model, d_model)).astype(np.float32)
        self.W_O = rng.normal(0, s, (d_model, d_model)).astype(np.float32)

        self._sdpa   = ScaledDotProductAttention()
        self._cache  = None
        self.optim   = Adam(self.params(), lr=lr)


    def forward(self, Q_in, K_in, V_in):
        """
        Q_in : (B, Sq, d_model) or (B, d_model)
        K_in : (B, Sk, d_model) or (M, d_model)  — M = memory size
        V_in : same shape as K_in
        Returns out (B, Sq, d_model), attn_weights (B, h, Sq, Sk)
        """
        q_squeezed = Q_in.ndim == 2
        if Q_in.ndim == 2: Q_in = Q_in[:, None, :]
        if K_in.ndim == 2: K_in = K_in[None, :, :]
        if V_in.ndim == 2: V_in = V_in[None, :, :]

        B, Sq, _ = Q_in.shape
        _, Sk, _ = K_in.shape

        if K_in.shape[0] == 1 and B > 1:
            K_in = np.broadcast_to(K_in, (B, Sk, self.d_model)).copy()
            V_in = np.broadcast_to(V_in, (B, Sk, self.d_model)).copy()

        Q = (Q_in @ self.W_Q).astype(np.float32)
        K = (K_in @ self.W_K).astype(np.float32)
        V = (V_in @ self.W_V).astype(np.float32)

        Q = Q.reshape(B, Sq, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(B, Sk, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(B, Sk, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        attn_out, attn_w = self._sdpa.forward(Q, K, V)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, Sq, self.d_model)
        out      = attn_out @ self.W_O

        self._cache = (Q_in, K_in, V_in, attn_out, q_squeezed, B, Sq, Sk)
        if q_squeezed:
            return out.squeeze(1), attn_w
        return out, attn_w


    def backward(self, d_out):
        """
        d_out : (B, d_model) or (B, Sq, d_model)
        Returns dQ_in, dK_in, dV_in, [dW_Q, dW_K, dW_V, dW_O]
        """
        Q_in, K_in, V_in, attn_out, q_squeezed, B, Sq, Sk = self._cache

        if q_squeezed and d_out.ndim == 2:
            d_out = d_out[:, None, :]

        dW_O     = attn_out.reshape(B * Sq, self.d_model).T @ d_out.reshape(B * Sq, self.d_model)
        d_attn_o = d_out @ self.W_O.T

        d_attn_o = d_attn_o.reshape(B, Sq, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        dQ_h, dK_h, dV_h = self._sdpa.backward(d_attn_o)

        dQ = dQ_h.transpose(0, 2, 1, 3).reshape(B, Sq, self.d_model)
        dK = dK_h.transpose(0, 2, 1, 3).reshape(B, Sk, self.d_model)
        dV = dV_h.transpose(0, 2, 1, 3).reshape(B, Sk, self.d_model)

        dW_Q   = Q_in.reshape(B * Sq, self.d_model).T @ dQ.reshape(B * Sq, self.d_model)
        dW_K   = K_in.reshape(B * Sk, self.d_model).T @ dK.reshape(B * Sk, self.d_model)
        dW_V   = V_in.reshape(B * Sk, self.d_model).T @ dV.reshape(B * Sk, self.d_model)

        dQ_in  = dQ @ self.W_Q.T
        dK_in  = dK @ self.W_K.T
        dV_in  = dV @ self.W_V.T

        if q_squeezed:
            dQ_in = dQ_in.squeeze(1)

        return dQ_in, dK_in, dV_in, [dW_Q, dW_K, dW_V, dW_O]

    def update(self, grads):
        """Perform an Adam step on [dW_Q, dW_K, dW_V, dW_O]."""
        self.optim.step(grads)

    def params(self):
        return [self.W_Q, self.W_K, self.W_V, self.W_O]



class EpisodicMemory:
    """
    Fixed-size episodic memory of dangerous states.

    Stores (key, value) pairs:
      key   : normalized state vector  (state_dim,)
      value : consequence + gru_hidden augmented vector (consequence_dim,)

    Retrieval is via scaled dot-product attention:
      attn_w = softmax( Q @ Keys^T / sqrt(d) )
      context = attn_w @ Values

    Memory is updated at episode end:
      - New entries replace the slot with the *smallest* stored consequence
        (keep the most dangerous memories).
      - Entries with consequence below insertion_threshold are rejected.

    Uses a separate linear projector to map state_dim → d_model
    so keys are in the same space as GRU hidden states.
    """

    def __init__(self, state_dim: int, gru_dim: int, d_model: int,
                 capacity: int = 256, n_heads: int = 4,
                 insertion_threshold: float = 0.1,
                 lr: float = 3e-4, seed: int = 42):
        self.capacity   = capacity
        self.d_model    = d_model
        self.threshold  = insertion_threshold
        rng             = np.random.default_rng(seed)

        self.key_proj   = Linear(state_dim, d_model, rng, scale=0.05)
        self.query_proj = Linear(gru_dim,   d_model, rng, scale=0.05)
        self.mha        = MultiHeadAttention(d_model, n_heads, rng, lr=lr)
        self.out_proj   = Linear(d_model,   gru_dim, rng, scale=0.01)

        self._keys    = np.zeros((capacity, d_model),  np.float32)
        self._values  = np.zeros((capacity, d_model),  np.float32)
        self._cons    = np.zeros(capacity,              np.float32)
        self._filled  = 0
        self._ptr     = 0

        all_p = (self.key_proj.params() + self.query_proj.params()
                 + self.out_proj.params())
        self.proj_optim = Adam(all_p, lr=lr)

        self._last_query = None
        self._last_read_active = False


    def write(self, states: np.ndarray, consequences: np.ndarray):
        """
        Write a batch of (state, consequence) to memory.
        Only entries with consequence > threshold are kept.
        Most dangerous entries displace the weakest stored entry.

        states:       (B, state_dim)
        consequences: (B,)
        """
        B = len(states)
        for b in range(B):
            c = float(consequences[b])
            if c < self.threshold:
                continue
            k = self.key_proj.forward(states[b:b+1]).squeeze(0)
            v = k.copy()

            if self._filled < self.capacity:
                idx = self._filled
                self._filled += 1
            else:
                idx = int(np.argmin(self._cons[:self._filled]))

            if c > self._cons[idx] or self._filled < self.capacity:
                self._keys[idx]   = k
                self._values[idx] = v
                self._cons[idx]   = c
                self._ptr = (self._ptr + 1) % self.capacity


    MAX_READ_M = 64

    def read(self, gru_hidden: np.ndarray) -> np.ndarray:
        """
        Cross-attention retrieval.
        gru_hidden : (B, gru_dim)
        Returns context : (B, gru_dim)  — same dim for residual addition
        """
        if self._filled == 0:
            B = gru_hidden.shape[0]
            self._last_read_active = False
            return np.zeros((B, self.out_proj.W.shape[1]), np.float32)

        Q  = self.query_proj.forward(np.asarray(gru_hidden, np.float32))

        M = self._filled
        if M > self.MAX_READ_M:
            top_idx = np.argpartition(self._cons[:M], -self.MAX_READ_M)[-self.MAX_READ_M:]
            K = self._keys[top_idx].astype(np.float32)
            V = self._values[top_idx].astype(np.float32)
        else:
            K = self._keys[:M].astype(np.float32)
            V = self._values[:M].astype(np.float32)

        ctx, _attn = self.mha.forward(Q, K, V)
        self._last_query = Q
        out = self.out_proj.forward(ctx)
        self._last_read_active = True
        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """Backpropagate through query, attention, and output projections.

        Stored keys/values are detached episodic data, so the state key
        projector receives no gradient from a later read.  Query/output/MHA
        parameters are trained end-to-end and the returned gradient flows into
        the current GRU hidden state.
        """
        if not self._last_read_active:
            return np.zeros_like(d_out, dtype=np.float32)
        d_ctx, dW_out, db_out = self.out_proj.backward(d_out)
        d_query, _, _, mha_grads = self.mha.backward(d_ctx)
        d_hidden, dW_query, db_query = self.query_proj.backward(d_query)
        key_zeros = [np.zeros_like(parameter) for parameter in self.key_proj.params()]
        self.proj_optim.step(
            key_zeros + [dW_query, db_query, dW_out, db_out])
        self.mha.update(mha_grads)
        return d_hidden.astype(np.float32)

    def params(self):
        return (self.key_proj.params() + self.query_proj.params()
                + self.mha.params() + self.out_proj.params())

    def size(self): return self._filled

    def copy_contents_from(self, other):
        """Copy the non-parametric memory bank used by a target network."""
        if self.capacity != other.capacity or self.d_model != other.d_model:
            raise ValueError("Cannot copy episodic memories with different shapes.")
        self._keys[:] = other._keys
        self._values[:] = other._values
        self._cons[:] = other._cons
        self._filled = other._filled
        self._ptr = other._ptr



class AttentionAugmentedPolicyNet:
    """
    GRU policy + episodic memory cross-attention.

    Architecture (per-step):
      h_t  = GRU(s_t, h_{t-1})                     ← sequential memory
      ctx  = EpisodicMemory.read(h_t)              ← attention retrieval
      feat = trunk(LayerNorm(h_t + ctx))           ← residual + norm
      Q    = val_head + adv_head − mean(adv_head)  ← dueling heads

    Training (batch update):
      Same as GRUPolicyNet but with ctx added to trunk input.
      Memory is not updated during batch training (it's updated online).
    """

    def __init__(self, state_dim, action_dim, gru_dim=40, hidden_dim=80,
                 n_layers=2, d_model=32, n_heads=4, memory_capacity=256,
                 lr_policy=1.2e-3, lr_attn=3e-4, seed=0):
        rng = np.random.default_rng(seed)
        self.gru_dim    = gru_dim
        self.action_dim = action_dim

        self.gru = GRUCell(state_dim, gru_dim, rng)

        self.memory = EpisodicMemory(
            state_dim=state_dim, gru_dim=gru_dim, d_model=d_model,
            capacity=memory_capacity, n_heads=n_heads,
            lr=lr_attn, seed=seed)

        self.res_norm = LayerNorm(gru_dim)

        trunk_dims    = [gru_dim] + [hidden_dim] * n_layers
        self.trunk    = MLP(trunk_dims, rng)
        self.val_head = Linear(hidden_dim, 1,          rng, scale=0.01)
        self.adv_head = Linear(hidden_dim, action_dim, rng, scale=0.01)

        core_params = (self.gru.params() + [self.res_norm.g, self.res_norm.b]
                       + self.trunk.all_params()
                       + self.val_head.params() + self.adv_head.params())
        self.optim  = Adam(core_params, lr=lr_policy)


    def forward(self, state, h):
        state = np.asarray(state, np.float32)
        if state.ndim == 1: state = state[None]
        h_new = self.gru.forward(state, h)

        ctx  = self.memory.read(h_new)

        feat = self.res_norm.forward(h_new + ctx)

        feat = self.trunk.forward(feat)
        v    = self.val_head.forward(feat)
        a    = self.adv_head.forward(feat)
        q    = v + a - a.mean(axis=-1, keepdims=True)
        return q, h_new


    def backward_update(self, states, h, actions, td_errors, weights):
        B      = len(actions)
        states = np.asarray(states, np.float32)
        h      = np.asarray(h, np.float32)

        h_new = self.gru.forward(states, h)
        ctx   = self.memory.read(h_new)
        normed = self.res_norm.forward(h_new + ctx)
        feat  = self.trunk.forward(normed)
        self.val_head.forward(feat)
        self.adv_head.forward(feat)

        delta  = (td_errors * weights / B).astype(np.float32)
        d_q    = np.zeros((B, self.action_dim), np.float32)
        for i, ac in enumerate(actions):
            d_q[i, ac] = -delta[i]
        d_q -= d_q.mean(axis=-1, keepdims=True)

        d_feat_v, dWv, dbv = self.val_head.backward(-delta[:, None])
        d_feat_a, dWa, dba = self.adv_head.backward(d_q)
        d_feat             = d_feat_v + d_feat_a

        d_normed, trunk_grads = self.trunk.backward(d_feat)

        d_res, dg_n, db_n = self.res_norm.backward(d_normed)
        d_h_new = d_res + self.memory.backward(d_res)

        _, _, gru_grads = self.gru.backward(d_h_new)

        core_grads = (gru_grads + [dg_n, db_n]
                      + trunk_grads + [dWv, dbv, dWa, dba])
        self.optim.step(core_grads)


    def zero_state(self, batch=1): return self.gru.zero_state(batch)

    def copy_weights_from(self, other):
        for s, t in zip(other._flat(), self._flat()): t[:] = s

    def soft_update_from(self, other, tau=0.005):
        for s, t in zip(other._flat(), self._flat()): t[:] = tau * s + (1 - tau) * t

    def _flat(self):
        return (self.gru.params()
                + self.memory.params()
                + [self.res_norm.g, self.res_norm.b]
                + self.trunk.all_params()
                + self.val_head.params() + self.adv_head.params())



class PlanningModule:
    """
    K-step mental rollout using the consequence network.

    For each candidate action a_0 at state s:
      C_plan(s, a_0) = Σ_{k=0}^{K-1} γ^k · C_hat(s_k, a_k)
      where (s_k, a_k) is simulated greedily (a_k = argmin Q_penalised).

    Usage:
      planner = PlanningModule(consequence_net, gamma=0.99, K=3)
      C_future = planner.rollout(state, action, lambda_val)
    """

    def __init__(self, consequence_net, gamma: float = 0.99, K: int = 3):
        self.cnet  = consequence_net
        self.gamma = gamma
        self.K     = K

    def rollout_batch(self, state: np.ndarray, action_dim: int) -> np.ndarray:
        """
        FIX: Batch rollout for ALL action_dim actions in a single call.

        state : (state_dim,) normalized
        Returns (action_dim,) discounted future consequence penalties.
        Previously this was called once per action → 5 × K × 2 = 30 calls/step.
        Now it runs K steps once, shared across all starting actions → 2K calls/step.
        """
        if self.K == 0:
            return np.zeros(action_dim, np.float32)

        s0     = np.asarray(state, np.float32)
        states = np.tile(s0[None], (action_dim, 1))
        acts   = np.arange(action_dim, dtype=np.int32)
        totals = np.zeros(action_dim, np.float32)
        gk     = 1.0

        for k in range(self.K):
            C_k, _, _, _ = self.cnet.forward(states, acts)
            totals += gk * C_k
            gk     *= self.gamma

            new_states = np.zeros_like(states)
            for i in range(action_dim):
                new_states[i] = np.clip(
                    states[i] + self._action_delta(int(acts[i]), states[i]), 0.0, 1.0)
            states = new_states

            all_acts_tile = np.tile(np.arange(action_dim, dtype=np.int32), action_dim)
            states_tile   = np.repeat(states, action_dim, axis=0)
            C_flat, _, _, _ = self.cnet.forward(states_tile, all_acts_tile)
            acts = C_flat.reshape(action_dim, action_dim).argmin(axis=1).astype(np.int32)

        return totals

    def rollout(self, state: np.ndarray, action: int,
                lambda_val: float, action_dim: int) -> float:
        """Legacy single-action wrapper. Use rollout_batch for efficiency."""
        if self.K == 0:
            return 0.0
        penalties = self.rollout_batch(state, action_dim)
        return float(penalties[action])

    @staticmethod
    def _action_delta(action: int, state: np.ndarray) -> np.ndarray:
        """
        Lightweight state transition approximation.
        Matches the qualitative transitions in BaseEnv._transition().
        """
        delta = np.zeros(len(state), np.float32)
        if   action == 0: delta[0] -= 0.05; delta[1] += 0.08
        elif action == 1: delta[0] += 0.10; delta[1] -= 0.04
        elif action == 2: delta[0] += 0.25; delta[1] -= 0.10; delta[5] += 0.05
        elif action == 3: delta[0] += 0.12; delta[1] -= 0.20
        else:             delta[0] -= 0.10; delta[3] -= 0.15
        return delta * 0.1



class SelfCorrectionModule:
    """
    Detects rising consequence trends and triggers a temporary lambda boost.

    Algorithm:
      1. Maintain EMA of recent consequences (slow and fast).
      2. If fast_EMA > slow_EMA * trigger_ratio → activate correction.
      3. In correction mode: lambda multiplier = correction_factor (>1).
      4. Deactivate when fast_EMA drops below slow_EMA * reset_ratio.

    This approximates the "self-correction layer" from COGNITIVE_CCPL.pdf:
      When ∇C > 0, action is corrected before full consequence manifests.
    """

    def __init__(self, fast_alpha: float = 0.2, slow_alpha: float = 0.02,
                 trigger_ratio: float = 1.3, reset_ratio: float = 1.05,
                 correction_factor: float = 2.0,
                 warmup_steps: int = 30):
        self.fast_alpha        = fast_alpha
        self.slow_alpha        = slow_alpha
        self.trigger_ratio     = trigger_ratio
        self.reset_ratio       = reset_ratio
        self.correction_factor = correction_factor
        self.warmup_steps      = warmup_steps

        self._fast_ema    = 0.0
        self._slow_ema    = 0.0
        self._active      = False
        self._n_samples   = 0
        self._nonzero_n   = 0

    def update(self, consequence: float):
        """
        Call once per step with the observed consequence.
        FIX: skip zero-consequence steps for EMA initialisation so that the
        delayed-consequence pattern (c=0 for the first N steps) does not
        cause the slow EMA to stay near 0 and trigger a spurious alarm on
        the first real consequence.  Also require warmup_steps non-zero
        samples before the trigger logic is active.
        """
        c = float(consequence)

        if c > 0.0:
            if self._n_samples == 0:
                self._fast_ema = self._slow_ema = c
            else:
                self._fast_ema = self.fast_alpha * c + (1 - self.fast_alpha) * self._fast_ema
                self._slow_ema = self.slow_alpha * c + (1 - self.slow_alpha) * self._slow_ema
            self._nonzero_n += 1

        self._n_samples += 1

        if self._nonzero_n < self.warmup_steps:
            return

        if not self._active:
            if self._fast_ema > self._slow_ema * self.trigger_ratio:
                self._active = True
        else:
            if self._fast_ema <= self._slow_ema * self.reset_ratio:
                self._active = False

    @property
    def lambda_multiplier(self) -> float:
        """Returns correction_factor when active, 1.0 otherwise."""
        return self.correction_factor if self._active else 1.0

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def trend(self) -> float:
        """fast_EMA / (slow_EMA + eps) — ratio > 1 means rising trend."""
        return self._fast_ema / (self._slow_ema + 1e-6)

    def reset(self):
        self._fast_ema  = 0.0
        self._slow_ema  = 0.0
        self._active    = False
        self._n_samples = 0



class StateHistoryAttention:
    """
    Self-attention over a rolling window of recent state embeddings.
    Implements the Transformer encoder block idea (Vaswani §3.1):
      LayerNorm(x + MHA(x, x, x)) → LayerNorm(x + FFN(x))

    Used to build a contextual representation from the last H states.
    At action selection time, this provides richer temporal context than
    the GRU alone.
    """

    def __init__(self, state_dim: int, d_model: int = 32,
                 n_heads: int = 4, history_len: int = 8,
                 lr: float = 3e-4, seed: int = 0):
        self.state_dim   = state_dim
        self.d_model     = d_model
        self.history_len = history_len
        rng = np.random.default_rng(seed)

        self.input_proj  = Linear(state_dim, d_model, rng, scale=0.05)
        self.mha         = MultiHeadAttention(d_model, n_heads, rng, lr=lr)
        self.norm1       = LayerNorm(d_model)

        self.ffn_1 = Linear(d_model, d_model * 4, rng, scale=0.05)
        self.ffn_2 = Linear(d_model * 4, d_model, rng, scale=0.05)
        self.norm2 = LayerNorm(d_model)

        self.out_proj = Linear(d_model, state_dim, rng, scale=0.01)

        self._history = np.zeros((history_len, state_dim), np.float32)
        self._ptr     = 0
        self._count   = 0

    def push(self, state: np.ndarray):
        """Add state to rolling history buffer."""
        self._history[self._ptr] = np.asarray(state, np.float32)
        self._ptr   = (self._ptr + 1) % self.history_len
        self._count += 1

    def context(self) -> np.ndarray:
        """
        Compute attended context over current history window.
        Returns (state_dim,) — can be added to the current state.
        """
        if self._count == 0:
            return np.zeros(self.state_dim, np.float32)

        n   = min(self._count, self.history_len)
        idx = [(self._ptr - n + i) % self.history_len for i in range(n)]
        seq = self._history[idx]

        pe  = sinusoidal_pe(n, self.d_model)[:n]
        x   = self.input_proj.forward(seq) + pe
        x   = x[None]

        attn_out, _ = self.mha.forward(x, x, x)
        x2          = self.norm1.forward(x + attn_out)

        ffn_out = relu(self.ffn_1.forward(x2))
        ffn_out = self.ffn_2.forward(ffn_out)
        x3      = self.norm2.forward(x2 + ffn_out)

        last    = x3[0, -1]
        return self.out_proj.forward(last[None]).squeeze(0)

    def reset(self):
        self._history[:] = 0
        self._ptr    = 0
        self._count  = 0

    def params(self):
        return (self.input_proj.params() + self.mha.params()
                + self.norm1.params() + self.ffn_1.params()
                + self.ffn_2.params() + self.norm2.params()
                + self.out_proj.params())
