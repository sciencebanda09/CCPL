"""Legacy delay utilities.

The active CCPL agent uses :mod:`delay_bellman`.  These compatibility classes
remain importable, but delayed consequences are accepted only with an explicit
source timestamp; guessing a source action from temporal proximity is not a
valid causal label.
"""

import numpy as np

try:
    from .networks import Adam, MLP, softmax
except ImportError:  # Legacy checkout imports.
    from networks import Adam, MLP, softmax


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Delay Estimator Network
# ─────────────────────────────────────────────────────────────────────────────

class DelayEstimatorNet:
    """
    Maps GRU hidden state h_t → P(τ | h_t) over support {0, 1, …, τ_max}.

    Architecture:
      MLP(gru_dim → hidden_dim → hidden_dim → τ_max+1) → softmax

    Training target: observed consequence arrives at step t+τ_obs.
    We treat τ_obs as a one-hot supervision signal and train with cross-entropy.
    When τ_obs is unknown (sparse), we skip that sample or use the prior.
    """

    def __init__(self, gru_dim: int = 40, hidden_dim: int = 32,
                 tau_max: int = 15, lr: float = 3e-4, seed: int = 42):
        if min(gru_dim, hidden_dim) <= 0 or tau_max < 0:
            raise ValueError("network dimensions must be positive and tau_max non-negative")
        self.tau_max   = tau_max
        self.n_classes = tau_max + 1
        rng = np.random.default_rng(seed)
        self.net   = MLP([gru_dim, hidden_dim, hidden_dim, self.n_classes], rng, scale=0.05)
        self.optim = Adam(self.net.all_params(), lr=lr)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, hidden: np.ndarray) -> np.ndarray:
        """
        hidden: (B, gru_dim) or (gru_dim,)
        Returns P(τ | h): (B, n_classes) or (n_classes,)
        """
        h = np.asarray(hidden, np.float32)
        scalar = h.ndim == 1
        if scalar:
            h = h[None]
        logits = self.net.forward(h)
        probs  = softmax(logits)
        return probs[0] if scalar else probs

    def expected_delay(self, hidden: np.ndarray) -> np.ndarray:
        """E[τ | h_t] = Σ_τ τ · P(τ | h_t). Shape: (B,) or scalar."""
        probs  = self.forward(hidden)
        taus   = np.arange(self.n_classes, dtype=np.float32)
        scalar = probs.ndim == 1
        if scalar:
            return float((probs * taus).sum())
        return (probs * taus[None, :]).sum(axis=-1)

    def delay_variance(self, hidden: np.ndarray) -> np.ndarray:
        """Var[τ | h_t] — proxy for uncertainty in delay estimate."""
        probs  = self.forward(hidden)
        taus   = np.arange(self.n_classes, dtype=np.float32)
        scalar = probs.ndim == 1
        if scalar:
            mu  = float((probs * taus).sum())
            return float((probs * (taus - mu)**2).sum())
        mu = (probs * taus[None, :]).sum(axis=-1, keepdims=True)
        return (probs * (taus[None, :] - mu)**2).sum(axis=-1)

    # ── Training ──────────────────────────────────────────────────────────────

    def update(self, hiddens: np.ndarray, observed_delays: np.ndarray,
               weights: np.ndarray = None) -> float:
        """
        Cross-entropy update on observed (hidden, delay) pairs.

        hiddens:         (B, gru_dim)
        observed_delays: (B,) int — actual observed delay τ for each sample
                         Use -1 to mark unknown/sparse; those rows are skipped.
        weights:         (B,) importance weights, default ones.
        Returns mean cross-entropy loss.
        """
        observed_delays = np.asarray(observed_delays)
        mask = ((observed_delays >= 0)
                & (observed_delays < self.n_classes))
        if mask.sum() == 0:
            return 0.0

        h_sel  = hiddens[mask]
        tau_sel = observed_delays[mask].astype(int)
        w_sel  = (weights[mask] if weights is not None
                  else np.ones(mask.sum(), np.float32))
        if np.any(w_sel < 0) or not np.all(np.isfinite(w_sel)):
            raise ValueError("delay-estimator weights must be finite and non-negative")
        w_norm = w_sel / (w_sel.sum() + 1e-8)

        logits = self.net.forward(h_sel)
        probs  = softmax(logits)

        d_logits = probs.copy()
        d_logits[np.arange(len(tau_sel)), tau_sel] -= 1.0
        d_logits *= w_norm[:, None]

        loss = float(-np.sum(
            w_norm * np.log(probs[np.arange(len(tau_sel)), tau_sel] + 1e-8)))
        _, grads = self.net.backward(d_logits)
        self.optim.step(grads)
        return loss

    def params(self):
        return self.net.all_params()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Temporal Consequence Estimator
# ─────────────────────────────────────────────────────────────────────────────

class TemporalConsequenceEstimator:
    """
    Replaces single fixed-delay consequence target with:

      E_τ[C_{t+τ}] = Σ_τ P(τ | h_t) · C_hat(s_t, a_t, τ)

    Each τ-slice is estimated by the multi-horizon consequence net heads
    mapped onto the discrete support via linear interpolation of short/mid/long
    horizon predictions.

    Usage:
      tce = TemporalConsequenceEstimator(delay_net, consequence_net,
                                         tau_max=15, tau_short=3,
                                         tau_mid=8, tau_long=15)
      expected_C, uncertainty = tce.forward(states, actions, hiddens)
    """

    def __init__(self, delay_net: DelayEstimatorNet,
                 consequence_net,          # MultiHorizonConsequenceNet
                 tau_max:   int   = 15,
                 tau_short: float = 3.0,
                 tau_mid:   float = 8.0,
                 tau_long:  float = 15.0):
        self.delay_net       = delay_net
        self.consequence_net = consequence_net
        self.tau_max         = tau_max
        self.taus            = np.arange(tau_max + 1, dtype=np.float32)
        self.tau_short       = tau_short
        self.tau_mid         = tau_mid
        self.tau_long        = tau_long

    def _interpolate_consequence(self, C_short, C_mid, C_long,
                                  tau: float) -> float:
        if tau <= self.tau_short:
            return float(C_short)
        elif tau <= self.tau_mid:
            t = (tau - self.tau_short) / (self.tau_mid - self.tau_short + 1e-6)
            return float((1 - t) * C_short + t * C_mid)
        else:
            t = (tau - self.tau_mid) / (self.tau_long - self.tau_mid + 1e-6)
            t = min(t, 1.0)
            return float((1 - t) * C_mid + t * C_long)

    def forward(self, states: np.ndarray, actions: np.ndarray,
                hiddens: np.ndarray):
        """
        Compute E_τ[C_{t+τ}] and associated uncertainty.

        states:  (B, state_dim)
        actions: (B,) int
        hiddens: (B, gru_dim)

        Returns:
          expected_C:  (B,) — consequence expectation under inferred delay dist
          uncertainty: (B,) — std of consequence under delay distribution
        """
        B = len(actions)

        delay_probs = self.delay_net.forward(hiddens)  # (B, n_classes)

        C_tot, C_short, C_mid, C_long, sigma = self.consequence_net.forward(
            states, actions)

        C_tau = np.zeros((B, self.tau_max + 1), np.float32)
        for tau_i, tau_val in enumerate(self.taus):
            for b in range(B):
                C_tau[b, tau_i] = self._interpolate_consequence(
                    C_short[b], C_mid[b], C_long[b], tau_val)

        expected_C = (delay_probs * C_tau).sum(axis=-1)

        diff2      = (C_tau - expected_C[:, None])**2
        var_C      = (delay_probs * diff2).sum(axis=-1)
        uncertainty = np.sqrt(var_C + sigma**2 + 1e-6)

        return expected_C, uncertainty


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Sparse Consequence Store
# ─────────────────────────────────────────────────────────────────────────────

class SparseConsequenceStore:
    """
    Stores (state, action, hidden, timestamp) tuples from the environment.
    Accepts consequence labels that may arrive:
      - At a future timestamp (delayed)
      - Not at all (missing → imputed)
      - Noisily (noisy proxy → downweighted)

    After alignment, yields (state, action, consequence, confidence, delay)
    tuples for training the delay estimator and consequence net.

    Temporal alignment is performed only from an explicit ``source_t`` or
    ``delay`` supplied by the environment.  Missing labels stay missing; they
    are never imputed as causal supervision.

    Bug fix (FIX-V6-7):
      flush_old_observations(cutoff_t) added — removes observations with
      t < cutoff_t.  Without this, _observations grew unbounded and _align()
      became O(observations × slots) which caused the ETA blowup in the logs.
    """

    def __init__(self, tau_max: int = 15, state_dim: int = 6,
                 gru_dim: int = 40, capacity: int = 2000,
                 impute_confidence: float = 0.10,
                 seed: int = 42):
        if tau_max < 0 or min(state_dim, gru_dim, capacity) <= 0:
            raise ValueError("dimensions/capacity must be positive and tau_max non-negative")
        self.tau_max            = tau_max
        self.state_dim          = state_dim
        self.gru_dim            = gru_dim
        self.capacity           = capacity
        self.impute_confidence  = impute_confidence
        self.rng                = np.random.default_rng(seed)

        self._slots:        list = []
        self._observations: list = []
        self._running_sum   = 0.0
        self._running_count = 0
        self._ready:        list = []

    # ── Recording ─────────────────────────────────────────────────────────────

    def record(self, state: np.ndarray, action: int,
               hidden: np.ndarray, t: int):
        """Call once per step when action is taken."""
        slot = {
            "t":          t,
            "state":      np.asarray(state,  np.float32).copy(),
            "action":     int(action),
            "hidden":     np.asarray(hidden, np.float32).flatten().copy(),
            "consequence":None,
            "confidence": 0.0,
            "delay":      -1,
            "aligned":    False,
        }
        self._slots.append(slot)
        if len(self._slots) > self.capacity:
            self._slots.pop(0)

    def observe_consequence(self, consequence: float, t: int,
                             noise_proxy: float = 0.0,
                             source_t: int | None = None,
                             delay: int | None = None):
        """
        Register a consequence at observation step ``t`` and attach it to its
        explicit source transition.
        noise_proxy ≥ 0; higher = noisier signal → lower confidence.
        """
        if source_t is None and delay is None:
            raise ValueError(
                "Delayed feedback needs source_t or delay; temporal proximity "
                "is not sufficient for attribution")
        if source_t is not None and delay is not None:
            if int(source_t) != int(t) - int(delay):
                raise ValueError("source_t and delay describe different transitions")
        if delay is not None:
            if not 0 <= int(delay) <= self.tau_max:
                raise ValueError("delay outside configured support")
            source_t = int(t) - int(delay)
        source_t = int(source_t)
        observed_delay = int(t) - source_t
        if not 0 <= observed_delay <= self.tau_max:
            raise ValueError("source timestamp outside configured delay support")

        conf = float(np.exp(-np.clip(noise_proxy, 0, 5)))
        self._observations.append((int(t), float(consequence), conf, source_t))
        self._running_sum   += abs(float(consequence))
        self._running_count += 1

        matches = [slot for slot in self._slots
                   if slot["t"] == source_t and not slot["aligned"]]
        if not matches:
            raise KeyError(f"No pending transition recorded at source_t={source_t}")
        slot = matches[-1]
        slot["consequence"] = float(consequence)
        slot["confidence"] = conf
        slot["delay"] = observed_delay
        slot["aligned"] = True
        self._ready.append(slot.copy())

    # ── FIX-V6-7: bounded observation cache ───────────────────────────────────

    def flush_old_observations(self, cutoff_t: int):
        """
        Remove observations with timestamp < cutoff_t.

        FIX-V6-7: Without this, _observations grows by ~100 entries per
        episode (one per step), and _align() iterates all observations × all
        slots — O(n²) — causing the ETA to blowup from 8m to 2h+ by ep 400.

        Call from CCPLv6Agent.episode_end() with:
            cutoff = self._store_step - self.sparse_store.tau_max - 1
        """
        self._observations = [item for item in self._observations
                              if item[0] >= cutoff_t]

    # ── Alignment ─────────────────────────────────────────────────────────────

    def _align(self):
        """Compatibility no-op: observations are aligned explicitly on input."""

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get_aligned_batch(self, min_batch: int = 32) -> dict | None:
        """
        Returns a batch of aligned samples for training.
        None if fewer than min_batch ready.
        """
        self._align()
        if len(self._ready) < min_batch:
            return None

        idxs  = self.rng.choice(len(self._ready), min_batch, replace=False)
        batch = [self._ready[i] for i in idxs]

        return {
            "states":       np.stack([b["state"]  for b in batch]),
            "actions":      np.array([b["action"] for b in batch], np.int32),
            "hiddens":      np.stack([b["hidden"] for b in batch]),
            "consequences": np.array([b["consequence"] for b in batch], np.float32),
            "confidences":  np.array([b["confidence"]  for b in batch], np.float32),
            "delays":       np.array([b["delay"]        for b in batch], np.int32),
            "timestamps":   np.array([b["t"]            for b in batch], np.int32),
        }

    def flush_ready(self):
        """Clear the aligned pool (e.g., after each epoch)."""
        self._ready.clear()

    def flush_observations(self):
        """Clear observation cache (e.g., after each episode)."""
        self._observations.clear()

    def size(self):
        return len(self._slots)

    def ready_size(self):
        self._align()
        return len(self._ready)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Delay-aware consequence update helper
# ─────────────────────────────────────────────────────────────────────────────

def update_delay_and_consequence(
    delay_net:       DelayEstimatorNet,
    consequence_net,
    sparse_store:    SparseConsequenceStore,
    min_batch:       int = 32,
) -> dict:
    """
    Pulls an aligned batch from the sparse store and updates:
      1. The delay estimator (cross-entropy on observed delays).
      2. The consequence net (weighted MSE using confidence as sample weight).

    Returns a diagnostics dict.
    """
    batch = sparse_store.get_aligned_batch(min_batch)
    if batch is None:
        return {"delay_loss": 0.0, "consequence_loss": 0.0, "n_aligned": 0}

    S    = batch["states"]
    A    = batch["actions"]
    H    = batch["hiddens"]
    C    = batch["consequences"]
    conf = batch["confidences"]
    tau  = batch["delays"]

    d_loss = delay_net.update(H, tau, weights=conf)
    c_loss = consequence_net.update_step(S, A, C, weights=conf)

    return {
        "delay_loss":       round(d_loss, 5),
        "consequence_loss": round(c_loss, 5),
        "n_aligned":        len(S),
        "mean_confidence":  round(float(conf.mean()), 4),
        "mean_delay":       round(float(tau[tau >= 0].mean()
                                       if (tau >= 0).any() else 0), 2),
    }
