"""Learn an observation-delay distribution and form a consistent TD target.

Delayed consequence feedback does not alter the temporal distance from
``s_t`` to ``s_{t+1}``, so the reward critic keeps the ordinary one-step
bootstrap ``gamma Q(s_{t+1}, a*)``.  The learned factor
``E[gamma**tau | h_t]`` is applied only to a consequence attributed to the
current action.  The standard reward Bellman operator therefore has modulus
``gamma``; delay-distribution checks below are numerical invariants, not a new
convergence theorem.
"""

import numpy as np
try:
    from .networks import Adam, sigmoid, softmax, MLP, Linear, softplus
except ImportError:
    from networks import Adam, sigmoid, softmax, MLP, Linear, softplus



class DelayDistributionNet:
    """
    Learns P(τ | h_t) over delay values τ ∈ {0..τ_max}.

    Input:  GRU hidden state h_t  (gru_dim,)  — encodes recent trajectory
    Output: Categorical distribution over τ values  (τ_max + 1,)

    Unlike the original DelayEstimatorNet which predicts E[τ], this network
    predicts the FULL DISTRIBUTION, enabling the weighted Bellman sum.

    Architecture: MLP → softmax → Categorical(τ_max + 1 categories)
    """

    def __init__(self, gru_dim: int = 40, hidden_dim: int = 32,
                 tau_max: int = 15, lr: float = 3e-4, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.tau_max    = tau_max
        self.gru_dim    = gru_dim
        self.tau_values = np.arange(0, tau_max + 1, dtype=np.float32)

        self.net   = MLP([gru_dim, hidden_dim, hidden_dim, tau_max + 1], rng)
        self.optim = Adam(self.net.all_params(), lr=lr)

        self._last_logits = None

    def forward(self, h: np.ndarray) -> np.ndarray:
        """
        h: (B, gru_dim) or (gru_dim,)
        Returns: (B, tau_max + 1) probability distribution over τ values
        """
        h = np.asarray(h, np.float32)
        scalar = h.ndim == 1
        if scalar: h = h[None]
        logits = self.net.forward(h)
        self._last_logits = logits
        return softmax(logits)

    def expected_tau(self, h: np.ndarray) -> float:
        """E[τ|h] = Σ_k k · P(τ=k|h)"""
        probs = self.forward(h)
        return float((probs * self.tau_values[None]).sum(-1).mean())

    def update_step(self, h_batch: np.ndarray,
                    observed_tau: np.ndarray,
                    weights: np.ndarray) -> float:
        """
        Supervised update: cross-entropy loss on observed delay values.
        observed_tau: (B,) integer delay observations ∈ {0..tau_max}
        """
        h_batch      = np.asarray(h_batch,      np.float32)
        observed_tau = np.asarray(observed_tau, np.int32)
        weights      = np.asarray(weights,      np.float32)
        B            = len(observed_tau)
        if B == 0:
            return 0.0
        if (np.any(observed_tau < 0)
                or np.any(observed_tau > self.tau_max)):
            raise ValueError("observed_tau outside configured delay support")
        if np.any(weights < 0) or not np.all(np.isfinite(weights)):
            raise ValueError("delay weights must be finite and non-negative")
        weight_norm = weights / (weights.sum() + 1e-8)

        probs  = self.forward(h_batch)
        tau_idx = observed_tau
        log_p   = np.log(probs[np.arange(B), tau_idx] + 1e-8)
        loss    = float(-np.sum(log_p * weight_norm))

        d_logits = probs.copy()
        d_logits[np.arange(B), tau_idx] -= 1.0
        d_logits *= weight_norm[:, None]

        _, grads = self.net.backward(d_logits)
        self.optim.step(grads)
        return loss

    def params(self): return self.net.all_params()



class DelayCorrectedBellman:
    """
    Computes a one-step reward target with delay-discounted cost attribution:

        ŷ_t = r_t + γ Q_target(s_{t+1}, a*_{t+1})
                  - λ(s_t) E[γ^τ | h_t] ΔC(s_t, a_t, h_t)

    Delayed observation does not change the temporal distance to the next
    reward state, so the reward bootstrap retains the ordinary discount γ.
    ``gamma_eff`` only discounts a cost estimate attributed to an action whose
    observed consequence arrives τ steps later.
    """

    def __init__(self, delay_net: DelayDistributionNet,
                 gamma: float = 0.99, tau_max: int = 15):
        self.delay_net = delay_net
        self.gamma     = gamma
        self.tau_max   = tau_max
        self.tau_vals  = np.arange(0, tau_max + 1, dtype=np.float32)
        self.gamma_pows = np.array([gamma**k for k in range(0, tau_max + 1)],
                                    dtype=np.float32)

    def gamma_eff(self, h_batch: np.ndarray) -> np.ndarray:
        """
        Compute effective discount γ_eff(h) = Σ_k P(τ=k|h) · γ^k
        h_batch: (B, gru_dim)
        Returns: (B,) effective discounts
        """
        probs = self.delay_net.forward(h_batch)
        return (probs * self.gamma_pows[None]).sum(-1).astype(np.float32)

    def operator_modulus(self, h_batch: np.ndarray) -> dict:
        """Return the exact pointwise modulus of the consequence operator."""
        values = self.gamma_eff(h_batch)
        return {
            "min": float(values.min()),
            "max": float(values.max()),
            "strict": bool(np.all(values < 1.0 - 1e-7)),
            "support_lower": float(self.gamma ** self.tau_max),
            "support_upper": 1.0,
        }

    def apply_consequence_operator(
        self, immediate: np.ndarray, continuation: np.ndarray,
        lam: np.ndarray, delta_C: np.ndarray, h_batch: np.ndarray,
    ) -> np.ndarray:
        """Apply the learned delayed-cost operator to a batch of values."""
        immediate = np.asarray(immediate, np.float32)
        continuation = np.asarray(continuation, np.float32)
        lam = np.asarray(lam, np.float32)
        delta_C = np.asarray(delta_C, np.float32)
        values = self.gamma_eff(h_batch)
        if not (immediate.shape == continuation.shape == lam.shape == delta_C.shape == values.shape):
            raise ValueError("operator inputs must have identical batch shape")
        return immediate + self.gamma * continuation - lam * values * np.maximum(delta_C, 0.0)

    def td_target(self, rewards: np.ndarray,
                  next_q: np.ndarray,
                  delta_C: np.ndarray,
                  lam: np.ndarray,
                  dones: np.ndarray,
                  h_batch: np.ndarray,
                  penalty_scale: float = 1.0) -> np.ndarray:
        """
        One-step reward target with delay-discounted cost attribution.

        rewards       : (B,)
        next_q        : (B,) — Q_target(s_{t+1}, argmax Q_online(s_{t+1}))
        delta_C       : (B,) — causal attribution ΔC(s, a, h)
        lam           : (B,) — λ(s_t)
        dones         : (B,)
        h_batch       : (B, gru_dim)
        penalty_scale : float — must match the action-selection penalty scale
        """
        gamma_e = self.gamma_eff(h_batch)
        target  = (rewards
                   + self.gamma * next_q * (1.0 - dones)
                   - lam * penalty_scale * gamma_e
                   * np.clip(delta_C, 0.0, None))
        return target.astype(np.float32), gamma_e

    def verify_contraction(self, h_sample: np.ndarray) -> dict:
        """
        Numerical support/bounds check.  The one-step Bellman bootstrap is a
        contraction because 0 <= gamma < 1; this is not an empirical theorem.
        """
        probs     = self.delay_net.forward(h_sample)
        gamma_e   = (probs * self.gamma_pows[None]).sum(-1)
        gamma_min = float(self.gamma_pows.min())
        lower = float(self.gamma_pows.min())
        upper = float(self.gamma_pows.max())
        valid_distribution = bool(
            np.all(np.isfinite(probs))
            and np.all(probs >= -1e-7)
            and np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
        )
        delay_bounds_satisfied = bool(
            valid_distribution
            and np.all(gamma_e >= lower - 1e-6)
            and np.all(gamma_e <= upper + 1e-6)
        )

        return {
            "gamma":          self.gamma,
            "gamma_eff_mean": float(gamma_e.mean()),
            "gamma_eff_std":  float(gamma_e.std()),
            "gamma_eff_min":  float(gamma_e.min()),
            "gamma_eff_max":  float(gamma_e.max()),
            "gamma_1":        float(self.gamma),
            "gamma_tau_max":  gamma_min,
            "delay_distribution_valid": valid_distribution,
            "delay_bounds_satisfied": delay_bounds_satisfied,
            "contraction_modulus": float(self.gamma),
            "contraction_satisfied": bool(0.0 <= self.gamma < 1.0),
            "proof": (
                f"Standard one-step Bellman modulus γ={self.gamma:.4f}; "
                f"delay factor observed in [{float(gamma_e.min()):.4f}, "
                f"{float(gamma_e.max()):.4f}] and must lie in "
                f"[{lower:.4f}, {upper:.4f}]."
            ),
        }
