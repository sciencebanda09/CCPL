"""
CCPL v8 — networks_v8.py
=========================
New cognitive primitives implementing the remaining Cognitive CCPL modules
not present in networks_v7.py:

  1. AbstractionLayer          — Prototype similarity → λ amplification
                                 λ'(s) = λ(s)·(1 + α·proto(s))   [§7]

  2. WorkingMemoryLambdaModifier — (s,a,c)_{t-k:t} modifies λ    [§6.2]

These are used by CCPLv8Agent alongside all attention primitives from
networks_v7.py. No external dependencies beyond numpy required.
"""
import numpy as np



class AbstractionLayer:
    """
    Prototype similarity layer.

    Maintains prototype centroids of historically dangerous states.
    When the current state resembles a known dangerous region, λ is amplified:

        λ'(s) = λ(s) · (1 + α · proto(s))

    where proto(s) ∈ [0, 1] is a danger-weighted Gaussian similarity score
    against stored prototypes.

    Online update rule:
      - New dangerous states (c ≥ danger_threshold) are absorbed.
      - If slots are available: claim a new prototype.
      - If slots are full:  find the nearest centroid and nudge it via EMA.
      - Danger average per centroid is updated with a running mean.
    """

    def __init__(self, state_dim: int, n_prototypes: int = 16,
                 alpha: float = 0.30, danger_threshold: float = 0.15,
                 seed: int = 42):
        self.state_dim        = state_dim
        self.n_prototypes     = n_prototypes
        self.alpha            = alpha
        self.danger_threshold = danger_threshold

        self._centroids  = np.zeros((n_prototypes, state_dim), np.float32)
        self._danger_avg = np.zeros(n_prototypes, np.float32)
        self._counts     = np.zeros(n_prototypes, np.float32)
        self._filled     = 0

    def update(self, state: np.ndarray, consequence: float):
        """
        Absorb a transition into the abstraction layer.
        Ignored if consequence < danger_threshold.
        """
        c = float(consequence)
        if c < self.danger_threshold:
            return

        s = np.asarray(state, np.float32)

        if self._filled < self.n_prototypes:
            idx = self._filled
            self._centroids[idx]  = s
            self._danger_avg[idx] = c
            self._counts[idx]     = 1.0
            self._filled         += 1
        else:
            dists = np.linalg.norm(self._centroids - s[None], axis=1)
            idx   = int(np.argmin(dists))
            n     = self._counts[idx]
            self._centroids[idx]  = 0.9 * self._centroids[idx] + 0.1 * s
            self._danger_avg[idx] = (n * self._danger_avg[idx] + c) / (n + 1)
            self._counts[idx]    += 1

    def similarity(self, state: np.ndarray) -> float:
        """
        Compute proto(s) ∈ [0, 1]: danger-weighted Gaussian similarity.

        For each prototype k:
          sim_k = danger_avg_k · exp(−‖s − centroid_k‖)
        proto(s) = mean(sim_k)
        """
        if self._filled == 0:
            return 0.0
        s     = np.asarray(state, np.float32)
        dists = np.linalg.norm(self._centroids[:self._filled] - s[None], axis=1)
        sim   = self._danger_avg[:self._filled] * np.exp(-dists)
        return float(np.clip(sim.mean(), 0.0, 1.0))

    def lambda_amplification(self, state: np.ndarray) -> float:
        """
        Returns the λ multiplier: (1 + α · proto(s)).
        Always ≥ 1.0 — never reduces λ, only boosts it near danger.
        """
        return 1.0 + self.alpha * self.similarity(state)



class WorkingMemoryLambdaModifier:
    """
    Working memory that modifies λ based on recent consequence trend.

    Implements: "recent hidden sequence (s,a,c)_{t-k:t} modifies lambda" (§6.2).

    A rolling window of recent consequence values is maintained. The modifier
    is computed from the trend between the older and newer halves of the window:

        trend   = mean(newer half) − mean(older half)
        modifier = clip(trend · modifier_scale, −clip, +clip)

    Effect:
      - Rising consequence trend  → positive modifier → λ increases earlier
      - Falling consequence trend → negative modifier → λ relaxes faster

    This is additive: effective_λ = λ(s) · abstraction_amp + wm_modifier
    """

    def __init__(self, window: int = 8, modifier_scale: float = 0.20,
                 clip: float = 0.30):
        self.window         = window
        self.modifier_scale = modifier_scale
        self.clip           = clip
        self._history: list = []

    def push(self, consequence: float):
        """Add a consequence observation to the working memory window."""
        self._history.append(float(consequence))
        if len(self._history) > self.window:
            self._history.pop(0)

    def lambda_modifier(self) -> float:
        """
        Returns an additive modifier for λ based on consequence trend.
        Returns 0.0 if fewer than 2 observations have been collected.
        """
        if len(self._history) < 2:
            return 0.0
        h   = np.array(self._history, np.float32)
        mid = max(len(h) // 2, 1)
        old = h[:mid].mean()
        new = h[mid:].mean()
        return float(np.clip((new - old) * self.modifier_scale,
                              -self.clip, self.clip))

    def reset(self):
        """Clear the working memory window (call at episode start)."""
        self._history.clear()
