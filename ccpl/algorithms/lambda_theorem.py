"""Diagnostics for state-conditioned Lagrange multipliers.

State-dependent multipliers can be a useful policy parameterisation, but cost
heterogeneity alone does not prove that they strictly dominate a scalar dual
variable under one global CMDP constraint.  This module therefore reports
observable heterogeneity and multiplier variation; it does not manufacture a
performance bound from those quantities.
"""

from __future__ import annotations

import numpy as np


class ConsequenceVarianceEstimator:
    """Estimate occupancy-weighted variation in coarse conditional mean cost."""

    def __init__(self, n_bins: int = 20, state_dim: int = 6):
        self.n_bins = max(int(n_bins), 2)
        self.state_dim = int(state_dim)
        self._bin_sum = np.zeros(self.n_bins, np.float64)
        self._bin_count = np.zeros(self.n_bins, np.int64)
        self._global_n = 0
        self._global_mean = 0.0
        self._global_M2 = 0.0

    def _state_bin(self, state: np.ndarray) -> int:
        state = np.asarray(state, np.float64).reshape(-1)
        if state.size == 0:
            return 0
        # Use a stable low-dimensional risk projection without assuming a
        # six-coordinate state (Safety Gymnasium observations differ by task).
        indices = sorted(set((0, min(1, state.size - 1), min(5, state.size - 1))))
        risk = float(np.mean(np.clip(state[indices], 0.0, 1.0)))
        return min(int(risk * self.n_bins), self.n_bins - 1)

    def update(self, state: np.ndarray, consequence: float) -> None:
        consequence = float(consequence)
        if not np.isfinite(consequence):
            return
        index = self._state_bin(state)
        self._bin_sum[index] += consequence
        self._bin_count[index] += 1

        self._global_n += 1
        delta = consequence - self._global_mean
        self._global_mean += delta / self._global_n
        self._global_M2 += delta * (consequence - self._global_mean)

    def mean_per_state(self) -> np.ndarray:
        means = np.full(self.n_bins, np.nan, np.float64)
        occupied = self._bin_count > 0
        means[occupied] = self._bin_sum[occupied] / self._bin_count[occupied]
        return means

    def variance_across_states(self) -> float:
        """Occupancy-weighted variance of estimated E[c | coarse state bin]."""
        occupied = self._bin_count > 0
        if np.count_nonzero(occupied) < 2:
            return 0.0
        counts = self._bin_count[occupied].astype(np.float64)
        means = self._bin_sum[occupied] / counts
        weights = counts / counts.sum()
        centre = float(np.dot(weights, means))
        return float(np.dot(weights, (means - centre) ** 2))

    def global_variance(self) -> float:
        if self._global_n < 2:
            return 0.0
        return float(self._global_M2 / (self._global_n - 1))

    def theorem2_bound(self, lambda_g: float, d: float) -> dict:
        """Compatibility API returning diagnostics, not an invalid bound."""
        variance = self.variance_across_states()
        return {
            "var_s": round(variance, 6),
            "lambda_g": round(float(lambda_g), 4),
            "d": float(d),
            "epsilon_bound": None,
            "bound_valid": False,
            "condition_met": variance > 1e-4,
            "message": (
                "Cost heterogeneity is measurable, but it does not imply a "
                "strict state-lambda performance bound. Compare matched, "
                "independent-seed ablations instead."
            ),
        }


class AdaptiveLambdaWithDominanceTracking:
    """Track cost heterogeneity and how strongly lambda varies by state."""

    def __init__(
        self,
        lambda_net,
        consequence_var: ConsequenceVarianceEstimator,
        constraint_d: float = 3.0,
    ):
        self.lambda_net = lambda_net
        self.var_estimator = consequence_var
        self.d = float(constraint_d)
        self._lambda_history: list[float] = []
        self._MAX_HIST = 5000

    def dominance_score(self) -> float:
        """Coefficient of variation; this measures variation, not dominance."""
        if len(self._lambda_history) < 10:
            return 0.0
        values = np.asarray(self._lambda_history[-2000:], np.float32)
        mean = float(values.mean())
        return 0.0 if abs(mean) < 1e-6 else float(values.std() / abs(mean))

    def record(
        self,
        state: np.ndarray,
        consequence: float,
        lambda_state: np.ndarray | None = None,
    ) -> None:
        lambda_input = state if lambda_state is None else lambda_state
        value = float(self.lambda_net.forward(lambda_input))
        self.var_estimator.update(state, consequence)
        self._lambda_history.append(value)
        if len(self._lambda_history) > self._MAX_HIST:
            self._lambda_history.pop(0)

    def theorem2_status(self, lambda_global_mean: float) -> dict:
        """Compatibility name for the state-conditioning diagnostic."""
        result = self.var_estimator.theorem2_bound(lambda_global_mean, self.d)
        variation = self.dominance_score()
        return {
            **result,
            "dominance_score": round(variation, 4),
            "state_variation_score": round(variation, 4),
            "state_conditioning_active": variation > 0.15,
            "exploiting_theorem2": False,
            "diagnostic_only": True,
        }


STATE_LAMBDA_NOTE = """
STATE-CONDITIONED LAMBDA — DIAGNOSTIC NOTE

The previous strict-dominance claim is withdrawn.  Under a CMDP with one
global expected-cost constraint, the associated Lagrange multiplier is a
scalar.  A state-conditioned penalty is a different policy parameterisation
unless statewise constraints or statewise budgets are explicitly defined.

Var_s(E[c|s]) > 0 only demonstrates heterogeneous costs.  It does not imply a
positive gap in J_c, and Lipschitz costs do not supply the dual curvature bound
needed by the previous proof.  Effectiveness must therefore be established by
a matched state-lambda versus scalar-lambda experiment using independent seeds
and uncertainty intervals.
"""


def print_theorem2() -> None:
    """Print the correction while preserving the historical entry point."""
    print(STATE_LAMBDA_NOTE)
