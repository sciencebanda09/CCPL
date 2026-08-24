"""Bounded, conservative handling of uncertain ICN consequence estimates.

The original implementation multiplied safety penalties by confidence and an
in-distribution score.  That made an uncertain or novel state *less* safe: the
penalty approached zero exactly when the estimator was least trustworthy.
This module keeps clipping consistent between training and inference, but uses
uncertainty/OOD scores as bounded risk multipliers rather than suppressors.
"""

from __future__ import annotations

import numpy as np


class HallucinationGate:
    """Track calibration/OOD statistics and conservatively bound penalties."""

    def __init__(
        self,
        sigma_baseline: float = 0.05,
        sigma_scale: float = 1.0,
        ood_percentile: float = 95.0,
        buffer_size: int = 2000,
        delta_clip: float = 10.0,
        recalib_target_mae: float = 0.10,
        max_uncertainty_multiplier: float = 2.0,
        max_ood_multiplier: float = 2.0,
    ):
        self.sigma_baseline = max(float(sigma_baseline), 1e-6)
        self.sigma_scale = max(float(sigma_scale), 0.0)
        if not 0.0 < float(ood_percentile) < 100.0:
            raise ValueError("ood_percentile must lie strictly between 0 and 100")
        if int(buffer_size) < 2:
            raise ValueError("buffer_size must be at least two")
        self.ood_percentile = float(ood_percentile)
        self.delta_clip = max(float(delta_clip), 0.0)
        self.recalib_target_mae = max(float(recalib_target_mae), 1e-6)
        self.max_uncertainty_multiplier = max(float(max_uncertainty_multiplier), 1.0)
        self.max_ood_multiplier = max(float(max_ood_multiplier), 1.0)

        self._state_buffer: list[np.ndarray] = []
        self._buffer_size = int(buffer_size)
        self._states_seen = 0
        self._state_mean: np.ndarray | None = None
        self._state_cov_inv: np.ndarray | None = None
        self._ood_threshold: float | None = None

        self._mae_ema = 0.0
        self._mae_alpha = 0.05

    def observe_state(self, state: np.ndarray) -> None:
        state = np.asarray(state, np.float32)
        if self._state_buffer and state.shape != self._state_buffer[0].shape:
            raise ValueError("OOD state shape changed after calibration began")
        self._states_seen += 1
        self._state_buffer.append(state.copy())
        if len(self._state_buffer) > self._buffer_size:
            self._state_buffer.pop(0)
        if len(self._state_buffer) == 50 or self._states_seen % 500 == 0:
            self._rebuild_ood_stats()

    def observe_icn_mae(self, mae: float) -> None:
        if np.isfinite(mae):
            self._mae_ema = (
                self._mae_alpha * abs(float(mae))
                + (1.0 - self._mae_alpha) * self._mae_ema
            )

    def _rebuild_ood_stats(self) -> None:
        states = np.stack(self._state_buffer).astype(np.float64)
        self._state_mean = states.mean(axis=0)
        covariance = np.atleast_2d(np.cov(states, rowvar=False))
        covariance += np.eye(states.shape[1]) * 1e-4
        self._state_cov_inv = np.linalg.pinv(covariance)
        distances = self._mahal_batch(states)
        self._ood_threshold = max(
            float(np.percentile(distances, self.ood_percentile)), 1e-6
        )

    def _mahal_batch(self, states: np.ndarray) -> np.ndarray:
        if self._state_mean is None or self._state_cov_inv is None:
            return np.zeros(len(states), dtype=np.float64)
        diff = states - self._state_mean[None]
        squared = np.einsum("bi,ij,bj->b", diff, self._state_cov_inv, diff)
        return np.sqrt(np.maximum(squared, 0.0))

    def _mahal(self, state: np.ndarray) -> float:
        return float(self._mahal_batch(np.asarray(state, np.float64)[None])[0])

    def sigma_gate(self, sigma: np.ndarray) -> np.ndarray:
        """Return a bounded risk multiplier (never a penalty suppressor)."""
        sigma = np.maximum(np.asarray(sigma, np.float32), 0.0)
        excess = np.maximum(sigma - self.sigma_baseline, 0.0)
        scaled = self.sigma_scale * excess / self.sigma_baseline
        return np.clip(
            1.0 + scaled, 1.0, self.max_uncertainty_multiplier
        ).astype(np.float32)

    def ood_gate(self, state: np.ndarray) -> float:
        """Return a bounded OOD risk multiplier (one while uncalibrated)."""
        if self._ood_threshold is None:
            return 1.0
        ratio = self._mahal(state) / self._ood_threshold
        return float(np.clip(ratio, 1.0, self.max_ood_multiplier))

    def clip_delta_C(self, delta_C: np.ndarray) -> np.ndarray:
        return np.clip(delta_C, 0.0, self.delta_clip).astype(np.float32)

    def gate_penalty(
        self,
        raw_penalty: np.ndarray,
        delta_C: np.ndarray,
        sigma: np.ndarray,
        state: np.ndarray,
        penalty_per_cost: float | np.ndarray | None = None,
    ) -> np.ndarray:
        """Clip the nominal estimate, then add bounded uncertainty/OOD margins.

        An additive uncertainty margin matters when the point estimate is zero;
        merely multiplying a zero estimate by a risk factor is not conservative.
        ``penalty_per_cost`` supplies the policy's lambda/scale conversion.
        """
        positive_delta = np.maximum(np.asarray(delta_C, np.float32), 0.0)
        clipped_delta = self.clip_delta_C(positive_delta)
        sigma = np.maximum(np.asarray(sigma, np.float32), 0.0)
        multiplier = self.sigma_gate(sigma)
        uncertainty_margin = (multiplier - 1.0) * np.maximum(
            clipped_delta, sigma)
        conservative_cost = np.clip(
            clipped_delta + uncertainty_margin, 0.0, self.delta_clip)

        if penalty_per_cost is None:
            nominal = np.asarray(raw_penalty, np.float32)
            inferred_scale = np.divide(
                nominal, positive_delta,
                out=np.zeros_like(nominal), where=positive_delta > 1e-8)
            fallback = float(inferred_scale.max()) if inferred_scale.size else 0.0
            scale = np.where(inferred_scale > 0.0, inferred_scale, fallback)
        else:
            scale = np.asarray(penalty_per_cost, np.float32)
        return (scale * conservative_cost * self.ood_gate(state)).astype(np.float32)

    def recalib_weight(self) -> float:
        ratio = self._mae_ema / self.recalib_target_mae
        return float(np.clip(0.3 + 0.7 * max(ratio - 1.0, 0.0), 0.3, 1.0))


def patch_ccpl_agent(agent, **gate_kwargs):
    """Configure the agent's native gate without replacing agent methods.

    Kept for compatibility with old launch scripts.  Earlier versions replaced
    ``select_action``, ``store``, and ``update`` with stale copies, which made
    later bug fixes ineffective and dropped the ``info`` argument to ``store``.
    """
    gate = HallucinationGate(**gate_kwargs)
    agent.hallucination_gate = gate
    return gate
