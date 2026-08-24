"""Synthetic intervention data for held-out causal evaluation."""

from __future__ import annotations

import numpy as np

try:
    from .causal_graph import EnvironmentSCM
except ImportError:  # Legacy checkout imports.
    from causal_graph import EnvironmentSCM


def make_intervention_dataset(
    n_states: int = 1000,
    seed: int = 0,
    train_fraction: float = 0.8,
    n_actions: int = 5,
) -> dict:
    """Generate randomized action interventions and a fixed held-out split."""
    if n_states < 2 or not 0.0 < train_fraction < 1.0:
        raise ValueError("n_states must be >= 2 and train_fraction must be in (0, 1)")
    if n_actions != 5:
        raise ValueError("the reference SCM defines exactly five actions")
    rng = np.random.default_rng(seed)
    states = rng.uniform(0.05, 0.95, size=(n_states, 6)).astype(np.float32)
    actions = rng.integers(0, n_actions, size=n_states, dtype=np.int32)
    scm = EnvironmentSCM()
    labels = np.array([
        scm.causal_attribution(state, int(action))
        for state, action in zip(states, actions)
    ], dtype=np.float32)
    split = max(1, min(n_states - 1, int(n_states * train_fraction)))
    return {
        "train": {"states": states[:split], "actions": actions[:split], "labels": labels[:split]},
        "test": {"states": states[split:], "actions": actions[split:], "labels": labels[split:]},
        "seed": int(seed),
        "n_actions": n_actions,
    }


def evaluate_intervention_predictions(predictions: np.ndarray, labels: np.ndarray) -> dict:
    """Return held-out MAE, correlation, and sign agreement."""
    pred = np.asarray(predictions, np.float64).reshape(-1)
    truth = np.asarray(labels, np.float64).reshape(-1)
    if pred.size == 0 or pred.size != truth.size:
        raise ValueError("predictions and labels must be non-empty and aligned")
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("predictions and labels must be finite")
    correlation = np.corrcoef(pred, truth)[0, 1] if pred.std() and truth.std() else np.nan
    return {
        "n": int(pred.size),
        "mae": float(np.mean(np.abs(pred - truth))),
        "rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
        "correlation": None if not np.isfinite(correlation) else float(correlation),
        "sign_agreement": float(np.mean(np.sign(pred) == np.sign(truth))),
    }
