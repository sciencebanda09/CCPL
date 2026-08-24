"""Legacy diagnostic entry points used by ``benchmark.py``.

The historical module called finite-sample trends “machine-verifiable
theorems.”  They are now reported as implementation or empirical diagnostics.
Function names and call signatures are retained for compatibility.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def verify_lambda_boundedness(lambda_log: List[float], lambda_max: float) -> Dict:
    values = np.asarray(lambda_log, np.float64)
    if values.size == 0:
        return {"satisfied": None, "note": "No lambda observations."}
    violations = int(np.count_nonzero(
        ~np.isfinite(values) | (values < 0.0) | (values > lambda_max)
    ))
    return {
        "min_lambda": round(float(np.nanmin(values)), 6),
        "max_lambda": round(float(np.nanmax(values)), 6),
        "lambda_max": float(lambda_max),
        "violations": violations,
        "satisfied": violations == 0,
        "description": "Numerical invariant: observed lambda is bounded.",
    }


def verify_monotonicity(
    episode_mean_C: List[float], episode_mean_lambda: List[float]
) -> Dict:
    costs = np.asarray(episode_mean_C, np.float64)
    lambdas = np.asarray(episode_mean_lambda, np.float64)
    if costs.size < 3 or costs.size != lambdas.size:
        return {"satisfied": None, "note": "Need aligned episode summaries."}
    try:
        from scipy.stats import spearmanr
        rho, p_value = spearmanr(costs, lambdas)
        rho, p_value = float(rho), float(p_value)
    except ImportError:
        rho = float(np.corrcoef(costs, lambdas)[0, 1])
        p_value = float("nan")
    return {
        "spearman_rho": round(rho, 6) if np.isfinite(rho) else None,
        "p_value": round(p_value, 6) if np.isfinite(p_value) else None,
        "satisfied": None,
        "description": "Episode-level association (time/confounding not controlled).",
    }


def verify_convergence(
    episode_lambda_stds: List[float], first_half_fraction: float = 0.5
) -> Dict:
    values = np.asarray(episode_lambda_stds, np.float64)
    if values.size < 10:
        return {"satisfied": None, "note": "Fewer than 10 episodes."}
    split = max(1, int(values.size * first_half_fraction))
    tail = values[split:]
    slope = float(np.polyfit(np.arange(tail.size), tail, 1)[0])
    return {
        "slope": round(slope, 8),
        "satisfied": None,
        "description": "Observed lambda-variance trend; not a convergence proof.",
    }


def verify_state_vs_global_lambda(
    state_lambda_J_c: List[float], global_lambda_J_c: List[float]
) -> Dict:
    state = np.asarray(state_lambda_J_c, np.float64)
    scalar = np.asarray(global_lambda_J_c, np.float64)
    if state.size < 2 or scalar.size < 2:
        return {"satisfied": None, "note": "Need independent seed aggregates."}
    return {
        "mean_state": round(float(state.mean()), 6),
        "mean_global": round(float(scalar.mean()), 6),
        "global_minus_state": round(float(scalar.mean() - state.mean()), 6),
        "satisfied": None,
        "description": (
            "Observed effect only. Inputs must be independent seed summaries; "
            "evaluation episodes from one trained seed are not replicates."
        ),
    }


def verify_delay_calibration(
    expected_delays: List[float], observed_delays: List[float], tau_max: int = 15
) -> Dict:
    expected = np.asarray(expected_delays, np.float64)
    observed = np.asarray(observed_delays, np.float64)
    if expected.size != observed.size:
        return {"satisfied": False, "note": "Delay arrays are not aligned."}
    valid = np.isfinite(expected) & np.isfinite(observed) & (observed >= 0)
    if np.count_nonzero(valid) < 5:
        return {"satisfied": None, "note": "Fewer than five paired observations."}
    mae = float(np.mean(np.abs(expected[valid] - observed[valid])))
    return {
        "mae": round(mae, 6),
        "n_valid": int(np.count_nonzero(valid)),
        "satisfied": bool(np.isfinite(mae)),
        "description": "Delay prediction calibration (report MAE; no theorem threshold).",
    }


def run_theoretical_checks(
    lambda_log: List[float],
    episode_mean_C: List[float],
    episode_mean_lambda: List[float],
    episode_lambda_stds: List[float],
    expected_delays: List[float],
    observed_delays: List[float],
    lambda_max: float = 3.0,
    tau_max: int = 15,
    state_lambda_J_c: List[float] | None = None,
    global_lambda_J_c: List[float] | None = None,
) -> Dict:
    checks = {
        "F1_boundedness": verify_lambda_boundedness(lambda_log, lambda_max),
        "F2_association": verify_monotonicity(episode_mean_C, episode_mean_lambda),
        "F3_variance_trend": verify_convergence(episode_lambda_stds),
        "F5_delay_calib": verify_delay_calibration(
            expected_delays, observed_delays, tau_max
        ),
    }
    checks["F4_state_vs_global"] = (
        verify_state_vs_global_lambda(state_lambda_J_c, global_lambda_J_c)
        if state_lambda_J_c is not None and global_lambda_J_c is not None
        else {"satisfied": None, "note": "No matched state/scalar results supplied."}
    )
    return checks


def print_theoretical_report(checks: Dict) -> None:
    print("\n" + "=" * 72)
    print("  CCPL LEGACY NUMERICAL / EMPIRICAL DIAGNOSTICS")
    print("  These checks are not theorem verification.")
    print("=" * 72)
    for name, result in checks.items():
        status = result.get("satisfied")
        label = "pass" if status is True else "fail" if status is False else "diagnostic"
        print(f"  {name:<28} {label}")
        if result.get("description"):
            print(f"    {result['description']}")
        if result.get("note"):
            print(f"    {result['note']}")
    print("=" * 72)


THEOREM_1 = """
The former lambda-network convergence theorem is withdrawn.  Bounded sigmoid
outputs are an architectural invariant; decreasing empirical variance and
delay calibration do not prove stochastic-approximation convergence.
"""
