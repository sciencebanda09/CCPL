"""Empirical and numerical diagnostics for CCPL.

These checks test implementation invariants and measured behaviour.  They do
not machine-verify convergence, causal identification, dominance, or transfer
theorems; those claims require mathematical assumptions or independent
experiments that a runtime assertion cannot establish.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def _unknown(note: str) -> Dict:
    return {"satisfied": None, "status": "not_evaluated", "note": note}


def effective_discount(probabilities: np.ndarray, gamma: float) -> np.ndarray:
    """Compute E[gamma**tau] for rows of delay probabilities.

    Columns correspond to delays 0..K.  The function validates the probability
    simplex because the contraction bound depends on this assumption.
    """
    probs = np.asarray(probabilities, np.float64)
    if probs.ndim == 1:
        probs = probs[None, :]
    if probs.ndim != 2 or probs.shape[1] == 0:
        raise ValueError("probabilities must have shape (batch, tau_max + 1)")
    if not np.isfinite(probs).all() or np.any(probs < 0.0):
        raise ValueError("delay probabilities must be finite and non-negative")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("each delay-probability row must sum to one")
    gamma = float(gamma)
    if not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must be in [0, 1)")
    delays = np.arange(probs.shape[1], dtype=np.float64)
    return probs @ np.power(gamma, delays)


def effective_discount_bounds(probabilities: np.ndarray, gamma: float) -> Dict:
    """Return the exact support bound implied by a delay distribution."""
    values = effective_discount(probabilities, gamma)
    tau_max = np.asarray(probabilities).shape[-1] - 1
    lower, upper = float(gamma ** tau_max), 1.0
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "lower_bound": lower,
        "upper_bound": upper,
        "support_valid": bool(np.all((values >= lower - 1e-8) & (values <= upper + 1e-8))),
        "strict_contraction_if_min_delay_positive": bool(
            np.all(values < 1.0 - 1e-12)
        ),
    }


def verify_T1_contraction(
    gamma_eff_log: List[float], gamma: float = 0.99, tau_max: int = 15
) -> Dict:
    """Check the standard Bellman modulus and delay-factor support."""
    contraction = bool(0.0 <= gamma < 1.0)
    result = {
        "satisfied": contraction,
        "status": "pass" if contraction else "fail",
        "contraction_modulus": float(gamma),
        "note": "The reward bootstrap contracts by standard gamma.",
    }
    if gamma_eff_log:
        values = np.asarray(gamma_eff_log, np.float64)
        lower, upper = gamma ** tau_max, 1.0
        support_ok = bool(
            np.all(np.isfinite(values))
            and np.all(values >= lower - 1e-6)
            and np.all(values <= upper + 1e-6)
        )
        result.update({
            "gamma_eff_min": round(float(values.min()), 6),
            "gamma_eff_mean": round(float(values.mean()), 6),
            "gamma_eff_max": round(float(values.max()), 6),
            "delay_support": [round(lower, 6), round(upper, 6)],
            "delay_support_valid": support_ok,
            "satisfied": contraction and support_ok,
            "status": "pass" if contraction and support_ok else "fail",
        })
    return result


def verify_T2_dominance(
    var_s: float,
    lambda_g_mean: float,
    constraint_d: float,
    state_lambda_J_c: List[float] | None = None,
    global_lambda_J_c: List[float] | None = None,
) -> Dict:
    """Report heterogeneity and, when supplied, a matched empirical effect."""
    result = {
        "satisfied": None,
        "status": "diagnostic_only",
        "var_s": round(float(var_s), 6),
        "lambda_g_mean": round(float(lambda_g_mean), 6),
        "constraint_d": float(constraint_d),
        "epsilon_bound": None,
        "note": "Cost variance alone does not imply state-lambda dominance.",
    }
    if state_lambda_J_c is None or global_lambda_J_c is None:
        return result
    state = np.asarray(state_lambda_J_c, np.float64)
    scalar = np.asarray(global_lambda_J_c, np.float64)
    if state.size < 2 or scalar.size < 2:
        result["note"] = "Need at least two independent seed summaries per variant."
        return result
    result.update({
        "observed_cost_difference": round(float(scalar.mean() - state.mean()), 6),
        "state_lambda_mean_cost": round(float(state.mean()), 6),
        "scalar_lambda_mean_cost": round(float(scalar.mean()), 6),
        "note": "Observed effect only; report a seed-level interval before inference.",
    })
    return result


def verify_T3_causal_consistency(
    states: np.ndarray,
    actions: np.ndarray,
    icn_delta_C: np.ndarray,
    n_actions: int = 5,
) -> Dict:
    """Measure prediction agreement with the repository's known synthetic SCM."""
    if states.ndim != 2 or states.shape[1] != 6 or n_actions != 5:
        return _unknown("Synthetic SCM calibration is defined only for 6D/5-action tasks.")
    from causal_graph import CausalLabelGenerator, EnvironmentSCM

    labels = CausalLabelGenerator(EnvironmentSCM()).generate_batch(
        states, actions, n_actions
    )["delta_C_scm"]
    prediction = np.asarray(icn_delta_C, np.float64)
    labels = np.asarray(labels, np.float64)
    mae = float(np.mean(np.abs(prediction - labels)))
    if prediction.std() > 1e-8 and labels.std() > 1e-8:
        correlation = float(np.corrcoef(prediction, labels)[0, 1])
    else:
        correlation = float("nan")
    finite = bool(np.isfinite(prediction).all())
    return {
        "satisfied": finite,
        "status": "pass" if finite else "fail",
        "mae": round(mae, 6),
        "correlation": round(correlation, 6) if np.isfinite(correlation) else None,
        "sign_agreement": round(float(np.mean(
            np.sign(prediction) == np.sign(labels))), 6),
        "note": (
            "Calibration against a programmed SCM is not evidence of causal "
            "identification in observational or real-environment data."
        ),
    }


def verify_T4_lambda_convergence(
    lambda_log: List[float], lambda_max: float = 3.0, window: int = 100
) -> Dict:
    """Check numerical boundedness and report, but do not assert, stationarity."""
    if len(lambda_log) < 10:
        return _unknown("Need at least 10 lambda observations.")
    values = np.asarray(lambda_log, np.float64)
    bounded = bool(
        np.isfinite(values).all()
        and np.all(values >= -1e-7)
        and np.all(values <= lambda_max + 1e-7)
    )
    half = len(values) // 2
    return {
        "satisfied": bounded,
        "status": "pass" if bounded else "fail",
        "bounded": bounded,
        "lambda_min": round(float(values.min()), 6),
        "lambda_max_observed": round(float(values.max()), 6),
        "lambda_mean": round(float(values.mean()), 6),
        "std_first_half": round(float(values[:half].std()), 6),
        "std_second_half": round(float(values[half:].std()), 6),
        "note": "A falling finite-sample standard deviation is not a convergence proof.",
    }


def verify_T5_csr_monotonicity(
    episode_rewards: List[float],
    episode_consequences: List[float],
    constraint_d: float = 3.0,
    window: int = 50,
) -> Dict:
    """Summarise first/second-half learning trends."""
    rewards = np.asarray(episode_rewards, np.float64)
    costs = np.asarray(episode_consequences, np.float64)
    if rewards.size < 20 or costs.size != rewards.size:
        return _unknown("Need at least 20 aligned episode reward/cost observations.")
    half = rewards.size // 2
    csr_first = float(np.mean(costs[:half] <= constraint_d))
    csr_second = float(np.mean(costs[half:] <= constraint_d))
    return {
        "satisfied": None,
        "status": "empirical_trend",
        "csr_first": round(csr_first, 6),
        "csr_second": round(csr_second, 6),
        "reward_first": round(float(rewards[:half].mean()), 6),
        "reward_second": round(float(rewards[half:].mean()), 6),
        "note": "Learning curves need seed-level uncertainty; monotonicity is not guaranteed.",
    }


def verify_T6_transfer_stability(
    source_csrs: Dict[str, float],
    transfer_csrs: Dict[str, float],
    source_rewards: Dict[str, float],
    transfer_rewards: Dict[str, float],
) -> Dict:
    """Report observed source-to-transfer changes without a theorem threshold."""
    if not source_csrs or not transfer_csrs:
        return _unknown("Need source and transfer evaluations.")
    source_csr = float(np.mean(list(source_csrs.values())))
    transfer_csr = float(np.mean(list(transfer_csrs.values())))
    source_reward = float(np.mean(list(source_rewards.values())))
    transfer_reward = float(np.mean(list(transfer_rewards.values())))
    return {
        "satisfied": None,
        "status": "empirical_transfer",
        "source_csr_mean": round(source_csr, 6),
        "transfer_csr_mean": round(transfer_csr, 6),
        "csr_change": round(transfer_csr - source_csr, 6),
        "source_reward_mean": round(source_reward, 6),
        "transfer_reward_mean": round(transfer_reward, 6),
        "note": "Causal transfer requires baseline comparisons and independent seeds.",
    }


def run_all_theory_checks(
    agent, histories: dict | None = None, eval_results: dict | None = None,
    constraint_d: float = 3.0,
) -> Dict:
    """Run all available diagnostics (historical function name retained)."""
    logs = agent.get_theory_logs()
    results: Dict[str, Dict] = {}
    results["D1_bellman_and_delay"] = verify_T1_contraction(
        logs.get("gamma_eff_log", []), agent.gamma, getattr(agent, "tau_max", 15)
    )
    lambda_logs = logs.get("lambda_diagnostic", logs.get("theorem2", {}))
    results["D2_state_lambda"] = verify_T2_dominance(
        lambda_logs.get("var_s", 0.0), agent.last_mean_lambda, constraint_d
    )

    rng = np.random.default_rng(42)
    raw_states = rng.uniform(0.1, 0.9, (200, agent.state_dim)).astype(np.float32)
    actions = rng.integers(0, agent.action_dim, 200).astype(np.int32)
    normalized = np.asarray(
        [agent.normalizer.normalize(state) for state in raw_states], np.float32
    )
    icn_states = raw_states if getattr(agent, "has_scm_labels", False) else normalized
    context = np.zeros((200, agent.icn.causal_dim), np.float32)
    delta, _, _, _ = agent.icn.forward(icn_states, actions, context)
    results["D3_scm_calibration"] = verify_T3_causal_consistency(
        raw_states, actions, delta, agent.action_dim
    )
    results["D4_lambda_numerics"] = verify_T4_lambda_convergence(
        logs.get("lambda_log", []), agent.lambda_max
    )

    if histories:
        history_values = (
            list(histories.values())
            if not ("rewards" in histories or "consequences" in histories)
            else [histories]
        )
        rewards = [value for item in history_values for value in item.get("rewards", [])]
        costs = [value for item in history_values for value in item.get("consequences", [])]
        results["D5_learning_trend"] = verify_T5_csr_monotonicity(
            rewards, costs, constraint_d
        )
    else:
        results["D5_learning_trend"] = _unknown("No training history supplied.")

    ccpl_results = (eval_results or {}).get("CCPL", {})
    source_names = [name for name in ccpl_results if name in {"standard", "noisy", "shifted"}]
    transfer_names = [name for name in ccpl_results if name not in source_names]
    if source_names and transfer_names:
        results["D6_transfer"] = verify_T6_transfer_stability(
            {name: ccpl_results[name]["constraint_satisfaction_rate"] / 100.0
             for name in source_names},
            {name: ccpl_results[name]["constraint_satisfaction_rate"] / 100.0
             for name in transfer_names},
            {name: ccpl_results[name]["mean_reward"] for name in source_names},
            {name: ccpl_results[name]["mean_reward"] for name in transfer_names},
        )
    else:
        results["D6_transfer"] = _unknown("Need source and transfer evaluations.")
    return results


def print_theory_report(checks: Dict) -> None:
    """Print a compact, explicitly non-theorem diagnostic report."""
    print("\n" + "=" * 72)
    print("  CCPL NUMERICAL / EMPIRICAL DIAGNOSTICS")
    print("  These checks do not constitute machine-verified theorems.")
    print("=" * 72)
    for name, result in checks.items():
        print(f"  {name:<30} {result.get('status', 'unknown')}")
        if result.get("note"):
            print(f"    {result['note']}")
    print("=" * 72)
