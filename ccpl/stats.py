"""Statistical utilities with the trained seed as the unit of replication."""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy import stats as _sp
    _HAS_SCIPY = True
except ImportError:
    _sp = None
    _HAS_SCIPY = False


def shapiro_wilk(vals: List[float]) -> Tuple[float, float]:
    values = np.asarray(vals, np.float64)
    if not _HAS_SCIPY or not 3 <= values.size <= 5000:
        return float("nan"), float("nan")
    statistic, p_value = _sp.shapiro(values)
    return float(statistic), float(p_value)


def mannwhitney(
    a: List[float], b: List[float], alternative: str = "greater"
) -> Dict:
    """Mann-Whitney test for independent *trained-seed* summaries."""
    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError("alternative must be greater, less, or two-sided")
    left = np.asarray(a, np.float64)
    right = np.asarray(b, np.float64)
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if left.size < 2 or right.size < 2:
        return {
            "U": None, "p": None, "r": None, "n1": int(left.size),
            "n2": int(right.size), "alternative": alternative,
            "significant": False, "valid": False,
            "note": "Need at least two independent trained seeds per method.",
        }
    if not _HAS_SCIPY:
        return {
            "U": None, "p": None, "r": None, "n1": int(left.size),
            "n2": int(right.size), "alternative": alternative,
            "significant": False, "valid": False, "note": "scipy required",
        }
    statistic, p_value = _sp.mannwhitneyu(
        left, right, alternative=alternative, method="auto"
    )
    raw_r = 2.0 * float(statistic) / (left.size * right.size) - 1.0
    oriented_r = -raw_r if alternative == "less" else raw_r
    return {
        "U": float(statistic), "p": float(p_value),
        "r": round(oriented_r, 4), "n1": int(left.size),
        "n2": int(right.size), "alternative": alternative,
        "significant": bool(p_value < 0.05), "valid": True,
    }


def paired_randomization(
    a: List[float], b: List[float], alternative: str = "greater",
    max_exact_pairs: int = 16, n_resamples: int = 100_000, seed: int = 42,
) -> Dict:
    """Paired sign-randomization test using one summary per trained seed."""
    left = np.asarray(a, np.float64)
    right = np.asarray(b, np.float64)
    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError("alternative must be greater, less, or two-sided")
    if left.ndim != 1 or right.ndim != 1 or left.size != right.size:
        return {
            "statistic": None, "p": None, "r": None,
            "n": int(min(left.size, right.size)), "valid": False,
            "significant": False, "alternative": alternative,
            "note": "Paired tests require equal-length seed arrays.",
        }
    finite = np.isfinite(left) & np.isfinite(right)
    left, right = left[finite], right[finite]
    if left.size < 2:
        return {
            "statistic": None, "p": None, "r": None,
            "n": int(left.size), "valid": False, "significant": False,
            "alternative": alternative,
            "note": "Need at least two paired trained seeds.",
        }

    raw_diff = left - right
    oriented = -raw_diff if alternative == "less" else raw_diff
    magnitudes = np.abs(oriented[np.abs(oriented) > 1e-15])
    observed = float(oriented[np.abs(oriented) > 1e-15].sum())
    if magnitudes.size == 0:
        p_value, effect, exact = 1.0, 0.0, True
    else:
        two_sided = alternative == "two-sided"
        threshold = abs(observed) if two_sided else observed
        if magnitudes.size <= max_exact_pairs:
            count = 0
            total = 1 << int(magnitudes.size)
            for mask in range(total):
                signs = np.fromiter(
                    (1.0 if mask & (1 << j) else -1.0
                     for j in range(magnitudes.size)),
                    dtype=np.float64, count=magnitudes.size)
                statistic = float(np.dot(signs, magnitudes))
                if two_sided:
                    count += int(abs(statistic) >= threshold - 1e-15)
                else:
                    count += int(statistic >= threshold - 1e-15)
            p_value = count / total
            exact = True
        else:
            rng = np.random.default_rng(seed)
            count = 0
            remaining = int(n_resamples)
            while remaining:
                batch = min(remaining, 4096)
                signs = rng.choice(
                    np.array([-1.0, 1.0]), size=(batch, magnitudes.size))
                statistics = signs @ magnitudes
                if two_sided:
                    count += int(np.count_nonzero(
                        np.abs(statistics) >= threshold - 1e-15))
                else:
                    count += int(np.count_nonzero(
                        statistics >= threshold - 1e-15))
                remaining -= batch
            p_value = (count + 1.0) / (n_resamples + 1.0)
            exact = False
        effect = float(observed / (magnitudes.sum() + 1e-15))

    paired_sd = float(raw_diff.std(ddof=1))
    paired_d = (float(raw_diff.mean()) / paired_sd
                if paired_sd > 1e-15 else float("nan"))
    return {
        "statistic": observed, "p": float(p_value), "r": round(effect, 4),
        "paired_d": paired_d, "mean_difference": float(raw_diff.mean()),
        "n": int(left.size), "n_nonzero": int(magnitudes.size),
        "alternative": alternative, "exact": exact,
        "significant": bool(p_value < 0.05), "valid": True,
    }


def welch_t(a: List[float], b: List[float], alternative: str = "greater") -> Dict:
    left, right = np.asarray(a, float), np.asarray(b, float)
    if not _HAS_SCIPY or left.size < 2 or right.size < 2:
        return {"t": None, "p": None, "d": None, "valid": False}
    statistic, p_value = _sp.ttest_ind(
        left, right, equal_var=False, alternative=alternative
    )
    return {
        "t": float(statistic), "p": float(p_value),
        "d": round(cohen_d(left, right), 4), "valid": True,
    }


def cohen_d(a: List[float], b: List[float]) -> float:
    left, right = np.asarray(a, float), np.asarray(b, float)
    if left.size < 2 or right.size < 2:
        return float("nan")
    pooled = np.sqrt((left.var(ddof=1) + right.var(ddof=1)) / 2.0)
    return 0.0 if pooled < 1e-12 else float((left.mean() - right.mean()) / pooled)


def bootstrap_ci(
    vals: List[float], n_boot: int = 10_000, alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    values = np.asarray(vals, float)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    if values.size == 1:
        value = float(values[0])
        return value, value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[indices].mean(axis=1)
    return (
        float(values.mean()),
        float(np.percentile(means, 100.0 * alpha / 2.0)),
        float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0))),
    )


def paired_effect_summary(a: List[float], b: List[float], seed: int = 42) -> Dict:
    """Summarize a paired seed comparison with CI and standardized effect."""
    left = np.asarray(a, np.float64)
    right = np.asarray(b, np.float64)
    if left.ndim != 1 or right.ndim != 1 or left.size != right.size:
        return {"valid": False, "note": "paired arrays must have equal 1D shape"}
    finite = np.isfinite(left) & np.isfinite(right)
    differences = left[finite] - right[finite]
    if differences.size < 2:
        return {"valid": False, "n": int(differences.size), "note": "need at least two seeds"}
    mean, lower, upper = bootstrap_ci(differences.tolist(), seed=seed)
    sd = float(differences.std(ddof=1))
    return {
        "valid": True,
        "n": int(differences.size),
        "mean_difference": mean,
        "ci95": [lower, upper],
        "paired_cohen_d": float(mean / sd) if sd > 1e-12 else None,
        "min_difference": float(differences.min()),
        "max_difference": float(differences.max()),
    }


def kruskal_wallis(*groups: List[float]) -> Dict:
    arrays = [np.asarray(group, float) for group in groups]
    if not _HAS_SCIPY or len(arrays) < 2 or any(group.size < 2 for group in arrays):
        return {"H": None, "p": None, "significant": False, "valid": False}
    statistic, p_value = _sp.kruskal(*arrays)
    return {
        "H": float(statistic), "p": float(p_value),
        "significant": bool(p_value < 0.05), "valid": True,
    }


def friedman_test(*groups: List[float]) -> Dict:
    """Repeated-measures omnibus test for variants on common seeds."""
    arrays = [np.asarray(group, float) for group in groups]
    equal = bool(arrays) and len({len(group) for group in arrays}) == 1
    if (not _HAS_SCIPY or len(arrays) < 3 or not equal
            or any(group.size < 2 for group in arrays)):
        return {"chi2": None, "p": None, "significant": False, "valid": False}
    statistic, p_value = _sp.friedmanchisquare(*arrays)
    return {
        "chi2": float(statistic), "p": float(p_value),
        "significant": bool(p_value < 0.05), "valid": True,
    }


def bonferroni_threshold(alpha: float, n_tests: int) -> float:
    return alpha / max(int(n_tests), 1)


def holm_correction(p_values: List[float], alpha: float = 0.05) -> List[Dict]:
    if not p_values:
        return []
    count = len(p_values)
    order = np.argsort(p_values)
    output: List[Dict] = [None] * count
    still_rejecting = True
    for rank, index in enumerate(order):
        threshold = alpha / (count - rank)
        significant = bool(still_rejecting and p_values[index] <= threshold)
        if not significant:
            still_rejecting = False
        output[index] = {
            "p": float(p_values[index]), "holm_threshold": round(threshold, 6),
            "significant": significant,
        }
    return output


def _sig_stars(p: float | None, significant: bool, alpha: float) -> str:
    if not significant or p is None:
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < alpha else ""


def _seed_values(record: Dict, metric: str) -> Optional[np.ndarray]:
    """Read explicitly seed-level summaries; never infer them from episodes."""
    for key in (f"{metric}_by_seed", f"seed_{metric}"):
        if key in record:
            values = np.asarray(record[key], np.float64)
            return values if values.ndim == 1 else None
    seed_means = record.get("seed_means")
    if isinstance(seed_means, dict) and metric in seed_means:
        values = np.asarray(seed_means[metric], np.float64)
        return values if values.ndim == 1 else None
    return None


def _aggregate_seeds(agent_results: Dict, env_names: List[str], metric: str):
    series = [_seed_values(agent_results[env], metric) for env in env_names]
    if any(values is None for values in series):
        return None
    lengths = {len(values) for values in series}
    if len(lengths) != 1 or next(iter(lengths), 0) < 2:
        return None
    return np.stack(series).mean(axis=0)


def _descriptive_values(agent_results: Dict, env_names: List[str], metric: str):
    return np.asarray([
        value for env in env_names
        for value in agent_results.get(env, {}).get(metric, [])
    ], np.float64)


def full_comparison_table(
    results_by_agent: Dict, env_names: List[str], ccpl_key: str = "CCPL",
    metric: str = "rewards", alpha: float = 0.05,
) -> List[Dict]:
    """Compare seed-aggregated methods; episode-only inputs are descriptive."""
    if ccpl_key not in results_by_agent:
        raise KeyError(f"Missing reference agent {ccpl_key!r}")
    alternative = "less" if metric in {"consequences", "costs", "Jc"} else "greater"
    ccpl_seeds = _aggregate_seeds(results_by_agent[ccpl_key], env_names, metric)
    baselines = [name for name in results_by_agent if name != ccpl_key]

    rows = []
    valid_indices, valid_p = [], []
    for name in baselines:
        base_seeds = _aggregate_seeds(results_by_agent[name], env_names, metric)
        valid = ccpl_seeds is not None and base_seeds is not None
        statistic = paired_randomization(
            ccpl_seeds, base_seeds, alternative) if valid else {
            "U": None, "p": None, "r": None, "significant": False, "valid": False,
            "note": "No explicit seed-level summaries; episode data are descriptive only.",
        }
        base_desc = _descriptive_values(results_by_agent[name], env_names, metric)
        ccpl_desc = _descriptive_values(results_by_agent[ccpl_key], env_names, metric)
        ci_source = base_seeds if base_seeds is not None else []
        mean, lower, upper = bootstrap_ci(ci_source)
        row = {
            "agent": name,
            "pooled_mean": round(float(base_desc.mean()), 3) if base_desc.size else None,
            "ci_lo": round(lower, 3) if np.isfinite(lower) else None,
            "ci_hi": round(upper, 3) if np.isfinite(upper) else None,
            "ccpl_pooled_mean": round(float(ccpl_desc.mean()), 3) if ccpl_desc.size else None,
            "U": statistic.get("statistic"), "p": statistic.get("p"),
            "r": statistic.get("r"),
            "cohen_d": (round(statistic.get("paired_d"), 3)
                        if valid and np.isfinite(
                            statistic.get("paired_d", np.nan)) else None),
            "significant": False, "sig_stars": "", "alpha": alpha,
            "n_tests": len(baselines), "alternative": alternative,
            "inferential_valid": bool(statistic.get("valid")),
            "note": statistic.get("note"),
        }
        if statistic.get("valid"):
            valid_indices.append(len(rows))
            valid_p.append(statistic["p"])
        rows.append(row)

    for row_index, correction in zip(valid_indices, holm_correction(valid_p, alpha)):
        rows[row_index]["significant"] = correction["significant"]
        rows[row_index]["sig_stars"] = _sig_stars(
            rows[row_index]["p"], correction["significant"], alpha
        )
        rows[row_index]["holm_threshold"] = correction["holm_threshold"]
    return rows


def lambda_ablation_stats(
    variant_results: Dict, env_names: List[str], metric: str = "consequences"
) -> Dict:
    groups = {
        name: _aggregate_seeds(results, env_names, metric)
        for name, results in variant_results.items()
    }
    if any(values is None for values in groups.values()):
        return {
            "friedman": {"valid": False, "p": None, "chi2": None,
                         "significant": False},
            "pairwise": {}, "omnibus_significant": False,
            "note": "Ablation inference requires explicit seed-level summaries.",
        }
    omnibus = friedman_test(*groups.values())
    pairs, p_values, keys = {}, [], []
    for left, right in combinations(groups, 2):
        test = paired_randomization(
            groups[left], groups[right], alternative="two-sided")
        key = f"{left}_vs_{right}"
        pairs[key] = test
        if test.get("valid"):
            keys.append(key)
            p_values.append(test["p"])
    for key, correction in zip(keys, holm_correction(p_values)):
        pairs[key]["holm_significant"] = correction["significant"]
        pairs[key]["holm_threshold"] = correction["holm_threshold"]
    return {
        "friedman": omnibus, "pairwise": pairs,
        "omnibus_significant": omnibus.get("significant", False),
    }


def print_stat_table(rows: List[Dict], title: str = "Statistical comparison vs CCPL"):
    print(f"\n{'=' * 90}\n  {title}\n{'=' * 90}")
    print(f"  {'Agent':<16} | {'Desc. mean':>10} | {'seed 95% CI':>17} | "
          f"{'p':>9} | {'effect':>8} | {'status':>12}")
    for row in rows:
        ci = (f"[{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]"
              if row["ci_lo"] is not None else "n/a")
        p_value = f"{row['p']:.4g}" if row["p"] is not None else "n/a"
        effect = f"{row['r']:.3f}" if row["r"] is not None else "n/a"
        status = row["sig_stars"] or ("n/s" if row["inferential_valid"] else "descriptive")
        mean = f"{row['pooled_mean']:.3f}" if row["pooled_mean"] is not None else "n/a"
        print(f"  {row['agent']:<16} | {mean:>10} | {ci:>17} | "
              f"{p_value:>9} | {effect:>8} | {status:>12}")
    print("=" * 90)


def print_lambda_ablation(result: Dict):
    omnibus = result["friedman"]
    if not omnibus.get("valid"):
        print(f"  {result.get('note', 'Seed-level data unavailable.')}")
        return
    print(f"  Friedman: chi2={omnibus['chi2']:.3f}, p={omnibus['p']:.4g}")


def latex_table(rows: List[Dict]) -> str:
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Agent & Descriptive mean & Seed 95\% CI & $p$ & Effect \\",
             r"\midrule"]
    for row in rows:
        mean = "n/a" if row["pooled_mean"] is None else f"{row['pooled_mean']:.3f}"
        ci = ("n/a" if row["ci_lo"] is None
              else f"[{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]")
        p_value = "n/a" if row["p"] is None else f"{row['p']:.4g}"
        effect = "n/a" if row["r"] is None else f"{row['r']:.3f}"
        lines.append(f"{row['agent']} & {mean} & {ci} & {p_value} & {effect} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def run_full_stats(
    eval_results: Dict, ablation_results: Optional[Dict], env_names: List[str],
    ccpl_key: str = "CCPL", variant_results: Optional[Dict] = None,
) -> Dict:
    print("\n" + "=" * 70 + "\n  Statistical Analysis\n" + "=" * 70)
    reward_rows = full_comparison_table(
        eval_results, env_names, ccpl_key, metric="rewards"
    )
    cost_rows = full_comparison_table(
        eval_results, env_names, ccpl_key, metric="consequences"
    )
    print_stat_table(reward_rows, "Reward: methods vs CCPL")
    print_stat_table(cost_rows, "Constraint cost: methods vs CCPL")
    output = {
        "reward_comparison": reward_rows,
        "constraint_comparison": cost_rows,
        "reward_latex": latex_table(reward_rows),
        "constraint_latex": latex_table(cost_rows),
    }
    if ablation_results:
        output["lambda_ablation"] = lambda_ablation_stats(
            ablation_results, env_names, "consequences"
        )
        print_lambda_ablation(output["lambda_ablation"])
    if variant_results:
        output["F4_state_vs_global_lambda"] = {
            "valid": False,
            "note": "Use explicit seed-level summaries through lambda_ablation_stats.",
        }
    return output
