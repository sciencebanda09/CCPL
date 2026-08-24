"""
evaluate.py — CCPL unified evaluation module
================================================
Replaces both evaluate.py and evaluate_v2.py (the two files were nearly
identical duplicates). All function signatures from both versions are
preserved for full backward compatibility.

Core metrics:
  mean_reward               — average episodic return
  mean_consequence          — average episodic J_c
  constraint_satisfaction_rate (CSR) — % episodes with J_c ≤ threshold
  stability                 — 1 / (std(rewards) + 1)
  delayed_failure_rate      — fraction of steps that triggered delayed hits
  resource_preservation     — mean(1 − resource_load)
  transfer_score            — normalised cross-env reward score

All evaluation is done in eval_mode (no exploration, no training updates).
"""
import time
import numpy as np
try:
    from .environments import ENV_REGISTRY
except ImportError:  # Legacy checkout imports.
    from environments import ENV_REGISTRY


# ── Single-agent, single-environment evaluation ───────────────────────────────

def evaluate_agent(agent, env_name: str, n_episodes: int = 100,
                   max_steps: int = 100, delay_steps: int = 5,
                   seed_offset: int = 9999,
                   constraint_threshold: float = None) -> dict:
    """
    Evaluate one agent on one named environment.

    Returns a metrics dict containing:
      mean_reward, std_reward, mean_consequence, std_consequence,
      mean_delayed_hits, delayed_failure_rate, resource_preservation,
      stability, mean_infer_ms, constraint_satisfaction_rate,
      constraint_threshold, rewards (list), consequences (list),
      delayed_hits_list (list).
    """
    if n_episodes < 1:
        raise ValueError("n_episodes must be positive")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    is_ccpl  = hasattr(agent, "reset_hidden")
    EnvClass = ENV_REGISTRY.get(env_name)
    if EnvClass is None:
        raise KeyError(
            f"Environment '{env_name}' not in ENV_REGISTRY. "
            f"Import adversarial_envs to register adversarial environments.")

    # Resolve constraint threshold
    if constraint_threshold is None:
        env_sample        = EnvClass(max_steps=1, seed=0)
        constraint_threshold = getattr(env_sample, "constraint_threshold", float("inf"))
        if hasattr(env_sample, "close"):
            env_sample.close()

    rewards, consequences, raw_consequences, delayed_hits = [], [], [], []
    episode_steps = []
    resource_loads, infer_times = [], []
    csr_violations, raw_csr_violations = [], []

    for ep in range(n_episodes):
        env = EnvClass(max_steps=max_steps, consequence_delay=delay_steps,
                       seed=seed_offset + ep)
        state   = env.reset()
        if is_ccpl:
            agent.reset_hidden(
                max_steps=getattr(env, "max_steps", max_steps),
                expected_delay=getattr(env, "consequence_delay", None))

        ep_r = ep_c = ep_c_raw = 0.0
        gamma_c = 1.0
        gamma = float(getattr(agent, "gamma", 1.0))
        ep_loads    = []

        while not env.done:
            t0     = time.perf_counter()
            action = agent.select_action(state, eval_mode=True)
            infer_times.append(time.perf_counter() - t0)

            ns, r, c, done, info = env.step(action)
            ep_r += r
            ep_c += gamma_c * c
            ep_c_raw += c
            gamma_c *= gamma
            if is_ccpl and hasattr(agent, "observe_transition"):
                agent.observe_transition(state, action, c)
            ep_loads.append(info.get("resource_load", 0.5))
            state = ns

        stats = env.episode_stats()
        rewards.append(ep_r)
        consequences.append(ep_c)
        raw_consequences.append(ep_c_raw)
        delayed_hits.append(stats["delayed_hits"])
        episode_steps.append(stats.get("steps", max_steps))
        resource_loads.append(float(np.mean(ep_loads)) if ep_loads else 0.5)
        csr_violations.append(int(ep_c > constraint_threshold))
        raw_csr_violations.append(int(ep_c_raw > constraint_threshold))
        if hasattr(env, "close"):
            env.close()

    csr = 100.0 * (1.0 - float(np.mean(csr_violations)))

    return {
        "mean_reward":                float(np.mean(rewards)),
        "std_reward":                 float(np.std(rewards)),
        "mean_consequence":           float(np.mean(consequences)),
        "std_consequence":            float(np.std(consequences)),
        "mean_delayed_hits":          float(np.mean(delayed_hits)),
        "delayed_failure_rate":       float(np.mean([
            h / max(s, 1) for h, s in zip(delayed_hits, episode_steps)])),
        "resource_preservation":      float(np.mean([1 - rl for rl in resource_loads])),
        "stability":                  float(1.0 / (np.std(rewards) + 1.0)),
        "mean_infer_ms":              float(np.mean(infer_times) * 1000),
        "constraint_satisfaction_rate": csr,
        "constraint_satisfaction_rate_undiscounted": 100.0 * (
            1.0 - float(np.mean(raw_csr_violations))),
        "constraint_threshold":       constraint_threshold,
        "rewards":                    rewards,
        "consequences":               consequences,
        "mean_consequence_undiscounted": float(np.mean(raw_consequences)),
        "consequences_undiscounted":  raw_consequences,
        "delayed_hits_list":          delayed_hits,
    }


# ── Multi-agent, multi-environment evaluation ─────────────────────────────────

def evaluate_all(agents: dict, env_names: list,
                 n_episodes: int = 100, max_steps: int = 100,
                 delay_steps: int = 5,
                 constraint_thresholds: dict = None) -> dict:
    """
    Evaluate all agents on all environments.

    agents               : {name: agent_object}
    constraint_thresholds: optional {env_name: threshold} overrides
    Returns              : {agent_name: {env_name: metrics_dict}}
    """
    results = {}
    for name, agent in agents.items():
        results[name] = {}
        for env_name in env_names:
            print(f"  Evaluating {name} on {env_name}...")
            thresh = (constraint_thresholds or {}).get(env_name)
            results[name][env_name] = evaluate_agent(
                agent, env_name, n_episodes, max_steps, delay_steps,
                constraint_threshold=thresh)
    return results


# ── Transfer score ────────────────────────────────────────────────────────────

def compute_transfer_score(results: dict, env_names: list) -> dict:
    """
    Min–max normalised mean reward across all environments.
    Scores are in [0, 1]; higher is better.
    """
    all_vals = [results[n][e]["mean_reward"]
                for n in results for e in env_names]
    r_min, r_max = min(all_vals), max(all_vals) + 1e-8
    return {
        name: float(np.mean([
            (results[name][e]["mean_reward"] - r_min) / (r_max - r_min)
            for e in env_names
        ]))
        for name in results
    }


# ── Print helpers ─────────────────────────────────────────────────────────────

def print_csr_table(results: dict, env_names: list):
    """Print constraint satisfaction rate table."""
    print("\n" + "=" * 80)
    print("  CONSTRAINT SATISFACTION RATE (CSR) — % episodes with J_c ≤ threshold")
    print("=" * 80)
    header = f"  {'Agent':<16}"
    for e in env_names:
        header += f"  {e[:12]:>12}"
    header += f"  {'Mean':>8}"
    print(header)
    print("  " + "-" * 75)

    for name, res in results.items():
        row  = f"  {name:<16}"
        csrs = []
        for e in env_names:
            csr = res[e].get("constraint_satisfaction_rate", float("nan"))
            csrs.append(csr)
            row += f"  {csr:>11.1f}%"
        row += f"  {np.nanmean(csrs):>7.1f}%"
        print(row)
    print("=" * 80)


def print_benchmark_table(results: dict, env_names: list,
                           transfer_scores: dict,
                           compute_stats: dict = None,
                           show_csr: bool = True):
    """Print full benchmark comparison table."""
    agents = list(results.keys())
    print("\n" + "=" * 110)
    print(f"{'CCPL Benchmark Table':^110}")
    print("=" * 110)
    hdr = (f"{'Algorithm':<18} | {'Reward':>8} | {'J_c':>9} | "
           f"{'CSR%':>6} | {'Hits':>6} | {'Stability':>9} | {'Transfer':>8}")
    if compute_stats:
        hdr += f" | {'Params':>8} | {'ms/step':>7}"
    print(hdr)
    print("-" * 110)

    for agent in agents:
        all_r   = [results[agent][e]["mean_reward"]                           for e in env_names]
        all_c   = [results[agent][e]["mean_consequence"]                      for e in env_names]
        all_csr = [results[agent][e].get("constraint_satisfaction_rate", 0.0) for e in env_names]
        all_dh  = [results[agent][e]["mean_delayed_hits"]                     for e in env_names]
        all_st  = [results[agent][e]["stability"]                             for e in env_names]
        row = (f"{agent:<18} | {np.mean(all_r):>8.3f} | {np.mean(all_c):>9.4f} | "
               f"{np.mean(all_csr):>5.1f}% | {np.mean(all_dh):>6.2f} | "
               f"{np.mean(all_st):>9.4f} | {transfer_scores.get(agent, 0):>8.4f}")
        if compute_stats and agent in compute_stats:
            cs   = compute_stats[agent]
            row += (f" | {cs.get('param_count', 0):>8d} | "
                    f"{cs.get('mean_infer_ms', 0):>7.3f}")
        print(row)
    print("=" * 110)

    if show_csr:
        print_csr_table(results, env_names)


# ── Multi-seed runner ─────────────────────────────────────────────────────────

def run_multiseed(build_fn, train_fn, n_seeds: int = 10,
                  seed_base: int = 0, summary_window: int = 50,
                  **train_kwargs):
    """
    Train across n_seeds independent seeds and aggregate statistics.

    Returns (all_histories, aggregated_stats)
    """
    if n_seeds < 2:
        raise ValueError(
            f"run_multiseed requires at least two independent seeds; got {n_seeds}.")
    if summary_window < 1:
        raise ValueError("summary_window must be positive")

    all_histories = []
    for i in range(n_seeds):
        seed  = seed_base + i * 100
        agent = build_fn(seed)
        hist  = train_fn(agent, seed, **train_kwargs)
        all_histories.append(hist)

    agg = {}
    from stats import bootstrap_ci
    for key in ("rewards", "consequences", "delayed_hits"):
        # One summary per independently trained model is the replicate.  A
        # single last episode is too noisy, so use a fixed final window.
        vals = [float(np.mean(h[key][-summary_window:])) for h in all_histories]
        mean, ci_lo, ci_hi = bootstrap_ci(vals, seed=seed_base)
        agg[key] = {
            "mean":   mean,
            "std":    float(np.std(vals, ddof=1)),
            "ci_lo":  ci_lo,
            "ci_hi":  ci_hi,
            "by_seed": vals,
        }

    return all_histories, agg
