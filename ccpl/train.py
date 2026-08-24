"""
train.py — Training loop utilities for CCPL
============================================
run_episode()          — CCPL agent episode runner (imported from ccpl_agent)
run_episode_baseline() — episode runner for PPO / A2C / DQN / constrained baselines
train_agent()          — full training loop with logging, ETA, memory tracking
count_parameters()     — parameter counter for any agent type
save_logs()            — persist training history to JSON
save_eval_results()    — persist evaluation results to JSON
"""

import os
import json
import time
import datetime
import tracemalloc
import warnings
import numpy as np

try:
    from .environments import ENV_REGISTRY
    from .algorithms.ccpl_agent import run_episode
except ImportError:  # Legacy checkout imports.
    from environments import ENV_REGISTRY
    from ccpl_agent import run_episode


TRAIN_ENVS       = ("standard", "noisy", "shifted", "randomised")
EVAL_ENVS        = ("standard", "noisy", "shifted")
UNSEEN_EVAL_ENVS = ("adversarial", "deceptive_reward", "resource_collapse")


# ─────────────────────────────────────────────────────────────────────────────
# Baseline episode runner  (PPO / A2C / DQN / constrained baselines)
# ─────────────────────────────────────────────────────────────────────────────

def run_episode_baseline(agent, env, train: bool = True,
                          update_freq: int = 4) -> dict:
    """
    Episode runner for on-policy and off-policy baselines.
    Clears _rollout at episode start to prevent cross-episode contamination
    (critical for PPO/A2C whose GAE assumes within-episode data).
    """
    if update_freq <= 0:
        raise ValueError("update_freq must be positive")
    if train and hasattr(agent, '_rollout'):
        agent._rollout = []

    state     = env.reset()
    ep_r = ep_c = ep_c_raw = ep_steps = 0.0
    gamma_c = 1.0
    losses    = []
    t_infer   = []

    while not env.done:
        t0     = time.perf_counter()
        action = agent.select_action(state, eval_mode=not train)
        t_infer.append(time.perf_counter() - t0)

        ns, r, c, done, info = env.step(action)

        if train:
            agent.store(state, action, r, ns, c, done)
            # On-policy agents must consume an exact rollout once it is ready;
            # checking only every fourth step skipped the first transition of
            # 64-step windows and repeatedly reused overlapping windows.
            should_update = (hasattr(agent, "_rollout")
                             or int(ep_steps) % update_freq == 0)
            if should_update:
                if agent.update():
                    for attr in ("last_policy_loss", "last_actor_loss", "last_loss"):
                        if hasattr(agent, attr):
                            losses.append(getattr(agent, attr))
                            break

        state     = ns
        ep_r     += r
        ep_c     += gamma_c * c
        ep_c_raw += c
        gamma_c  *= float(getattr(agent, "gamma", 1.0))
        ep_steps += 1

    stats = env.episode_stats()
    return {
        "episode_reward":      ep_r,
        "episode_consequence": ep_c,
        "episode_consequence_undiscounted": ep_c_raw,
        "delayed_hits":        stats["delayed_hits"],
        "steps":               int(ep_steps),
        "mean_loss":           float(np.mean(losses)) if losses else 0.0,
        "mean_infer_ms":       float(np.mean(t_infer) * 1000) if t_infer else 0.0,
        "hit_occurred":        False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Parameter counter
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(agent) -> int:
    """Count unique arrays registered with optimizers, excluding target nets.

    The former hand-written list omitted CCPL's cost critic, ICN and delay
    estimator.  Optimizer registration is the most reliable definition of a
    trainable parameter in this NumPy codebase.
    """
    from networks import Adam

    parameter_ids = set()
    object_ids = set()
    total = 0
    target_attributes = {
        "target", "target_net", "q_c_target", "q1_target", "q2_target",
        "q_cost_target", "critic_target",
    }

    def visit(obj):
        nonlocal total
        if obj is None or isinstance(obj, (str, bytes, int, float, bool, np.ndarray)):
            return
        object_id = id(obj)
        if object_id in object_ids:
            return
        object_ids.add(object_id)

        if isinstance(obj, Adam):
            for parameter in obj.params:
                parameter_id = id(parameter)
                if parameter_id not in parameter_ids:
                    parameter_ids.add(parameter_id)
                    total += int(parameter.size)
            return
        if isinstance(obj, dict):
            for value in obj.values():
                visit(value)
            return
        if isinstance(obj, (list, tuple, set)):
            for value in obj:
                visit(value)
            return
        for name, value in getattr(obj, "__dict__", {}).items():
            if name in target_attributes:
                continue
            visit(value)

    visit(agent)
    if total == 0:
        warnings.warn(
            f"No optimizer-registered parameters found on {type(agent).__name__}.",
            RuntimeWarning, stacklevel=2,
        )
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_eta(seconds: float) -> str:
    if seconds < 0:
        return "?"
    td = datetime.timedelta(seconds=int(seconds))
    h, rem = divmod(td.seconds, 3600)
    m, s   = divmod(rem, 60)
    if td.days > 0: return f"{td.days}d {h:02d}h"
    if h > 0:       return f"{h}h {m:02d}m"
    return f"{m}m {s:02d}s"


def _clean_for_json(v):
    if isinstance(v, bool):           return v
    if isinstance(v, (np.bool_,)):    return bool(v)
    if isinstance(v, (np.integer,)):  return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    if isinstance(v, np.ndarray):     return v.tolist()
    if isinstance(v, list):           return [_clean_for_json(x) for x in v]
    if isinstance(v, dict):           return {k: _clean_for_json(vv) for k, vv in v.items()}
    return v


def save_logs(history: dict, agent_name: str, log_dir: str) -> str:
    agent_dir = os.path.join(log_dir, agent_name.replace("/", "_").replace(" ", "_"))
    os.makedirs(agent_dir, exist_ok=True)
    path = os.path.join(agent_dir, "training_log.json")
    with open(path, "w") as f:
        json.dump(_clean_for_json(history), f, indent=2)
    return path


def save_eval_results(results: dict, label: str, log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"{label}.json")
    with open(path, "w") as f:
        json.dump(_clean_for_json(results), f, indent=2)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_agent(agent, n_episodes: int = 600, max_steps: int = 100,
                delay_steps: int = 5, update_freq: int = 4,
                log_freq: int = 50, seed: int = 42, verbose: bool = True,
                env_names=TRAIN_ENVS, log_dir: str = None,
                save_every: int = 100) -> dict:
    """
    Train any agent (CCPL or baseline) for n_episodes episodes.

    Rotates uniformly over env_names each episode.
    Returns a history dict with per-episode metrics.
    """
    if n_episodes < 1 or max_steps < 1:
        raise ValueError("n_episodes and max_steps must be positive")
    if delay_steps < 0 or update_freq < 1 or log_freq < 1:
        raise ValueError("delay_steps must be non-negative; frequencies must be positive")
    if not env_names:
        raise ValueError("env_names must contain at least one environment")
    is_ccpl  = hasattr(agent, "reset_hidden")
    run_ep   = run_episode if is_ccpl else run_episode_baseline
    env_list = [ENV_REGISTRY[n] for n in env_names]
    rng      = np.random.default_rng(seed)

    history  = {k: [] for k in (
        "rewards", "consequences", "delayed_hits", "losses",
        "cumulative_reward", "cumulative_consequence",
        "env_name", "episode_time_s", "infer_ms")}

    # Extended keys for CCPL agents
    if is_ccpl:
        for key in ("hit_freq_ema", "lambda_scale", "mean_lambda",
                    "expected_delay", "delay_loss"):
            history[key] = []

    cum_r = cum_c = 0.0
    ep_times: list = []

    tracemalloc.start()
    peak_mb     = 0.0
    train_start = time.time()

    for ep in range(1, n_episodes + 1):
        EnvCls = env_list[rng.integers(len(env_list))]
        env    = EnvCls(max_steps=max_steps, consequence_delay=delay_steps,
                        seed=seed + ep)

        ep_start = time.perf_counter()
        try:
            result = run_ep(agent, env, train=True, update_freq=update_freq)
        finally:
            if hasattr(env, "close"):
                env.close()
        ep_time  = time.perf_counter() - ep_start
        ep_times.append(ep_time)

        _, mem_peak = tracemalloc.get_traced_memory()
        peak_mb     = max(peak_mb, mem_peak / 1e6)

        cum_r += result["episode_reward"]
        cum_c += result["episode_consequence"]
        history["rewards"].append(result["episode_reward"])
        history["consequences"].append(result["episode_consequence"])
        history["delayed_hits"].append(result["delayed_hits"])
        history["losses"].append(result["mean_loss"])
        history["cumulative_reward"].append(cum_r)
        history["cumulative_consequence"].append(cum_c)
        history["env_name"].append(env.name)
        history["episode_time_s"].append(ep_time)
        history["infer_ms"].append(result["mean_infer_ms"])

        if is_ccpl:
            diag = agent.diagnostics() if hasattr(agent, "diagnostics") else {}
            history["hit_freq_ema"].append(diag.get("hit_freq_ema",   0.0))
            history["lambda_scale"].append(diag.get("lambda_scale",   1.0))
            history["mean_lambda"].append(diag.get("mean_lambda",     0.0))
            history["expected_delay"].append(diag.get("expected_delay", 0.0))
            history["delay_loss"].append(diag.get("delay_loss",       0.0))

        if log_dir and ep % save_every == 0:
            save_logs(history, getattr(agent, "name", "agent"), log_dir)

        if verbose and ep % log_freq == 0:
            diag    = agent.diagnostics() if hasattr(agent, "diagnostics") else {}
            avg_t   = float(np.mean(ep_times[-50:])) if ep_times else 0.0
            eta_str = _fmt_eta((n_episodes - ep) * avg_t)

            parts = [
                f"[{agent.name}] Ep {ep:4d}/{n_episodes}",
                f"ETA {eta_str}",
                f"Env: {env.name:14s}",
                f"R: {result['episode_reward']:6.2f}",
                f"C: {result['episode_consequence']:.3f}",
                f"Hits: {result['delayed_hits']:2d}",
            ]
            if is_ccpl:
                parts += [
                    f"λ: {diag.get('mean_lambda', 0):.3f}",
                    f"λ_w: {diag.get('lambda_scale', 1):.2f}",
                    f"hit%: {diag.get('hit_freq_ema', 0):.2f}",
                    f"σ: {diag.get('mean_sigma', 0):.3f}",
                ]
                if diag.get("expected_delay", 0) > 0:
                    parts.append(f"E[τ]: {diag['expected_delay']:.2f}")
            elif "epsilon" in diag:
                parts.append(f"ε: {diag['epsilon']:.3f}")
            print(" | ".join(parts))

    tracemalloc.stop()
    history["peak_memory_mb"] = peak_mb
    history["param_count"]    = count_parameters(agent)
    history["total_train_s"]  = time.time() - train_start

    if log_dir:
        path = save_logs(history, getattr(agent, "name", "agent"), log_dir)
        if verbose:
            print(f"  [log] {path}")

    return history
