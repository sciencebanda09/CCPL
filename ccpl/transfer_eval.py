"""
CCPL v6 — Upgrade 4: Cross-Environment Transfer Evaluation.

Evaluates zero-shot transfer of a trained CCPL agent to unseen environments:
  - No retraining; only forward inference.
  - Tracks reward retention, consequence retention, and λ adaptation behavior.
  - Compares whether λ dynamics remain stable across domain shifts.
  - Generates a per-agent transfer matrix and domain-shift stability report.

Key metrics:
  reward_retention   = 1 + (R_transfer - R_source)/|R_source|
                       (directionally valid even when returns are negative)
  consequence_ratio  = consequence_transfer / consequence_source
  lambda_drift       = std(λ values across transfer envs) / mean(λ values)
  stability_score    = 1 / (std(episode_rewards) + 1)

Usage:
  from transfer_eval import TransferEvaluator
  evaluator = TransferEvaluator(source_envs=TRAIN_ENVS, transfer_envs=UNSEEN_ENVS)
  results = evaluator.evaluate(agent, n_episodes=50)
  evaluator.print_report(results)
"""
import time
import numpy as np
from typing import Dict, List, Optional, Tuple



class LambdaTrajectoryLogger:
    """
    Logs λ(s) values per step during an episode.
    Used to analyse whether λ correctly adapts to domain shifts.
    """

    def __init__(self):
        self._episodes: list = []
        self._current:  list = []

    def log_step(self, lam_value: float):
        self._current.append(float(lam_value))

    def end_episode(self):
        if self._current:
            self._episodes.append(list(self._current))
        self._current = []

    def mean_lambda(self) -> float:
        all_vals = [v for ep in self._episodes for v in ep]
        return float(np.mean(all_vals)) if all_vals else 0.0

    def std_lambda(self) -> float:
        all_vals = [v for ep in self._episodes for v in ep]
        return float(np.std(all_vals)) if all_vals else 0.0

    def drift_coefficient(self) -> float:
        """CV = std / mean — proxy for how much λ adapts."""
        mu = self.mean_lambda()
        return self.std_lambda() / (mu + 1e-6)

    def trajectory_summary(self) -> dict:
        all_ep_means = [float(np.mean(ep)) for ep in self._episodes]
        return {
            "mean":   round(self.mean_lambda(), 4),
            "std":    round(self.std_lambda(),  4),
            "drift":  round(self.drift_coefficient(), 4),
            "ep_means": [round(m, 4) for m in all_ep_means],
        }

    def clear(self):
        self._episodes.clear()
        self._current.clear()



def evaluate_transfer_episode(
    agent,
    env,
    lam_logger: LambdaTrajectoryLogger = None,
    constraint_threshold: float = None,
) -> dict:
    """
    Run one evaluation episode on env with no training.
    Logs λ values if agent has lambda_net attribute.
    Returns per-episode metrics dict.
    """
    is_ccpl = hasattr(agent, "reset_hidden")
    state = env.reset()
    if is_ccpl:
        agent.reset_hidden(
            max_steps=getattr(env, "max_steps", None),
            expected_delay=getattr(env, "consequence_delay", None))

    ep_r = ep_c = ep_c_raw = 0.0
    gamma_c = 1.0
    gamma = float(getattr(agent, "gamma", 1.0))
    ep_steps    = 0
    resource_loads = []
    infer_times    = []

    while not env.done:
        t0     = time.perf_counter()
        action = agent.select_action(state, eval_mode=True)
        infer_times.append(time.perf_counter() - t0)

        ns, r, c, done, info = env.step(action)

        if lam_logger is not None and hasattr(agent, "lambda_net"):
            if getattr(agent, "_lambda_log", None):
                lam_val = float(agent._lambda_log[-1])
            else:
                s_norm = agent.normalizer.normalize(state)
                lam_val = (float(agent.lambda_net.forward(s_norm))
                           * float(getattr(agent, "lambda_scale", 1.0)))
            lam_logger.log_step(lam_val)

        ep_r += r
        ep_c += gamma_c * c
        ep_c_raw += c
        gamma_c *= gamma
        ep_steps += 1
        resource_loads.append(info.get("resource_load", 0.5))
        if is_ccpl and hasattr(agent, "observe_transition"):
            agent.observe_transition(state, action, c)
        state = ns

    if lam_logger is not None:
        lam_logger.end_episode()

    stats = env.episode_stats()
    threshold = (constraint_threshold
                 if constraint_threshold is not None
                 else getattr(env, "constraint_threshold", float("inf")))
    csr_violation = int(ep_c > threshold)

    return {
        "reward":             ep_r,
        "consequence":        ep_c,
        "consequence_undiscounted": ep_c_raw,
        "delayed_hits":       stats["delayed_hits"],
        "steps":              ep_steps,
        "mean_resource_load": float(np.mean(resource_loads)) if resource_loads else 0.5,
        "mean_infer_ms":      float(np.mean(infer_times) * 1000) if infer_times else 0.0,
        "csr_violation":      csr_violation,
    }



class TransferEvaluator:
    """
    Evaluates zero-shot transfer from source → transfer environments.

    Parameters
    ----------
    source_envs : list[str]   — environment names agent was trained on
    transfer_envs : list[str] — unseen environments for zero-shot transfer
    n_episodes : int          — episodes per environment
    max_steps : int
    delay_steps : int
    seed_offset : int
    """

    def __init__(
        self,
        source_envs:   List[str],
        transfer_envs: List[str],
        n_episodes:    int = 50,
        max_steps:     int = 100,
        delay_steps:   int = 5,
        seed_offset:   int = 88888,
    ):
        self.source_envs   = source_envs
        self.transfer_envs = transfer_envs
        self.n_episodes    = n_episodes
        self.max_steps     = max_steps
        self.delay_steps   = delay_steps
        self.seed_offset   = seed_offset
        if self.n_episodes < 1 or self.max_steps < 1 or self.delay_steps < 0:
            raise ValueError("episode/step counts must be positive and delay non-negative")
        if not self.source_envs or not self.transfer_envs:
            raise ValueError("source_envs and transfer_envs must both be non-empty")
        overlap = set(self.source_envs) & set(self.transfer_envs)
        if overlap:
            raise ValueError(f"source and transfer environments overlap: {sorted(overlap)}")

    def _evaluate_on_env(self, agent, env_name: str,
                          log_lambda: bool = True) -> dict:
        """Evaluate agent on a single named environment."""
        from environments import ENV_REGISTRY
        try:
            from adversarial_envs import ADVERSARIAL_ENV_REGISTRY
            ENV_REGISTRY.update(ADVERSARIAL_ENV_REGISTRY)
        except (ImportError, AttributeError):
            pass
        EnvClass = ENV_REGISTRY[env_name]
        lam_logger = LambdaTrajectoryLogger() if log_lambda else None

        rewards, consequences, delayed_hits = [], [], []
        resource_loads, infer_times, csr_violations = [], [], []

        for ep in range(self.n_episodes):
            env = EnvClass(max_steps=self.max_steps,
                           consequence_delay=self.delay_steps,
                           seed=self.seed_offset + ep)
            result = evaluate_transfer_episode(agent, env, lam_logger)
            rewards.append(result["reward"])
            consequences.append(result["consequence"])
            delayed_hits.append(result["delayed_hits"])
            resource_loads.append(result["mean_resource_load"])
            infer_times.append(result["mean_infer_ms"])
            csr_violations.append(result["csr_violation"])
            if hasattr(env, "close"):
                env.close()

        csr = 100.0 * (1.0 - float(np.mean(csr_violations)))

        out = {
            "mean_reward":       float(np.mean(rewards)),
            "std_reward":        float(np.std(rewards)),
            "mean_consequence":  float(np.mean(consequences)),
            "std_consequence":   float(np.std(consequences)),
            "mean_delayed_hits": float(np.mean(delayed_hits)),
            "stability":         float(1.0 / (np.std(rewards) + 1.0)),
            "mean_infer_ms":     float(np.mean(infer_times)),
            "csr":               csr,
            "rewards":           rewards,
            "consequences":      consequences,
        }
        if lam_logger is not None:
            out["lambda_summary"] = lam_logger.trajectory_summary()

        return out

    def evaluate(self, agent, verbose: bool = True) -> dict:
        """
        Full transfer evaluation.

        Returns dict:
          {
            "source":   {env_name: metrics},
            "transfer": {env_name: metrics},
            "retention": {metric: value},
            "lambda_stability": {env_name: drift_coefficient},
          }
        """
        if verbose:
            print(f"\n[Transfer] Evaluating {getattr(agent, 'name', 'agent')} ...")

        source_results   = {}
        transfer_results = {}

        for env_name in self.source_envs:
            if verbose:
                print(f"  Source env:   {env_name}")
            source_results[env_name] = self._evaluate_on_env(agent, env_name)

        for env_name in self.transfer_envs:
            if verbose:
                print(f"  Transfer env: {env_name}")
            transfer_results[env_name] = self._evaluate_on_env(agent, env_name)

        src_r   = float(np.mean([v["mean_reward"]      for v in source_results.values()]))
        src_c   = float(np.mean([v["mean_consequence"] for v in source_results.values()]))
        xfr_r   = float(np.mean([v["mean_reward"]      for v in transfer_results.values()]))
        xfr_c   = float(np.mean([v["mean_consequence"] for v in transfer_results.values()]))

        reward_retention = 1.0 + (xfr_r - src_r) / (abs(src_r) + 1e-6)
        retention = {
            "source_mean_reward":       round(src_r, 4),
            "transfer_mean_reward":     round(xfr_r, 4),
            "reward_retention":         round(reward_retention, 4),
            "reward_change":            round(xfr_r - src_r, 4),
            "source_mean_consequence":  round(src_c, 4),
            "transfer_mean_consequence":round(xfr_c, 4),
            "consequence_ratio":        round(xfr_c / (src_c + 1e-6), 4),
            "transfer_stability":       round(
                float(np.mean([v["stability"] for v in transfer_results.values()])), 4),
        }

        lambda_stability = {}
        for env_name, res in transfer_results.items():
            ls = res.get("lambda_summary", {})
            lambda_stability[env_name] = {
                "mean":  ls.get("mean",  0.0),
                "std":   ls.get("std",   0.0),
                "drift": ls.get("drift", 0.0),
            }

        overall_lambda_drift = float(np.mean(
            [v["drift"] for v in lambda_stability.values()])) if lambda_stability else 0.0

        return {
            "source":          source_results,
            "transfer":        transfer_results,
            "retention":       retention,
            "lambda_stability":lambda_stability,
            "overall_lambda_drift": overall_lambda_drift,
        }

    def print_report(self, results: dict):
        """Print a formatted transfer evaluation report."""
        print("\n" + "="*80)
        print("  ZERO-SHOT TRANSFER EVALUATION REPORT")
        print("="*80)

        print("\n  Source environments (trained on):")
        for env, res in results["source"].items():
            print(f"    {env:<22} R={res['mean_reward']:>7.3f}  "
                  f"C={res['mean_consequence']:>7.4f}  "
                  f"CSR={res['csr']:>5.1f}%")

        print("\n  Transfer environments (zero-shot):")
        for env, res in results["transfer"].items():
            lam = res.get("lambda_summary", {})
            print(f"    {env:<22} R={res['mean_reward']:>7.3f}  "
                  f"C={res['mean_consequence']:>7.4f}  "
                  f"CSR={res['csr']:>5.1f}%  "
                  f"λ_drift={lam.get('drift', 0):>6.4f}")

        r = results["retention"]
        print("\n  Retention summary:")
        print(f"    Reward retention index:   {r['reward_retention']:.4f}  "
              f"({r['source_mean_reward']:.3f} → {r['transfer_mean_reward']:.3f})")
        print(f"    Consequence ratio:        {r['consequence_ratio']:.4f}  "
              f"({r['source_mean_consequence']:.4f} → {r['transfer_mean_consequence']:.4f})")
        print(f"    Transfer stability:       {r['transfer_stability']:.4f}")
        print(f"    Overall λ drift (CV):     {results['overall_lambda_drift']:.4f}")
        print("="*80)



def compare_transfer(agents: dict, source_envs: List[str],
                     transfer_envs: List[str],
                     n_episodes: int = 50,
                     max_steps: int = 100) -> dict:
    """
    Compare multiple agents on zero-shot transfer.

    agents: {name: agent_object}
    Returns: {agent_name: transfer_evaluation_dict}
    """
    evaluator = TransferEvaluator(source_envs, transfer_envs,
                                  n_episodes=n_episodes, max_steps=max_steps)
    all_results = {}
    for name, agent in agents.items():
        print(f"\n[TransferCompare] Agent: {name}")
        all_results[name] = evaluator.evaluate(agent, verbose=True)
    return all_results


def print_transfer_comparison(all_results: dict):
    """Compact comparison table across agents."""
    print("\n" + "="*90)
    print("  MULTI-AGENT TRANSFER COMPARISON")
    print("="*90)
    print(f"  {'Agent':<18} {'Reward↑':>9} {'Consequence↓':>13} "
          f"{'Retention↑':>11} {'CSR↑':>7} {'λ Drift↓':>9}")
    print("  " + "-"*85)

    for name, res in all_results.items():
        r    = res["retention"]
        xfr  = res["transfer"]
        csr  = float(np.mean([v["csr"] for v in xfr.values()])) if xfr else 0
        drift = res.get("overall_lambda_drift", 0)
        print(f"  {name:<18} {r['transfer_mean_reward']:>9.3f} "
              f"{r['transfer_mean_consequence']:>13.4f} "
              f"{r['reward_retention']:>11.4f} "
              f"{csr:>6.1f}% "
              f"{drift:>9.4f}")

    print("="*90)
