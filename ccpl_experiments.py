"""
ccpl_experiments.py — Extended Experimental Suite
==================================================
Nine experiments designed to make CCPL unquestionable:

  E1  Full baseline comparison (9 agents × 7 envs × 5 seeds)
  E2  Ablation — isolates contribution of each direction
  E3  Causal attribution accuracy (ICN vs SCM ground truth)
  E4  Delay distribution calibration (P(τ|h) vs observed τ)
  E5  Adversarial robustness (DeceptionBench, HiddenStateShift, ConflictZone)
  E6  Sample efficiency (reward/CSR vs training steps)
  E7  Transfer (zero-shot to unseen environments)
  E8  Optional Safety Gymnasium evaluation (requires safety-gymnasium)
  E9  Safety Gymnasium implementation ablations

Run a single experiment:
  python ccpl_experiments.py --exp E1 --seeds 3 --episodes 500
  python ccpl_experiments.py --exp E8 --seeds 3 --episodes 500   # needs safety-gymnasium

Run all:
  python ccpl_experiments.py --all --seeds 5 --episodes 1000
"""

import os, time, json, argparse
import sys
from pathlib import Path
import numpy as np
import warnings

ROOT = Path(__file__).resolve().parent
for _source in (ROOT / "ccpl", ROOT / "ccpl" / "algorithms", ROOT / "ccpl" / "environments"):
    _source = str(_source)
    if _source not in sys.path:
        sys.path.insert(0, _source)

from plots import (
    generate_all_plots,
    plot_ci_reward_curves,
    plot_convergence_speed,
    plot_ablation_comparison,
    plot_per_env_ranking,
    plot_transfer_score,
    plot_unseen_transfer,
    plot_sample_efficiency,
    plot_final_ranking,
    plot_ccpl_diagnostics,
)
warnings.filterwarnings('ignore')

import adversarial_envs
from adversarial_envs import SAFETY_GYM_ENV_REGISTRY, _SAFETY_GYM_AVAILABLE
from environments   import ENV_REGISTRY
from ccpl_agent     import make_ccpl, run_episode, build_ccpl_ablation, CCPLAgent, make_ccpl_base
from ppo_agent      import PPOAgent
from a2c_agent      import A2CAgent
from dqn_agent      import DQNAgent
from constrained_baselines import (CPOAgent, RCPOAgent,
                                    PIDLagrangianAgent, SACLagrangianAgent)
from train          import train_agent, run_episode_baseline, count_parameters, save_eval_results
from evaluate       import evaluate_all, compute_transfer_score, print_benchmark_table, print_csr_table
from ccpl_theory    import run_all_theory_checks, print_theory_report
from causal_graph   import EnvironmentSCM, CausalLabelGenerator
from stats          import paired_randomization, holm_correction, run_full_stats
from transfer_eval  import TransferEvaluator


STATE_DIM  = 6
ACTION_DIM = 5
TRAIN_ENVS = ("standard", "noisy", "shifted", "randomised")
EVAL_ENVS  = ("standard", "noisy", "shifted")
UNSEEN_ENVS= ("adversarial", "deceptive_reward", "resource_collapse",
               "deception_bench", "hidden_state_shift", "conflict_zone")


def _sep(title="", w=72):
    print("\n" + "=" * w)
    if title:
        print(f"  {title}")
        print("=" * w)


def _save(data, name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.json")

    def _clean(v):
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating,)): return float(v)
        if isinstance(v, np.ndarray):    return v.tolist()
        if isinstance(v, dict):          return {k: _clean(vv) for k, vv in v.items()}
        if isinstance(v, list):          return [_clean(x) for x in v]
        return v

    with open(path, "w") as f:
        json.dump(_clean(data), f, indent=2)
    print(f"  Saved: {path}")
    return path


def _seed_values(args):
    if args.seed_values:
        values = [int(value.strip()) for value in args.seed_values.split(",") if value.strip()]
        if len(values) != args.seeds:
            raise ValueError("--seed-values must contain exactly --seeds comma-separated integers")
        return values
    return [args.seed + index * 100 for index in range(args.seeds)]


def _build_all_baselines(seed):
    """Build all 9 comparison agents."""
    return {
        "CCPL":    make_ccpl(STATE_DIM, ACTION_DIM, seed=seed, pretrain_steps=200),
        "PPO":     PPOAgent(STATE_DIM, ACTION_DIM, seed=seed),
        "A2C":     A2CAgent(STATE_DIM, ACTION_DIM, seed=seed),
        "DQN":     DQNAgent(STATE_DIM, ACTION_DIM, seed=seed),
        "CPO-FO":  CPOAgent(STATE_DIM, ACTION_DIM, seed=seed),
        "RCPO":    RCPOAgent(STATE_DIM, ACTION_DIM, seed=seed),
        "PID-Lag": PIDLagrangianAgent(STATE_DIM, ACTION_DIM, seed=seed),
        "SAC-Lag": SACLagrangianAgent(STATE_DIM, ACTION_DIM, seed=seed),
        "CCPL-Base": make_ccpl_base(STATE_DIM, ACTION_DIM, seed=seed),
    }



def run_E1(args):
    _sep("E1 — Full Baseline Comparison (9 agents × 6 envs × seeds)")
    out = os.path.join(args.out, "E1")
    os.makedirs(out, exist_ok=True)

    eval_envs   = list(EVAL_ENVS) + ["adversarial", "deceptive_reward", "resource_collapse"]
    all_results = {}
    all_histories = {}

    for seed_i, seed in enumerate(_seed_values(args)):
        _sep(f"  Seed {seed_i+1}/{args.seeds}  (seed={seed})")
        agents = _build_all_baselines(seed)

        for name, agent in agents.items():
            print(f"  Training {name}...")
            h = train_agent(agent, n_episodes=args.episodes, max_steps=args.max_steps,
                            delay_steps=args.delay, seed=seed,
                            verbose=args.verbose,
                            log_freq=max(1, args.episodes // 5),
                            env_names=list(TRAIN_ENVS))
            all_histories.setdefault(name, []).append(h)

        res = evaluate_all(agents, eval_envs, args.eval_episodes,
                           args.max_steps, args.delay)
        for name, r in res.items():
            all_results.setdefault(name, []).append(r)

    def _avg(res_list, envs):
        return {e: {k: float(np.mean([r[e][k] for r in res_list]))
                    for k in res_list[0][e]
                    if isinstance(res_list[0][e].get(k), (int, float))}
                for e in envs}

    averaged = {n: _avg(v, eval_envs) for n, v in all_results.items()}
    xfer     = compute_transfer_score(averaged, eval_envs)

    print_benchmark_table(averaged, eval_envs, xfer)
    print_csr_table(averaged, eval_envs)

    _sep("  Seed-level Statistical Tests (trained seed is the replicate)")
    ccpl_rewards = [
        float(np.mean([seed_result[e]["mean_reward"] for e in eval_envs]))
        for seed_result in all_results["CCPL"]
    ]
    comparisons = []
    for name in averaged:
        if name == "CCPL": continue
        base_rewards = [
            float(np.mean([seed_result[e]["mean_reward"] for e in eval_envs]))
            for seed_result in all_results[name]
        ]
        comparisons.append((name, paired_randomization(
            ccpl_rewards, base_rewards, alternative="greater")))
    valid = [(name, test) for name, test in comparisons if test.get("valid")]
    corrections = holm_correction([test["p"] for _, test in valid])
    corrected = {name: correction for (name, _), correction in zip(valid, corrections)}
    for name, test in comparisons:
        if not test.get("valid"):
            print(f"    CCPL vs {name:<12}: descriptive only ({test['note']})")
            continue
        is_significant = corrected[name]["significant"]
        print(f"    CCPL > {name:<12}: paired p={test['p']:.4f}  "
              f"r={test['r']:.3f}  Holm={'yes' if is_significant else 'no'}")

    _save({"results": averaged, "seed_results": all_results,
           "seed_values": _seed_values(args),
           "transfer": xfer}, "E1_results", out)

    flat_histories = {name: hlist[-1] for name, hlist in all_histories.items()}
    generate_all_plots(
        histories          = flat_histories,
        eval_results       = averaged,
        transfer_scores    = xfer,
        eval_envs          = eval_envs,
        out_dir            = out,
        all_seed_histories = all_histories,
    )
    return averaged, all_histories



def run_E2(args):
    _sep("E2 — Ablation: Contribution of Each Direction")
    out = os.path.join(args.out, "E2")
    os.makedirs(out, exist_ok=True)

    eval_envs   = list(EVAL_ENVS) + ["adversarial", "deceptive_reward"]
    all_results = {}

    for seed_i, seed in enumerate(_seed_values(args)):
        print(f"\n  Seed {seed_i+1}/{args.seeds}")
        variants = build_ccpl_ablation(STATE_DIM, ACTION_DIM, seed=seed)

        for name, agent in variants.items():
            print(f"  Training {name}...")
            train_agent(agent, n_episodes=args.episodes, max_steps=args.max_steps,
                        delay_steps=args.delay, seed=seed, verbose=args.verbose,
                        log_freq=max(1, args.episodes // 4),
                        env_names=list(TRAIN_ENVS))

        res = evaluate_all(variants, eval_envs, args.eval_episodes,
                           args.max_steps, args.delay)
        for name, r in res.items():
            all_results.setdefault(name, []).append(r)

    def _avg(res_list, envs):
        return {e: {k: float(np.mean([r[e][k] for r in res_list]))
                    for k in res_list[0][e]
                    if isinstance(res_list[0][e].get(k), (int, float))}
                for e in envs}

    averaged = {n: _avg(v, eval_envs) for n, v in all_results.items()}
    xfer     = compute_transfer_score(averaged, eval_envs)

    print_benchmark_table(averaged, eval_envs, xfer)

    _sep("  Marginal Contribution of Each Direction")
    base_r = {e: averaged.get("CCPL-Base", averaged.get("CCPL", {})).get(e, {}).get("mean_reward", 0)
               for e in eval_envs}
    base_c = {e: averaged.get("CCPL-Base", averaged.get("CCPL", {})).get(e, {}).get("mean_consequence", 0)
               for e in eval_envs}

    print(f"  {'Variant':<20}  {'ΔReward':>10}  {'ΔJ_c':>10}  {'ΔCSR%':>8}")
    print("  " + "-" * 55)
    for name, res in averaged.items():
        if "Base" in name: continue
        dr   = np.mean([res[e]["mean_reward"]      - base_r[e] for e in eval_envs])
        dc   = np.mean([res[e]["mean_consequence"] - base_c[e] for e in eval_envs])
        dcsr = np.mean([res[e].get("constraint_satisfaction_rate", 0) -
                        averaged.get("CCPL-Base", averaged.get("CCPL", {})).get(e, {}).get(
                            "constraint_satisfaction_rate", 0)
                        for e in eval_envs])
        print(f"  {name:<20}  {dr:>+10.3f}  {dc:>+10.4f}  {dcsr:>+7.1f}%")

    _save({"results": averaged, "seed_results": all_results,
           "seed_values": _seed_values(args),
           "transfer": xfer}, "E2_ablation", out)

    plot_ablation_comparison(averaged, eval_envs, out)
    plot_final_ranking(averaged, eval_envs, out)
    return averaged



def run_E3(args):
    _sep("E3 — ICN Agreement with the Synthetic One-Step SCM Reference")
    out = os.path.join(args.out, "E3")
    os.makedirs(out, exist_ok=True)
    action_names = ["DEFER", "PARTIAL", "FULL", "INVEST", "REBALANCE"]
    checkpoints = list(range(0, args.episodes + 1, 50))
    if checkpoints[-1] != args.episodes:
        checkpoints.append(args.episodes)
    scm = EnvironmentSCM()
    seed_results = {}
    for seed in _seed_values(args):
        rng = np.random.default_rng(seed)
        states = rng.uniform(0.1, 0.9, (500, STATE_DIM)).astype(np.float32)
        actions = rng.integers(0, ACTION_DIM, 500).astype(np.int32)
        labels = CausalLabelGenerator(scm).generate_batch(states, actions, ACTION_DIM)
        scm_dc = labels["delta_C_scm"]
        agent = make_ccpl(STATE_DIM, ACTION_DIM, seed=seed, pretrain_steps=200)
        seed_results[str(seed)] = {}
        trained = 0
        for n_ep in checkpoints:
            from environments import StandardEnv
            for ep in range(trained, n_ep):
                run_episode(agent, StandardEnv(max_steps=100, seed=seed + ep), train=True)
            trained = n_ep
            ctx = np.zeros((500, agent.icn.causal_dim), np.float32)
            delta_C, _, _, _ = agent.icn.forward(states, actions, ctx)
            mae = float(np.abs(delta_C - scm_dc).mean())
            corr = float(np.corrcoef(delta_C, scm_dc)[0, 1]) if np.std(delta_C) > 1e-12 else 0.0
            sign_ag = float(np.mean(np.sign(delta_C) == np.sign(scm_dc)))
            per_action = {}
            for a in range(ACTION_DIM):
                mask = actions == a
                if mask.sum() > 5:
                    per_action[action_names[a]] = {
                        "mae": round(float(np.abs(delta_C[mask] - scm_dc[mask]).mean()), 5),
                        "mean_scm": round(float(scm_dc[mask].mean()), 5),
                        "mean_icn": round(float(delta_C[mask].mean()), 5),
                    }
            seed_results[str(seed)][str(n_ep)] = {
                "mae": round(mae, 5), "corr": round(corr, 4),
                "sign_agreement": round(sign_ag, 4), "per_action": per_action,
            }
            print(f"  Seed {seed} after {n_ep:4d} eps: MAE={mae:.5f}  Corr={corr:.4f}  SignAg={sign_ag:.4f}")

    results_by_episodes = {}
    for n_ep in checkpoints:
        points = [seed_results[str(seed)][str(n_ep)] for seed in _seed_values(args)]
        aggregate = {}
        for metric in ("mae", "corr", "sign_agreement"):
            values = np.asarray([point[metric] for point in points], dtype=float)
            aggregate[metric] = round(float(values.mean()), 5 if metric == "mae" else 4)
            aggregate[f"{metric}_std"] = round(float(values.std(ddof=1)) if len(values) > 1 else 0.0, 5)
        aggregate["per_action"] = points[-1]["per_action"]
        results_by_episodes[n_ep] = aggregate

    _sep("  Per-Action ICN vs SCM (final checkpoint)")
    final = results_by_episodes[max(results_by_episodes.keys())]
    print(f"  {'Action':<12}  {'SCM ΔC':>10}  {'ICN ΔC':>10}  {'MAE':>8}")
    for aname, v in final["per_action"].items():
        print(f"  {aname:<12}  {v['mean_scm']:>+10.5f}  {v['mean_icn']:>+10.5f}  {v['mae']:>8.5f}")

    _sep("  Synthetic-reference summary")
    positive_actions = [
        name for name, values in final["per_action"].items()
        if values["mean_scm"] > 0.0
    ]
    print(f"  Positive mean reference contrasts: {positive_actions or ['none']}")
    print("  This is agreement with a specified simulator model, not causal")
    print("  identification or a comparison against every prior method.")

    _save({"protocol": {"seeds": _seed_values(args), "episodes": args.episodes,
                          "reference": "synthetic SCM; not observational causal discovery"},
           "aggregate": results_by_episodes, "seed_results": seed_results},
          "E3_causal_accuracy", out)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    eps = sorted(results_by_episodes.keys())
    axes[0].errorbar(eps, [results_by_episodes[e]["corr"] for e in eps],
                     yerr=[results_by_episodes[e]["corr_std"] for e in eps], marker="o", capsize=3)
    axes[0].set_title("ICN vs SCM Correlation", fontweight="bold")
    axes[0].set_xlabel("Episodes"); axes[0].set_ylabel("Pearson r"); axes[0].grid(alpha=0.25)
    axes[1].errorbar(eps, [results_by_episodes[e]["mae"] for e in eps],
                     yerr=[results_by_episodes[e]["mae_std"] for e in eps], marker="o", color="tomato", capsize=3)
    axes[1].set_title("ICN MAE vs SCM", fontweight="bold")
    axes[1].set_xlabel("Episodes"); axes[1].set_ylabel("MAE"); axes[1].grid(alpha=0.25)
    axes[2].errorbar(eps, [results_by_episodes[e]["sign_agreement"] for e in eps],
                     yerr=[results_by_episodes[e]["sign_agreement_std"] for e in eps], marker="o", color="green", capsize=3)
    axes[2].set_title("Sign Agreement (ICN vs SCM)", fontweight="bold")
    axes[2].set_xlabel("Episodes"); axes[2].set_ylabel("Fraction"); axes[2].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{out}/E3_causal_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}/E3_causal_accuracy.png")
    return results_by_episodes



def run_E4(args):
    _sep("E4 — Delay Distribution Calibration: P(τ|h) vs Observed Delays")
    out = os.path.join(args.out, "E4")
    os.makedirs(out, exist_ok=True)

    from environments import StandardEnv, ShiftedConsequenceEnv
    agent = make_ccpl(STATE_DIM, ACTION_DIM, seed=args.seed, pretrain_steps=200)

    # Train on the same consequence_delay used for evaluation below (10, per
    # ShiftedConsequenceEnv), not StandardEnv's default of 5 — otherwise the
    # delay estimator is scored against a delay value it never trained on.
    for ep in range(min(args.episodes, 300)):
        env = ShiftedConsequenceEnv(max_steps=100, seed=ep)
        run_episode(agent, env, train=True)

    observed_taus  = []
    predicted_taus = []

    for ep in range(100):
        env   = ShiftedConsequenceEnv(max_steps=100, seed=9000 + ep)
        state = env.reset()
        agent.reset_hidden(
            max_steps=env.max_steps,
            expected_delay=env.consequence_delay)
        step   = 0
        predictions_by_step = {}

        while not env.done:
            action = agent.select_action(state, eval_mode=True)
            h_flat = agent._h.squeeze(0) if agent._h.ndim > 1 else agent._h
            predictions_by_step[step] = agent.delay_dist.expected_tau(h_flat[None])
            ns, r, c, done, info = env.step(action)
            if info.get("delay_supervision_valid", False):
                observed_tau = info.get("actual_tau")
                source_step = (step - int(observed_tau)
                               if observed_tau is not None else None)
                if source_step is not None and source_step in predictions_by_step:
                    observed_taus.append(float(observed_tau))
                    predicted_taus.append(predictions_by_step[source_step])
            agent.observe_transition(state, action, c)
            state = ns
            step += 1

    if len(observed_taus) > 10:
        obs   = np.array(observed_taus, np.float32)
        pred  = np.array(predicted_taus, np.float32)
        mae   = float(np.abs(obs - pred).mean())
        corr  = (float(np.corrcoef(obs, pred)[0, 1])
                 if obs.std() > 1e-8 and pred.std() > 1e-8 else None)

        print(f"  Observed τ: mean={obs.mean():.2f}  std={obs.std():.2f}")
        print(f"  Predicted τ: mean={pred.mean():.2f}  std={pred.std():.2f}")
        corr_text = f"{corr:.4f}" if corr is not None else "n/a (constant delay)"
        print(f"  MAE={mae:.3f}  Correlation={corr_text}")

        result = {
            "obs_mean": round(float(obs.mean()), 4),
            "pred_mean": round(float(pred.mean()), 4),
            "mae":  round(mae, 4),
            "corr": round(corr, 4) if corr is not None else None,
            "calibrated": mae < agent.tau_max / 4.0,
        }
    else:
        result = {"note": "Insufficient consequence observations in test episodes."}
        print("  Note: Low consequence frequency in test environment.")

    rng    = np.random.default_rng(0)
    h_test = rng.standard_normal((32, agent.gru_dim)).astype(np.float32)
    probs  = agent.delay_dist.forward(h_test).mean(0)
    print("\n  Mean delay distribution P(τ|h) across 32 hidden states:")
    for k in range(min(16, len(probs))):
        bar = "█" * int(probs[k] * 50)
        print(f"  τ={k:2d}: {bar:<50} {probs[k]:.4f}")

    _save(result, "E4_delay_calibration", out)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(probs)), probs, color="#2563EB", alpha=0.85)
    ax.set_xlabel("Delay τ"); ax.set_ylabel("P(τ|h)")
    ax.set_title("Learned Delay Distribution P(τ|h)", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{out}/E4_delay_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}/E4_delay_distribution.png")
    return result



def run_E5(args):
    _sep("E5 — Adversarial Robustness (3 adversarial environments)")
    out = os.path.join(args.out, "E5")
    os.makedirs(out, exist_ok=True)

    adv_envs = ["deception_bench", "hidden_state_shift", "conflict_zone"]
    agents = {
        "CCPL":     make_ccpl(STATE_DIM, ACTION_DIM, seed=args.seed, pretrain_steps=200),
        "PPO":      PPOAgent(STATE_DIM, ACTION_DIM, seed=args.seed),
        "SAC-Lag":  SACLagrangianAgent(STATE_DIM, ACTION_DIM, seed=args.seed),
        "CCPL-Base":make_ccpl_base(STATE_DIM, ACTION_DIM, seed=args.seed),
    }

    for name, agent in agents.items():
        print(f"  Training {name} on standard envs...")
        train_agent(agent, n_episodes=args.episodes, max_steps=args.max_steps,
                    delay_steps=args.delay, seed=args.seed,
                    verbose=args.verbose, log_freq=max(1, args.episodes // 4),
                    env_names=list(TRAIN_ENVS))

    results = evaluate_all(agents, adv_envs, args.eval_episodes,
                           args.max_steps, args.delay)
    ts      = compute_transfer_score(results, adv_envs)
    print_benchmark_table(results, adv_envs, ts)
    print_csr_table(results, adv_envs)

    _sep("  DeceptionBench action-attribution stress test")
    print("  DeceptionBench rewards FULL highly but accumulates hidden penalty.")
    print("  Compare whether each trained policy avoids that delayed failure mode.")
    for name, res in results.items():
        r   = res.get("deception_bench", {}).get("mean_reward", 0)
        csr = res.get("deception_bench", {}).get("constraint_satisfaction_rate", 0)
        print(f"    {name:<14}: R={r:+.3f}  CSR={csr:.1f}%")

    _save({"results": {n: {e: v for e, v in r.items()} for n, r in results.items()},
           "transfer": ts}, "E5_adversarial", out)

    plot_unseen_transfer(results, adv_envs, out)
    plot_transfer_score(ts, out, title="E5 — Adversarial Robustness Transfer Score")
    plot_per_env_ranking(results, adv_envs, out)
    return results



def run_E6(args):
    _sep("E6 — Sample Efficiency: Reward & CSR vs Training Steps")
    out = os.path.join(args.out, "E6")
    os.makedirs(out, exist_ok=True)

    checkpoints = [50, 100, 200, 300, 500]
    checkpoints = [c for c in checkpoints if c <= args.episodes]

    agents_configs = {
        "CCPL":      lambda: make_ccpl(STATE_DIM, ACTION_DIM, seed=args.seed, pretrain_steps=100),
        "PPO":       lambda: PPOAgent(STATE_DIM, ACTION_DIM, seed=args.seed),
        "SAC-Lag":   lambda: SACLagrangianAgent(
            STATE_DIM, ACTION_DIM, seed=args.seed),
        "CCPL-Base": lambda: make_ccpl_base(STATE_DIM, ACTION_DIM, seed=args.seed),
    }

    efficiency = {name: [] for name in agents_configs}

    for name, build_fn in agents_configs.items():
        print(f"  {name}...")
        agent = build_fn()
        prev_ep = 0
        for ckpt in checkpoints:
            n_ep = ckpt - prev_ep
            train_agent(agent, n_episodes=n_ep, max_steps=args.max_steps,
                        delay_steps=args.delay, seed=args.seed + prev_ep,
                        verbose=False, env_names=list(TRAIN_ENVS))
            res = evaluate_all({name: agent}, list(EVAL_ENVS),
                               20, args.max_steps, args.delay)
            r   = np.mean([res[name][e]["mean_reward"] for e in EVAL_ENVS])
            csr = np.mean([res[name][e]["constraint_satisfaction_rate"] for e in EVAL_ENVS])
            efficiency[name].append({"episodes": ckpt, "reward": round(r, 4), "csr": round(csr, 2)})
            print(f"    ep {ckpt:4d}: R={r:+.3f}  CSR={csr:.1f}%")
            prev_ep = ckpt

    _sep("  Sample Efficiency Summary")
    print(f"  {'Agent':<14}", end="")
    for c in checkpoints:
        print(f"  ep{c:>4}R ", end="")
    print()
    for name, curve in efficiency.items():
        print(f"  {name:<14}", end="")
        for pt in curve:
            print(f"  {pt['reward']:>8.3f}", end="")
        print()

    _save(efficiency, "E6_sample_efficiency", out)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, curve in efficiency.items():
        eps  = [pt["episodes"] for pt in curve]
        rews = [pt["reward"]   for pt in curve]
        csrs = [pt["csr"]      for pt in curve]
        from plots import COLORS
        c = COLORS.get(name, "#888")
        axes[0].plot(eps, rews, marker="o", label=name, color=c, lw=1.8)
        axes[1].plot(eps, csrs, marker="o", label=name, color=c, lw=1.8)
    for ax, ylabel, title in [
        (axes[0], "Mean Reward", "E6 — Sample Efficiency: Reward"),
        (axes[1], "CSR %",       "E6 — Sample Efficiency: CSR"),
    ]:
        ax.set_xlabel("Training Episodes"); ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{out}/E6_sample_efficiency.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}/E6_sample_efficiency.png")
    return efficiency



def run_E7(args):
    _sep("E7 — Zero-Shot Transfer to Unseen Environments")
    out = os.path.join(args.out, "E7")
    os.makedirs(out, exist_ok=True)

    agents = {
        "CCPL":      make_ccpl(STATE_DIM, ACTION_DIM, seed=args.seed, pretrain_steps=200),
        "PPO":       PPOAgent(STATE_DIM, ACTION_DIM, seed=args.seed),
        "SAC-Lag":   SACLagrangianAgent(
            STATE_DIM, ACTION_DIM, seed=args.seed),
        "CCPL-Base": make_ccpl_base(STATE_DIM, ACTION_DIM, seed=args.seed),
    }

    for name, agent in agents.items():
        print(f"  Training {name} on source envs...")
        train_agent(agent, n_episodes=args.episodes, max_steps=args.max_steps,
                    delay_steps=args.delay, seed=args.seed,
                    verbose=args.verbose, log_freq=max(1, args.episodes // 4),
                    env_names=list(TRAIN_ENVS))

    evaluator = TransferEvaluator(
        source_envs   = list(TRAIN_ENVS[:3]),
        transfer_envs = list(UNSEEN_ENVS[:4]),
        n_episodes    = args.eval_episodes,
        max_steps     = args.max_steps,
        delay_steps   = args.delay,
    )

    all_transfer = {}
    for name, agent in agents.items():
        print(f"\n  Transfer eval: {name}")
        res = evaluator.evaluate(agent, verbose=False)
        evaluator.print_report(res)
        all_transfer[name] = res

    _sep("  Transfer Summary")
    print(f"  {'Agent':<14}  {'Src CSR':>8}  {'Xfr CSR':>8}  {'CSR Drop':>9}  {'R-Ret':>7}")
    print("  " + "-" * 55)
    for name, res in all_transfer.items():
        ret  = res["retention"]
        xfr  = res.get("transfer", {})
        src_csr = np.mean([v.get("csr", 0) for v in res.get("source", {}).values()])
        xfr_csr = np.mean([v.get("csr", 0) for v in xfr.values()]) if xfr else 0
        drop    = src_csr - xfr_csr
        r_ret   = ret.get("reward_retention", 0)
        print(f"  {name:<14}  {src_csr:>7.1f}%  {xfr_csr:>7.1f}%  {drop:>+8.1f}%  {r_ret:>7.4f}")

    _save({n: r["retention"] for n, r in all_transfer.items()},
          "E7_transfer", out)

    unseen_plot = {name: res.get("transfer", {}) for name, res in all_transfer.items()}
    transfer_envs_list = list(UNSEEN_ENVS[:4])
    plot_unseen_transfer(unseen_plot, transfer_envs_list, out)
    retention_scores = {n: r["retention"].get("reward_retention", 0) for n, r in all_transfer.items()}
    plot_transfer_score(retention_scores, out, title="E7 — Zero-Shot Transfer: Reward Retention")
    return all_transfer



def run_theory_verification(args):
    _sep("Implementation and Empirical Diagnostics (Not Theorem Verification)")

    agent = make_ccpl(STATE_DIM, ACTION_DIM, seed=args.seed, pretrain_steps=200)
    histories = {}

    print(f"  Training {args.episodes} episodes across all train envs...")
    h = train_agent(agent, n_episodes=args.episodes, max_steps=args.max_steps,
                    delay_steps=args.delay, seed=args.seed,
                    verbose=args.verbose, log_freq=max(1, args.episodes // 5),
                    env_names=list(TRAIN_ENVS))
    histories["all"] = h

    eval_envs = list(EVAL_ENVS) + ["adversarial", "deceptive_reward"]
    results   = evaluate_all({"CCPL": agent}, eval_envs,
                              args.eval_episodes, args.max_steps, args.delay)

    checks = run_all_theory_checks(agent, histories=histories,
                                    eval_results=results, constraint_d=3.0)
    print_theory_report(checks)

    out = os.path.join(args.out, "theory")
    _save(checks, "theory_verification", out)
    return checks



def run_E8(args):
    """
    E8: Full Safety Gymnasium benchmark.
    Trains four representative agents on the registered Safety Gymnasium tasks
    and compares held-out reward, discounted cost, and constraint satisfaction.

    Each Safety Gym task uses a neutral-plus-signed-axis discretisation of its
    continuous action space.

    No synthetic SCM labels are used for these tasks.  Raw Safety Gymnasium
    costs are optionally displaced by the deterministic delayed wrapper.
    """
    _sep(f"E8 — Safety Gymnasium Evaluation ({args.delay_mode} costs)")

    if not _SAFETY_GYM_AVAILABLE:
        print("  safety-gymnasium not installed. Run: pip install safety-gymnasium")
        print("  Skipping E8.")
        return {}

    out = os.path.join(args.out, "E8")
    os.makedirs(out, exist_ok=True)

    requested_tasks = getattr(args, "tasks", "")
    requested_tasks = [name.strip() for name in requested_tasks.split(",") if name.strip()]
    sg_tasks = list(SAFETY_GYM_ENV_REGISTRY.keys())
    if requested_tasks:
        unknown = sorted(set(requested_tasks) - set(sg_tasks))
        if unknown:
            raise ValueError(f"Unknown Safety Gymnasium tasks: {unknown}")
        sg_tasks = [name for name in sg_tasks if name in requested_tasks]
    if not sg_tasks:
        print("  No Safety Gym tasks registered. Skipping E8.")
        return {}

    print(f"  Tasks: {sg_tasks}")

    compatible_tasks = []
    for task_name in sg_tasks:
        try:
            probe = SAFETY_GYM_ENV_REGISTRY[task_name](
                seed=args.seed, max_steps=args.max_steps,
                consequence_delay=args.delay, delay_mode=args.delay_mode)
            if hasattr(probe, "close"):
                probe.close()
            compatible_tasks.append(task_name)
        except Exception as exc:
            print(f"  Skipping {task_name}: incompatible external dependency ({exc})")
    sg_tasks = compatible_tasks
    if not sg_tasks:
        print("  No compatible Safety Gymnasium tasks available. Skipping E8.")
        return {}

    all_results = {}

    for seed_i, seed in enumerate(_seed_values(args)):
        _sep(f"  Seed {seed_i+1}/{args.seeds}  (seed={seed})")

        task_results = {}

        for task_name in sg_tasks:
            print(f"\n  Task: {task_name}")
            probe_env = SAFETY_GYM_ENV_REGISTRY[task_name](
                seed=seed, max_steps=args.max_steps,
                consequence_delay=args.delay, delay_mode=args.delay_mode)
            sg_state_dim  = probe_env.state_dim
            sg_action_dim = probe_env.action_dim
            cost_budget   = probe_env.constraint_threshold
            shared_env = probe_env

            print(f"    state_dim={sg_state_dim}  action_dim={sg_action_dim}")

            sg_agents = {
                "CCPL":     make_ccpl(sg_state_dim, sg_action_dim, seed=seed,
                                      pretrain_steps=0, constraint_d=cost_budget),
                "CPO-FO":   CPOAgent(sg_state_dim, sg_action_dim, seed=seed,
                                     cost_limit=cost_budget),
                "PPO":      PPOAgent(sg_state_dim, sg_action_dim, seed=seed),
                "SAC-Lag":  SACLagrangianAgent(
                    sg_state_dim, sg_action_dim,
                    cost_limit=cost_budget, seed=seed),
                "CCPL-Base":make_ccpl_base(sg_state_dim, sg_action_dim, seed=seed),
            }

            for name, agent in sg_agents.items():
                print(f"    Training {name} on {task_name}...")
                ep_rewards, ep_costs, ep_csrs = [], [], []

                for ep in range(args.episodes):
                    env = (shared_env if ep == 0 else
                           SAFETY_GYM_ENV_REGISTRY[task_name](
                               seed=seed + ep, max_steps=args.max_steps,
                               consequence_delay=args.delay,
                               delay_mode=args.delay_mode))
                    runner = run_episode if isinstance(agent, CCPLAgent) else run_episode_baseline
                    episode = runner(agent, env, train=True)
                    ep_rewards.append(episode["episode_reward"])
                    ep_costs.append(episode["episode_consequence"])
                    ep_csrs.append(100.0 * (
                        episode["episode_consequence"] <= cost_budget
                    ))
                    if ep != 0 and hasattr(env, "close"):
                        env.close()

                    if (ep + 1) % max(1, args.episodes // 4) == 0:
                        recent_r   = float(np.mean(ep_rewards[-20:]))
                        recent_csr = float(np.mean(ep_csrs[-20:]))
                        print(f"      ep {ep+1:4d}: R={recent_r:+.3f}  CSR={recent_csr:.1f}%")

                eval_rewards, eval_costs, eval_costs_raw, eval_csrs = [], [], [], []
                for eval_ep in range(args.eval_episodes):
                    eval_seed = seed + 1_000_000 + eval_ep
                    env = SAFETY_GYM_ENV_REGISTRY[task_name](
                        seed=eval_seed, max_steps=args.max_steps,
                        consequence_delay=args.delay,
                        delay_mode=args.delay_mode)
                    runner = run_episode if isinstance(agent, CCPLAgent) else run_episode_baseline
                    episode = runner(agent, env, train=False)
                    eval_rewards.append(episode["episode_reward"])
                    eval_costs.append(episode["episode_consequence"])
                    eval_costs_raw.append(episode["episode_consequence_undiscounted"])
                    eval_csrs.append(100.0 * (
                        episode["episode_consequence"] <= cost_budget))
                    if hasattr(env, "close"):
                        env.close()

                task_results.setdefault(task_name, {})[name] = {
                    "mean_reward":      round(float(np.mean(eval_rewards)), 4),
                    "mean_consequence": round(float(np.mean(eval_costs)), 4),
                    "mean_consequence_undiscounted": round(
                        float(np.mean(eval_costs_raw)), 4),
                    "constraint_satisfaction_rate": round(float(np.mean(eval_csrs)), 2),
                    "all_rewards":      [round(float(r), 4) for r in ep_rewards],
                    "all_csrs":         [round(float(c), 2) for c in ep_csrs],
                    "eval_rewards":     [round(float(r), 4) for r in eval_rewards],
                    "eval_costs":       [round(float(c), 4) for c in eval_costs],
                    "eval_costs_undiscounted": [
                        round(float(c), 4) for c in eval_costs_raw],
                    "eval_csrs":        [round(float(c), 2) for c in eval_csrs],
                }

            if hasattr(shared_env, "close"):
                shared_env.close()

            # Preserve completed work if a long MuJoCo benchmark is cancelled.
            _save({"completed_task": task_name, "seed": seed,
                   "seed_results": all_results.get("seed_results", []) + [task_results]},
                  "E8_safety_gym_progress", out)

        all_results.setdefault("seed_results", []).append(task_results)

    averaged = {}
    for task in sg_tasks:
        averaged[task] = {}
        for name in ["CCPL", "CPO-FO", "PPO", "SAC-Lag", "CCPL-Base"]:
            seed_data = [all_results["seed_results"][si].get(task, {}).get(name, {})
                         for si in range(args.seeds)
                         if all_results["seed_results"][si].get(task, {}).get(name)]
            if not seed_data:
                continue
            averaged[task][name] = {
                "mean_reward":      round(float(np.mean([d["mean_reward"] for d in seed_data])), 4),
                "mean_consequence": round(float(np.mean([d["mean_consequence"] for d in seed_data])), 4),
                "mean_consequence_undiscounted": round(float(np.mean([
                    d["mean_consequence_undiscounted"] for d in seed_data])), 4),
                "constraint_satisfaction_rate": round(float(np.mean([d["constraint_satisfaction_rate"]
                                                                      for d in seed_data])), 2),
            }

    _sep("  E8 Results Summary")
    for task, res in averaged.items():
        print(f"\n  Task: {task}")
        print(f"  {'Agent':<14}  {'Reward':>8}  {'J_c':>8}  {'CSR%':>7}")
        print("  " + "-" * 42)
        for name, r in res.items():
            print(f"  {name:<14}  {r['mean_reward']:>+8.3f}  "
                  f"{r['mean_consequence']:>8.4f}  {r['constraint_satisfaction_rate']:>6.1f}%")

    _save({"averaged": averaged, "seed_results": all_results.get("seed_results", [])},
          "E8_safety_gym", out)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plots import COLORS, _smooth

    agent_names = ["CCPL", "CPO-FO", "PPO", "SAC-Lag", "CCPL-Base"]
    n_tasks = len(sg_tasks)

    fig, axes = plt.subplots(1, n_tasks, figsize=(6 * n_tasks, 5), sharey=False)
    if n_tasks == 1: axes = [axes]
    for ax, task in zip(axes, sg_tasks):
        names   = [n for n in agent_names if n in averaged.get(task, {})]
        rewards = [averaged[task][n]["mean_reward"] for n in names]
        bars = ax.bar(names, rewards, color=[COLORS.get(n, "#888") for n in names], alpha=0.87, width=0.55)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        ax.set_title(f"{task}", fontweight="bold", fontsize=10)
        ax.set_ylabel("Mean Reward"); ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("E8 — Safety Gymnasium: Reward per Task", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{out}/E8_reward_per_task.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out}/E8_reward_per_task.png")

    fig, axes = plt.subplots(1, n_tasks, figsize=(6 * n_tasks, 5), sharey=True)
    if n_tasks == 1: axes = [axes]
    for ax, task in zip(axes, sg_tasks):
        names = [n for n in agent_names if n in averaged.get(task, {})]
        csrs  = [averaged[task][n]["constraint_satisfaction_rate"] for n in names]
        bars = ax.bar(names, csrs, color=[COLORS.get(n, "#888") for n in names], alpha=0.87, width=0.55)
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
        ax.set_title(f"{task}", fontweight="bold", fontsize=10)
        ax.set_ylabel("CSR %"); ax.set_ylim(0, 115); ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("E8 — Safety Gymnasium: Constraint Satisfaction Rate", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{out}/E8_csr_per_task.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out}/E8_csr_per_task.png")

    fig, ax = plt.subplots(figsize=(9, 6))
    for name in agent_names:
        for task in sg_tasks:
            if name not in averaged.get(task, {}): continue
            r  = averaged[task][name]["mean_reward"]
            jc = averaged[task][name]["mean_consequence"]
            c  = COLORS.get(name, "#888")
            ax.scatter(jc, r, color=c, s=100, zorder=5)
            ax.annotate(f"{name}\n({task[:8]})", (jc, r),
                        textcoords="offset points", xytext=(6, 4), fontsize=7, color=c)
    ax.set_xlabel("Mean Consequence J_c"); ax.set_ylabel("Mean Reward")
    ax.set_title("E8 — Safety Gymnasium: Reward vs J_c", fontweight="bold")
    ax.grid(alpha=0.25, linestyle="--"); fig.tight_layout()
    fig.savefig(f"{out}/E8_reward_vs_jc.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out}/E8_reward_vs_jc.png")

    last_seed = all_results.get("seed_results", [{}])[-1]
    for task in sg_tasks:
        task_data = last_seed.get(task, {})
        if not task_data: continue
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for name, d in task_data.items():
            c = COLORS.get(name, "#888")
            if d.get("all_rewards"): axes[0].plot(_smooth(d["all_rewards"]), label=name, color=c, lw=1.8, alpha=0.9)
            if d.get("all_csrs"):    axes[1].plot(_smooth(d["all_csrs"]),    label=name, color=c, lw=1.8, alpha=0.9)
        for ax, ylabel, subtitle in [(axes[0], "Reward", "Reward over Training"),
                                     (axes[1], "CSR %",  "CSR over Training")]:
            ax.set_xlabel("Episode"); ax.set_ylabel(ylabel)
            ax.set_title(f"{task} — {subtitle}", fontweight="bold")
            ax.legend(fontsize=9); ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(f"{out}/E8_curves_{task.replace(' ','_')}.png", dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  Saved: {out}/E8_curves_{task.replace(' ','_')}.png")

    return averaged



def run_E9(args):
    """
    E9: Ablation on Safety Gymnasium tasks.
    Tests which CCPL components matter on several registered Safety Gymnasium
    benchmarks using held-out environment seeds.
    """
    _sep("E9 — Safety Gymnasium Ablation (D1/D2/D3 isolation)")

    if not _SAFETY_GYM_AVAILABLE:
        print("  safety-gymnasium not installed. Run: pip install safety-gymnasium")
        print("  Skipping E9.")
        return {}

    out = os.path.join(args.out, "E9")
    os.makedirs(out, exist_ok=True)

    e9_tasks = ["SafetyPointGoal1", "SafetyPointGoal2", "SafetyPointButton1"]
    e9_tasks = [t for t in e9_tasks if t in SAFETY_GYM_ENV_REGISTRY]
    if not e9_tasks:
        print("  No E9 tasks available. Skipping E9.")
        return {}
    print(f"  Ablation tasks: {e9_tasks}")

    all_results = {task: {} for task in e9_tasks}

    for task_name in e9_tasks:
        _sep(f"  E9 Task: {task_name}")
        probe_env     = SAFETY_GYM_ENV_REGISTRY[task_name](
            seed=args.seed, max_steps=args.max_steps)
        sg_state_dim  = probe_env.state_dim
        sg_action_dim = probe_env.action_dim
        cost_budget   = probe_env.constraint_threshold
        if hasattr(probe_env, "close"):
            probe_env.close()
        print(f"  state_dim={sg_state_dim}  action_dim={sg_action_dim}")

        for seed_i, seed in enumerate(_seed_values(args)):
            print(f"\n  Seed {seed_i+1}/{args.seeds}")

            ablation_agents = build_ccpl_ablation(sg_state_dim, sg_action_dim, seed=seed)

            for name, agent in ablation_agents.items():
                print(f"    Training {name}...")
                ep_rewards = []

                for ep in range(args.episodes):
                    env = SAFETY_GYM_ENV_REGISTRY[task_name](
                        seed=seed + ep, max_steps=args.max_steps)
                    episode = run_episode(agent, env, train=True)
                    ep_rewards.append(episode["episode_reward"])
                    if hasattr(env, "close"):
                        env.close()

                eval_rewards, eval_costs, eval_csrs = [], [], []
                for eval_ep in range(args.eval_episodes):
                    env = SAFETY_GYM_ENV_REGISTRY[task_name](
                        seed=seed + 1_000_000 + eval_ep,
                        max_steps=args.max_steps)
                    episode = run_episode(agent, env, train=False)
                    eval_rewards.append(episode["episode_reward"])
                    eval_costs.append(episode["episode_consequence"])
                    eval_csrs.append(100.0 * (
                        episode["episode_consequence"] <= cost_budget))
                    if hasattr(env, "close"):
                        env.close()

                all_results[task_name].setdefault(name, []).append({
                    "seed":        seed,
                    "mean_reward": round(float(np.mean(eval_rewards)), 4),
                    "mean_cost":   round(float(np.mean(eval_costs)), 4),
                    "csr":         round(float(np.mean(eval_csrs)), 2),
                    "eval_rewards": [round(float(x), 4) for x in eval_rewards],
                    "eval_costs":   [round(float(x), 4) for x in eval_costs],
                    "training_rewards": [round(float(x), 4) for x in ep_rewards],
                })

    averaged = {}
    for task_name in e9_tasks:
        averaged[task_name] = {
            n: {
                "mean_reward": round(float(np.mean([r["mean_reward"] for r in v])), 4),
                "mean_cost":   round(float(np.mean([r["mean_cost"] for r in v])), 4),
                "csr":         round(float(np.mean([r["csr"]         for r in v])), 2),
            }
            for n, v in all_results[task_name].items()
        }

    for task_name, res in averaged.items():
        _sep(f"  E9 Ablation Summary — {task_name}")
        print(f"  {'Variant':<20}  {'Reward':>8}  {'CSR%':>7}")
        print("  " + "-" * 40)
        base_r = res.get("CCPL-Base", {}).get("mean_reward", 0)
        for name, r in res.items():
            dr = r["mean_reward"] - base_r
            print(f"  {name:<20}  {r['mean_reward']:>+8.3f} ({dr:>+.3f})  {r['csr']:>6.1f}%")

    _save({"tasks": e9_tasks, "results": averaged,
           "seed_results": all_results}, "E9_safety_gym_ablation", out)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plots import COLORS

    n_tasks = len(e9_tasks)

    fig, axes = plt.subplots(1, n_tasks, figsize=(7 * n_tasks, 5), sharey=False)
    if n_tasks == 1: axes = [axes]
    for ax, task_name in zip(axes, e9_tasks):
        res    = averaged[task_name]
        names  = list(res.keys())
        colors = [COLORS.get(n, "#888") for n in names]
        bars   = ax.bar(names, [res[n]["mean_reward"] for n in names],
                        color=colors, alpha=0.87, width=0.55)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.set_title(f"{task_name}\nReward", fontweight="bold", fontsize=9)
        ax.set_ylabel("Mean Reward"); ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("E9 — Safety Gym Ablation: Reward per Variant per Task",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{out}/E9_ablation_reward_per_task.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"  Saved: {out}/E9_ablation_reward_per_task.png")

    fig, axes = plt.subplots(1, n_tasks, figsize=(7 * n_tasks, 5), sharey=True)
    if n_tasks == 1: axes = [axes]
    for ax, task_name in zip(axes, e9_tasks):
        res    = averaged[task_name]
        names  = list(res.keys())
        colors = [COLORS.get(n, "#888") for n in names]
        bars   = ax.bar(names, [res[n]["csr"] for n in names],
                        color=colors, alpha=0.87, width=0.55)
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
        ax.set_title(f"{task_name}\nCSR%", fontweight="bold", fontsize=9)
        ax.set_ylabel("CSR %"); ax.set_ylim(0, 115); ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("E9 — Safety Gym Ablation: CSR per Variant per Task",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{out}/E9_ablation_csr_per_task.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"  Saved: {out}/E9_ablation_csr_per_task.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    variant_names = list(next(iter(averaged.values())).keys())
    x = np.arange(len(variant_names))
    task_colors = ["#2563EB", "#DC2626", "#16A34A"]
    width = 0.25
    for i, (task_name, tc) in enumerate(zip(e9_tasks, task_colors)):
        res    = averaged[task_name]
        base_r = res.get("CCPL-Base", {}).get("mean_reward", 0)
        deltas = [res.get(n, {}).get("mean_reward", 0) - base_r for n in variant_names]
        bars   = ax.bar(x + i * width, deltas, width, label=task_name,
                        color=tc, alpha=0.85)
        ax.bar_label(bars, fmt="%+.3f", padding=2, fontsize=7)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x + width)
    ax.set_xticklabels(variant_names, rotation=20, fontsize=9)
    ax.set_ylabel("ΔReward vs CCPL-Base")
    ax.set_title("E9 — Direction Contribution (Δ vs CCPL-Base) across Tasks",
                 fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{out}/E9_direction_contribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"  Saved: {out}/E9_direction_contribution.png")

    return averaged


class _SCMQualityLabels:
    def __init__(self, base, mode, seed):
        self.base = base
        self.mode = mode
        self.rng = np.random.default_rng(seed)

    def generate_batch(self, states, actions, n_actions=5):
        labels = self.base.generate_batch(states, actions, n_actions)
        if self.mode == "noisy_scm":
            labels["delta_C_scm"] = labels["delta_C_scm"] + self.rng.normal(
                0.0, 0.15, len(states)).astype(np.float32)
        elif self.mode == "misspecified_scm":
            labels = dict(labels)
            wrong_action = (np.asarray(actions) + 1) % n_actions
            labels["delta_C_scm"] = labels["delta_C_all"][
                np.arange(len(states)), wrong_action]
        return labels

    def icn_calibration_error(self, states, actions, icn_delta_C):
        """Expose the agent label-generator calibration contract for E10 modes."""
        labels = self.generate_batch(states, actions)
        scm_delta = labels["delta_C_scm"]
        mae = float(np.abs(np.asarray(icn_delta_C) - scm_delta).mean())
        if (len(icn_delta_C) > 1 and np.std(icn_delta_C) > 1e-12
                and np.std(scm_delta) > 1e-12):
            correlation = float(np.corrcoef(icn_delta_C, scm_delta)[0, 1])
        else:
            correlation = 0.0
        return {
            "mae": round(mae, 5),
            "correlation": round(correlation, 4),
            "well_calibrated": mae < 0.05 and correlation > 0.7,
        }


def _multi_action_diagnostic(agent, seed, n_sequences=128, horizon=3):
    rng = np.random.default_rng(seed)
    states = rng.uniform(0.1, 0.9, (n_sequences, horizon, STATE_DIM)).astype(np.float32)
    actions = rng.integers(0, ACTION_DIM, (n_sequences, horizon), dtype=np.int32)
    flat_states = states.reshape(-1, STATE_DIM)
    flat_actions = actions.reshape(-1)
    contexts = np.zeros((len(flat_states), agent.icn.causal_dim), np.float32)
    predicted, _, _, _ = agent.icn.forward(flat_states, flat_actions, contexts)
    labels = CausalLabelGenerator(EnvironmentSCM()).generate_batch(
        flat_states, flat_actions, ACTION_DIM)
    target = labels["delta_C_scm"]
    return {
        "horizon": horizon,
        "sequences": n_sequences,
        "mae": float(np.mean(np.abs(predicted - target))),
        "sign_agreement": float(np.mean(np.sign(predicted) == np.sign(target))),
        "scope": "sequential per-action SCM contrasts; not joint Shapley attribution",
    }


def run_E10(args):
    _sep("E10 - SCM Quality and Attribution Robustness")
    out = os.path.join(args.out, "E10")
    os.makedirs(out, exist_ok=True)
    modes = ("oracle_scm", "noisy_scm", "misspecified_scm", "observational_only")
    all_results = []
    for seed_i, seed in enumerate(_seed_values(args)):
        seed_results = {"seed": seed, "modes": {}}
        for mode in modes:
            agent = make_ccpl(STATE_DIM, ACTION_DIM, seed=seed,
                              pretrain_steps=0 if args.quick or args.episodes < 10 else 200)
            if mode != "oracle_scm":
                agent.label_gen = _SCMQualityLabels(agent.label_gen, mode, seed + 17)
            if mode == "observational_only":
                agent.has_scm_labels = False
            train_agent(agent, n_episodes=args.episodes, max_steps=args.max_steps,
                        delay_steps=args.delay, seed=seed, verbose=args.verbose,
                        log_freq=max(1, args.episodes // 4), env_names=list(TRAIN_ENVS))
            metrics = evaluate_all({mode: agent}, list(EVAL_ENVS), args.eval_episodes,
                                   args.max_steps, args.delay)[mode]
            seed_results["modes"][mode] = {
                "environment_metrics": metrics,
                "multi_action": _multi_action_diagnostic(agent, seed + 31),
            }
        all_results.append(seed_results)
    summary = {}
    for mode in modes:
        rows = [metrics
                for item in all_results
                for metrics in item["modes"][mode]["environment_metrics"].values()]
        summary[mode] = {
            "mean_reward": float(np.mean([row["mean_reward"] for row in rows])),
            "mean_consequence": float(np.mean([row["mean_consequence"] for row in rows])),
            "csr": float(np.mean([row["constraint_satisfaction_rate"] for row in rows])),
            "seeds": len(all_results),
            "environment_metrics": len(rows),
        }
    _save({"protocol": {
        "label_source": "SCM-generated for oracle/noisy/misspecified modes; none for observational_only",
        "modes": list(modes), "seed_values": [item["seed"] for item in all_results],
        "train_environments": list(TRAIN_ENVS), "eval_environments": list(EVAL_ENVS),
        "multi_action": "sequential per-action diagnostic; not a joint credit-assignment theorem",
    }, "summary": summary, "seed_results": all_results}, "E10_scm_quality", out)
    print(json.dumps(summary, indent=2))
    return summary



def main(argv=None):
    p = argparse.ArgumentParser(description="CCPL Extended Experiment Suite")
    p.add_argument("--exp",  default="E1",
                   choices=["E1","E2","E3","E4","E5","E6","E7","E8","E9","E10","theory","all"])
    p.add_argument("--all",  action="store_true", help="Run all experiments")
    p.add_argument("--episodes",      type=int, default=500)
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--max-steps",     type=int, default=100)
    p.add_argument("--delay",         type=int, default=5)
    p.add_argument("--delay-mode",    choices=["immediate", "fixed", "stochastic", "distribution"],
                   default="immediate")
    p.add_argument("--seeds",         type=int, default=5)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--seed-values",   type=str, default="",
                   help="Explicit comma-separated seeds; must match --seeds")
    p.add_argument("--out",           type=str, default="results/experiments")
    p.add_argument("--verbose",       action="store_true")
    p.add_argument("--quick",         action="store_true",
                   help="200 episodes, 1 seed, 20 eval episodes")
    p.add_argument("--tasks",         type=str, default="",
                   help="Comma-separated E8 task names; empty means all")
    args = p.parse_args(argv)

    if args.quick:
        args.episodes, args.eval_episodes, args.seeds = 200, 20, 1
    if min(args.episodes, args.eval_episodes, args.max_steps, args.seeds) < 1:
        p.error("episodes, eval-episodes, max-steps, and seeds must be positive")
    if args.delay < 0:
        p.error("delay must be non-negative")

    runners = {
        "E1": run_E1, "E2": run_E2, "E3": run_E3,
        "E4": run_E4, "E5": run_E5, "E6": run_E6,
        "E7": run_E7, "E8": run_E8, "E9": run_E9, "E10": run_E10,
        "theory": run_theory_verification,
    }

    if args.all or args.exp == "all":
        failures = []
        for exp_name, fn in runners.items():
            _sep(f"Running {exp_name}")
            try:
                fn(args)
            except Exception as e:
                print(f"  ERROR in {exp_name}: {e}")
                import traceback; traceback.print_exc()
                failures.append((exp_name, e))
        if failures:
            names = ", ".join(name for name, _ in failures)
            raise RuntimeError(f"Experiment suite failed in: {names}")
    else:
        runners[args.exp](args)


if __name__ == "__main__":
    main()
