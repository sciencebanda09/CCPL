"""
run_benchmark_v7.py — CCPL V7 Full Benchmark
==============================================
Runs the repository benchmark suite.  Results must be regenerated after the
audit before comparing them with the submitted PDF:
  E1: Main comparison (Table 1)
  E2: Ablation study  (Table 2)
  E3: ICN calibration (Table 4 T3)
  E4: Delay calibration (Table 4 T1)
  E5: Numerical diagnostics (historically labelled theory verification)
  E6: Adversarial robustness (Table 3)
  E7: Safety Gymnasium (new — CCPL vs all baselines)

Usage:
  python run_benchmark_v7.py --episodes 1000 --seeds 3 --verbose
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "ccpl"))
sys.path.insert(0, str(ROOT / "ccpl" / "algorithms"))
sys.path.insert(0, str(ROOT / "ccpl" / "environments"))

sys.path.insert(0, os.path.dirname(__file__))

from environments import make_env, ENV_REGISTRY
from ccpl_agent import make_ccpl, make_ccpl_base, build_ccpl_ablation, run_episode
from dqn_agent import DQNAgent
from a2c_agent import A2CAgent
from ppo_agent import PPOAgent
from constrained_baselines import CPOAgent, RCPOAgent, PIDLagrangianAgent, SACLagrangianAgent
from train import train_agent, run_episode_baseline, count_parameters, save_eval_results
from evaluate import evaluate_agent, evaluate_all, compute_transfer_score, print_benchmark_table
from adversarial_envs import ADVERSARIAL_ENV_REGISTRY
from safety_gym_adapter import SafetyPointGoal1Env, SafetyCarGoal1Env, SAFETY_GYM_REGISTRY, make_safety_env

STATE_DIM  = 6
ACTION_DIM = 5

TRAIN_ENVS = ("standard", "noisy", "shifted", "randomised")
EVAL_ENVS  = ("standard", "noisy", "shifted")
ADV_ENVS   = ("deception_bench", "hidden_state_shift", "conflict_zone")


# ── Agent factories ───────────────────────────────────────────────────────────

def _make_all_agents(seed: int) -> dict:
    sh = dict(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    return {
        "CCPL":     make_ccpl(**sh, seed=seed),
        "PPO":      PPOAgent(**sh, seed=seed),
        "A2C":      A2CAgent(**sh, seed=seed),
        "DQN":      DQNAgent(**sh, seed=seed, double=True),
        "CPO-FO":   CPOAgent(**sh, seed=seed),
        "RCPO":     RCPOAgent(**sh, seed=seed),
        "PID-Lag":  PIDLagrangianAgent(**sh, seed=seed),
        "SAC-Lag":  SACLagrangianAgent(**sh, seed=seed),
        "CCPL-Base": make_ccpl_base(**sh, seed=seed),
    }


# ── Training helpers ──────────────────────────────────────────────────────────

def _train(agent, n_eps, seed, envs=TRAIN_ENVS, verbose=False,
           max_steps=100, delay=5):
    return train_agent(agent, n_episodes=n_eps, max_steps=max_steps,
                       delay_steps=delay, seed=seed, verbose=verbose,
                       env_names=envs, save_every=9999)


def _eval_all(agents, envs, n_eval, seed, max_steps=100, delay=5):
    """Evaluate agents on given envs, returns {agent: {env: metrics}}."""
    results = {}
    for name, agent in agents.items():
        results[name] = {}
        for env in envs:
            if env not in ENV_REGISTRY and env not in ADVERSARIAL_ENV_REGISTRY:
                raise KeyError(f"Unknown evaluation environment: {env}")
            try:
                results[name][env] = evaluate_agent(
                    agent, env, n_eval, max_steps, delay, seed + 10000)
            except Exception as exc:
                raise RuntimeError(
                    f"Evaluation failed for agent={name!r}, env={env!r}"
                ) from exc
    return results


def _agg(results, envs):
    """Aggregate results across environments."""
    agg = {}
    for name, res in results.items():
        rs = [res[e]["mean_reward"]                           for e in envs if e in res]
        cs = [res[e]["mean_consequence"]                      for e in envs if e in res]
        cr = [res[e].get("constraint_satisfaction_rate", 0.0) for e in envs if e in res]
        agg[name] = {
            "reward": float(np.mean(rs)) if rs else 0.0,
            "Jc":     float(np.mean(cs)) if cs else 0.0,
            "CSR":    float(np.mean(cr)) if cr else 0.0,
        }
    return agg


# ── Theory verification ───────────────────────────────────────────────────────

def verify_theory(agent) -> dict:
    """Run numerical and synthetic-SCM calibration diagnostics."""
    rng = np.random.default_rng(99)
    h_samp = rng.normal(size=(64, agent.gru_dim)).astype(np.float32)
    t1     = agent.bellman.verify_contraction(h_samp)
    t2     = agent.lam_tracker.theorem2_status(agent.last_mean_lambda)

    # T3: ICN calibration
    from causal_graph import EnvironmentSCM, CausalLabelGenerator
    scm  = EnvironmentSCM(noise_std=0.0)
    lgen = CausalLabelGenerator(scm)
    sts  = rng.uniform(0.1, 0.9, (300, agent.state_dim)).astype(np.float32)
    sts_norm = np.asarray([agent.normalizer.normalize(s) for s in sts], np.float32)
    icn_states = sts if getattr(agent, "has_scm_labels", False) else sts_norm
    afull = np.full(300, 2, dtype=np.int32)
    ctx   = np.zeros((300, agent.icn.causal_dim), np.float32)
    dc, _, _, _ = agent.icn.forward(icn_states, afull, ctx)
    labs = lgen.generate_batch(sts, afull, agent.action_dim)
    scm_d = labs["delta_C_scm"].astype(np.float32)
    corr  = float(np.corrcoef(dc, scm_d)[0, 1]) if dc.std() > 1e-6 else 0.0
    mae   = float(np.mean(np.abs(dc - scm_d)))

    # Lambda boundedness/trend (the latter is not a convergence proof).
    lam_hist = np.array(agent._lambda_log[-500:]) if agent._lambda_log else np.array([0.5])
    lam_in_range = bool(0.0 <= lam_hist.mean() <= agent.lambda_max)
    lam_std_dec  = float(lam_hist[-100:].std()) < float(lam_hist[:100].std()) + 0.1 \
                   if len(lam_hist) >= 200 else True

    return {
        "T1_contraction":    t1["contraction_satisfied"],
        "T1_gamma_eff_max":  round(float(t1.get("gamma_eff_max", 0.0)), 4),
        "T1_delay_support":  t1.get("delay_bounds_satisfied", False),
        "T2_var_s":          round(float(t2.get("var_s", 0)), 5),
        "T2_epsilon":        None,
        "T2_state_variation": round(float(t2.get("state_variation_score", 0)), 4),
        "T2_diagnostic_only": True,
        "T3_icn_corr":       round(corr, 3),
        "T3_icn_mae":        round(mae, 3),
        "T4_lam_in_range":   lam_in_range,
        "T4_lam_std_dec":    lam_std_dec,
    }


# ── Safety Gymnasium benchmark ────────────────────────────────────────────────

def run_safety_gym_benchmark(n_eps, seed, verbose, max_steps=200):
    """CCPL vs baselines on repository-local synthetic safety analogues."""
    print("\n" + "="*70)
    print("  E7: Synthetic Safety-Environment Benchmark")
    print("="*70)

    sg_results = {}
    for env_name, env_cls in SAFETY_GYM_REGISTRY.items():
        sample = env_cls(seed=0)
        S, A   = sample.state_dim, sample.action_dim
        D      = sample.constraint_threshold
        agents = {
            "CCPL":      make_ccpl(S, A, seed=seed, constraint_d=D,
                                    lambda_warmup=100, penalty_scale=2.0,
                                    buffer_capacity=80_000, eps_decay=3000),
            "CCPL-Base": make_ccpl_base(S, A, seed=seed),
            "DQN":       DQNAgent(state_dim=S, action_dim=A, seed=seed),
            "PPO":       PPOAgent(state_dim=S, action_dim=A, seed=seed),
            "CPO-FO":    CPOAgent(state_dim=S, action_dim=A, cost_limit=D, seed=seed),
            "SAC-Lag":   SACLagrangianAgent(
                state_dim=S, action_dim=A, cost_limit=D, seed=seed),
        }
        sg_results[env_name] = {}
        for name, agent in agents.items():
            if verbose:
                print(f"  Training {name} on {env_name} ({n_eps} eps)...")
            rews, costs, csrs = [], [], []
            for ep in range(n_eps):
                env  = env_cls(max_steps=max_steps, seed=seed * 10000 + ep)
                is_c = hasattr(agent, "reset_hidden")
                fn   = run_episode if is_c else run_episode_baseline
                r    = fn(agent, env, train=True, update_freq=4)
                rews.append(r["episode_reward"])
                costs.append(r["episode_consequence"])
                csrs.append(1.0 if r["episode_consequence"] <= D else 0.0)

            eval_rews, eval_costs, eval_csrs = [], [], []
            n_eval = max(10, min(50, n_eps // 5))
            for eval_ep in range(n_eval):
                env = env_cls(
                    max_steps=max_steps,
                    seed=seed * 10000 + 1_000_000 + eval_ep)
                is_c = hasattr(agent, "reset_hidden")
                fn = run_episode if is_c else run_episode_baseline
                episode = fn(agent, env, train=False, update_freq=4)
                eval_rews.append(episode["episode_reward"])
                eval_costs.append(episode["episode_consequence"])
                eval_csrs.append(episode["episode_consequence"] <= D)
            sg_results[env_name][name] = {
                "mean_reward":  float(np.mean(eval_rews)),
                "mean_cost":    float(np.mean(eval_costs)),
                "csr":          float(np.mean(eval_csrs)) * 100,
                "std_reward":   float(np.std(eval_rews)),
                "training_rewards": rews,
                "training_costs": costs,
                "training_csrs": csrs,
            }
            if verbose:
                r = sg_results[env_name][name]
                print(f"    {name:<14} R={r['mean_reward']:+.3f} "
                      f"Jc={r['mean_cost']:.3f} CSR={r['csr']:.1f}%")

    # Print table
    print("\n" + "="*70)
    print("  SYNTHETIC SAFETY RESULTS (held-out episodes; one trained seed)")
    print("="*70)
    for env_name, er in sg_results.items():
        print(f"\n  {env_name}:")
        print(f"  {'Agent':<16} {'Reward':>9} {'J_c':>8} {'CSR%':>7}")
        print("  " + "-"*44)
        for name in ["CCPL", "CCPL-Base", "DQN", "PPO", "CPO-FO", "SAC-Lag"]:
            if name in er:
                r = er[name]
                print(f"  {name:<16} {r['mean_reward']:>+9.3f} "
                      f"{r['mean_cost']:>8.3f} {r['csr']:>6.1f}%")
    print("="*70)
    return sg_results


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(description="CCPL V7 Benchmark")
    parser.add_argument("--episodes",  type=int,  default=600,  help="Training episodes")
    parser.add_argument("--eval-eps",  type=int,  default=50,   help="Eval episodes per env")
    parser.add_argument("--seeds",     type=int,  default=1,    help="Number of seeds")
    parser.add_argument("--seed",      type=int,  default=42,   help="Base random seed")
    parser.add_argument("--max-steps", type=int,  default=100,  help="Episode horizon")
    parser.add_argument("--delay",     type=int,  default=5,    help="Synthetic feedback delay")
    parser.add_argument("--verbose",   action="store_true",     help="Verbose output")
    parser.add_argument("--ablation",  action="store_true",     help="Run ablation study")
    parser.add_argument("--safety",    action="store_true",     help="Run Safety-Gym benchmark")
    parser.add_argument("--out",       type=str,  default="results_v7", help="Output directory")
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    t0  = time.time()
    all_results = {}

    # ── E1: Main comparison ────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  E1: Main Comparison (Table 1 equivalent)")
    print("="*70)

    all_seed_results = {env: {} for env in EVAL_ENVS}
    theory_checks    = {}
    trained_agents   = None

    for seed_i in range(args.seeds):
        s = args.seed + seed_i * 100
        print(f"\n  --- Seed {seed_i+1}/{args.seeds} (seed={s}) ---")
        agents = _make_all_agents(s)

        for name, agent in agents.items():
            print(f"  Training {name}...")
            _train(agent, args.episodes, s, verbose=args.verbose,
                   max_steps=args.max_steps, delay=args.delay)

        # Evaluate
        res = _eval_all(
            agents, EVAL_ENVS, args.eval_eps, s,
            max_steps=args.max_steps, delay=args.delay)
        for env in EVAL_ENVS:
            for name in res:
                if name not in all_seed_results[env]:
                    all_seed_results[env][name] = []
                all_seed_results[env][name].append(res[name].get(env, {}))

        # Theory checks on CCPL
        if "CCPL" in agents:
            theory_checks[f"seed_{s}"] = verify_theory(agents["CCPL"])
        trained_agents = agents

    # Aggregate across seeds
    agg_results = {}
    agent_names = list(trained_agents.keys()) if trained_agents else []
    for name in agent_names:
        rews, jcs, csrs = [], [], []
        for env in EVAL_ENVS:
            for ep_r in all_seed_results[env].get(name, []):
                rews.append(ep_r.get("mean_reward", 0))
                jcs.append(ep_r.get("mean_consequence", 0))
                csrs.append(ep_r.get("constraint_satisfaction_rate", 0))
        agg_results[name] = {
            "reward": round(float(np.mean(rews)), 3) if rews else 0.0,
            "Jc":     round(float(np.mean(jcs)), 4)  if jcs  else 0.0,
            "CSR":    round(float(np.mean(csrs)), 1)  if csrs else 0.0,
        }

    # Print Table 1
    print("\n" + "="*70)
    print("  TABLE 1: Main benchmark results")
    print(f"  (mean over {args.seeds} seeds x {len(EVAL_ENVS)} environments)")
    print("="*70)
    print(f"  {'Algorithm':<18} {'Reward':>9} {'Jc':>8} {'CSR%':>7}")
    print("  " + "-"*45)
    for name in ["CCPL", "PPO", "A2C", "DQN", "CPO-FO", "RCPO", "PID-Lag", "SAC-Lag", "CCPL-Base"]:
        if name in agg_results:
            r = agg_results[name]
            star = " *" if name == "CCPL" else ""
            print(f"  {name:<18} {r['reward']:>+9.3f} {r['Jc']:>8.4f} {r['CSR']:>6.1f}%{star}")
    print("="*70)
    print("  Note: submitted-paper values are not acceptance targets for this run.")

    all_results["E1_main"] = agg_results
    all_results["E1_seed_results"] = all_seed_results
    all_results["E1_seeds"] = [args.seed + i * 100 for i in range(args.seeds)]
    save_eval_results(all_results, "E1_main", args.out)

    # ── E5: numerical diagnostics ──────────────────────────────────────────
    print("\n" + "="*70)
    print("  E5/T: Numerical and Synthetic-SCM Diagnostics")
    print("="*70)
    if theory_checks:
        tc = list(theory_checks.values())[0]
        print(f"  D1 Bellman modulus: gamma < 1.0 -> "
              f"{'PASS' if tc['T1_contraction'] else 'FAIL'} "
              f"(gamma_eff={tc['T1_gamma_eff_max']})")
        print(f"  D2 State variation: diagnostic only "
              f"(Var_s={tc['T2_var_s']}, "
              f"CV={tc['T2_state_variation']}; no epsilon bound)")
        print(f"  D3 SCM calibration: MAE<0.10 diagnostic -> "
              f"{'PASS' if tc['T3_icn_mae'] < 0.10 else 'WARN'} "
              f"(MAE={tc['T3_icn_mae']}, Corr={tc['T3_icn_corr']})")
        print(f"  D4 Lambda bounds:  in_range -> "
              f"{'PASS' if tc['T4_lam_in_range'] else 'FAIL'}")
        all_results["E5_theory"] = theory_checks

    # ── E2: Ablation study ─────────────────────────────────────────────────
    if args.ablation:
        print("\n" + "="*70)
        print("  E2: Ablation Study (Table 2)")
        print("="*70)
        abl_variants = build_ccpl_ablation(STATE_DIM, ACTION_DIM, args.seed)
        for name, agent in abl_variants.items():
            print(f"  Training {name}...")
            _train(agent, args.episodes, args.seed, verbose=args.verbose,
                   max_steps=args.max_steps, delay=args.delay)
        abl_eval = _eval_all(
            abl_variants, EVAL_ENVS, args.eval_eps, args.seed,
            max_steps=args.max_steps, delay=args.delay)
        abl_agg  = _agg(abl_eval, EVAL_ENVS)

        print(f"\n  {'Variant':<18} {'Reward':>9} {'Jc':>8} {'CSR%':>7} {'Missing'}")
        print("  " + "-"*65)
        MISSING = {
            "CCPL":          "- (full system)",
            "CCPL-NoDelay":  "D1 removed",
            "CCPL-NoStateλ": "D2 removed",
            "CCPL-NoCausal": "D3 removed",
            "CCPL-SingleQ":  "D4 removed",
            "CCPL-Base":     "D1–D4 removed",
        }
        for name in ["CCPL", "CCPL-NoDelay", "CCPL-NoStateλ",
                      "CCPL-NoCausal", "CCPL-SingleQ", "CCPL-Base"]:
            if name in abl_agg:
                r = abl_agg[name]
                print(f"  {name:<18} {r['reward']:>+9.3f} {r['Jc']:>8.4f} "
                      f"{r['CSR']:>6.1f}%  {MISSING.get(name,'')}")
        print("="*70)
        all_results["E2_ablation"] = abl_agg
        save_eval_results(all_results, "E2_ablation", args.out)

    # ── E7: Safety Gymnasium ───────────────────────────────────────────────
    if args.safety:
        sg_r = run_safety_gym_benchmark(
            n_eps=min(args.episodes, 300), seed=args.seed,
            verbose=args.verbose, max_steps=args.max_steps)
        all_results["E7_safety_gym"] = sg_r
        save_eval_results(all_results, "E7_safety_gym", args.out)

    # ── Save final results ─────────────────────────────────────────────────
    total_time = time.time() - t0
    all_results["meta"] = {
        "total_time_s": round(total_time, 1),
        "episodes": args.episodes,
        "seeds": args.seeds,
        "max_steps": args.max_steps,
        "delay": args.delay,
    }
    out_path = os.path.join(args.out, "all_results.json")
    with open(out_path, "w") as f:
        def clean(v):
            if isinstance(v, (np.integer,)):  return int(v)
            if isinstance(v, (np.floating,)): return float(v)
            if isinstance(v, np.ndarray):     return v.tolist()
            if isinstance(v, dict):           return {k: clean(vv) for k, vv in v.items()}
            if isinstance(v, list):           return [clean(x) for x in v]
            return v
        json.dump(clean(all_results), f, indent=2)
    print(f"\n  Results saved to {out_path}")
    print(f"  Total time: {total_time/60:.1f} min")


if __name__ == "__main__":
    main()
