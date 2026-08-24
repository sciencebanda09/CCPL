"""
run_ccpl.py — CCPL entry point
================================
  python run_ccpl.py demo              # CCPL vs CCPL-Base vs PPO
  python run_ccpl.py train             # full training + numerical diagnostics
  python run_ccpl.py ablation          # 4-variant ablation (D1/D2/D3 isolation)
  python run_ccpl.py benchmark         # full baseline comparison
  python run_ccpl.py theory            # standalone numerical diagnostics
  python run_ccpl.py causal            # SCM causal attribution analysis
  --quick                              # 200 episodes fast smoke test
"""
import argparse, os
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "ccpl"))
sys.path.insert(0, str(ROOT / "ccpl" / "algorithms"))
sys.path.insert(0, str(ROOT / "ccpl" / "environments"))
import adversarial_envs  # noqa: F401  (registers additional environments)

from ccpl_agent   import make_ccpl, run_episode, build_ccpl_ablation, make_ccpl_base
from ppo_agent    import PPOAgent
from a2c_agent    import A2CAgent
from dqn_agent    import DQNAgent
from train        import train_agent, count_parameters, save_eval_results, _fmt_eta
from evaluate     import evaluate_all, compute_transfer_score, print_benchmark_table
from causal_graph import EnvironmentSCM
from lambda_theorem import print_theorem2
from ccpl_theory import run_all_theory_checks, print_theory_report
from hallucination_fix import patch_ccpl_agent

STATE_DIM  = 6
ACTION_DIM = 5
TRAIN_ENVS = ("standard", "noisy", "shifted", "randomised")
EVAL_ENVS  = ("standard", "noisy", "shifted")
UNSEEN     = ("adversarial", "deceptive_reward", "resource_collapse",
               "deception_bench", "hidden_state_shift", "conflict_zone")


def _sep(title="", w=72):
    print("\n" + "=" * w)
    if title: print(f"  {title}"); print("=" * w)

def _bar(label, val, hi, w=28):
    f = int(w * max(0, val) / (hi + 1e-6))
    print(f"  {label:<26} [{'█'*f}{'░'*(w-f)}] {val:+.3f}")


def _train_ccpl(args, seed=None):
    seed = args.seed if seed is None else seed
    agent = make_ccpl(STATE_DIM, ACTION_DIM, seed=seed, pretrain_steps=args.pretrain)
    patch_ccpl_agent(agent)
    hist  = train_agent(agent, n_episodes=args.episodes, max_steps=args.max_steps,
                        delay_steps=args.delay, seed=seed, verbose=not args.quiet,
                        log_freq=max(1, args.episodes // 10), env_names=list(args.envs),
                        log_dir=args.out)
    return agent, hist


def run_demo(args):
    _sep("CCPL — Demo: Three-Direction Breakthrough vs Baselines")
    os.makedirs(args.out, exist_ok=True)

    print("\n  Three evaluated mechanisms:")
    print("  D1  Delay-aware cost attribution — E[γ^τ|h] scales attributed cost")
    print("  D2  State-conditioned penalty    — empirical, not a dominance theorem")
    print("  D3  Action-centred attribution   — causal only under SCM assumptions")

    agents = {
        "CCPL": make_ccpl(STATE_DIM, ACTION_DIM, seed=args.seed, pretrain_steps=args.pretrain),
        "CCPL-Base": make_ccpl_base(STATE_DIM, ACTION_DIM, seed=args.seed),
        "PPO":  PPOAgent(STATE_DIM, ACTION_DIM, seed=args.seed),
    }

    for name, agent in agents.items():
        _sep(f"Training  {name}")
        train_agent(agent, n_episodes=args.episodes, max_steps=args.max_steps,
                    delay_steps=args.delay, seed=args.seed, verbose=True,
                    log_freq=max(1, args.episodes // 10), env_names=list(args.envs),
                    log_dir=args.out)

    _sep("Evaluation")
    results = evaluate_all(agents, list(args.eval_envs),
                            args.eval_episodes, args.max_steps, args.delay)
    ts      = compute_transfer_score(results, list(args.eval_envs))
    cstats  = {n: {"param_count": count_parameters(a)}
               for n, a in agents.items()}
    print_benchmark_table(results, list(args.eval_envs), ts, cstats)

    _sep("CCPL Numerical Diagnostics")
    ccpl_agent = agents["CCPL"]
    logs = ccpl_agent.get_theory_logs()
    t1   = logs["theorem1"]
    t2   = logs["theorem2"]
    print("\n  Bellman/delay support check:")
    print(f"    {t1['proof']}")
    gamma_log = logs["gamma_eff_log"]
    if gamma_log:
        print(f"    γ_eff range: [{min(gamma_log):.4f}, {max(gamma_log):.4f}]")
    else:
        print("    γ_eff range: no action samples")
    print("\n  State-lambda diagnostics (no dominance theorem):")
    print(f"    Var_s[E[c|s]] = {t2['var_s']:.6f}")
    print("    ε-bound       = withdrawn")
    print(f"    Variation score = {t2.get('state_variation_score', 0):.4f}")
    print(f"    State conditioning active = {t2.get('state_conditioning_active', False)}")

    save_eval_results(results, "demo_eval", args.out)
    print(f"\n  Results: {args.out}/")


def run_ablation(args):
    _sep("CCPL — Ablation: Direction 1 / 2 / 3 Isolation")
    os.makedirs(args.out, exist_ok=True)

    suite = build_ccpl_ablation(STATE_DIM, ACTION_DIM, seed=args.seed)
    for name, agent in suite.items():
        _sep(f"Training: {name}")
        train_agent(agent, n_episodes=args.episodes, max_steps=args.max_steps,
                    delay_steps=args.delay, seed=args.seed, verbose=not args.quiet,
                    log_freq=max(1, args.episodes // 8), env_names=list(args.envs),
                    log_dir=args.out)

    results = evaluate_all(suite, list(args.eval_envs),
                            args.eval_episodes, args.max_steps, args.delay)
    ts      = compute_transfer_score(results, list(args.eval_envs))
    print_benchmark_table(results, list(args.eval_envs), ts)
    save_eval_results(results, "ablation_eval", args.out)

    _sep("Contribution of Each Direction (Δ vs CCPL baseline)")
    base = results.get("CCPL-Base", results.get("CCPL", {}))
    for name, res in results.items():
        if name == "CCPL-Base": continue
        dr = np.mean([res[e]["mean_reward"]      - base.get(e, {}).get("mean_reward",      0)
                      for e in args.eval_envs])
        dc = np.mean([res[e]["mean_consequence"] - base.get(e, {}).get("mean_consequence", 0)
                      for e in args.eval_envs])
        print(f"  {name:<20}  ΔR={dr:+.3f}  ΔJ_c={dc:+.4f}")


def run_train(args):
    _sep("CCPL — Full Training + Diagnostics")
    os.makedirs(args.out, exist_ok=True)

    agent, hist = _train_ccpl(args)
    _sep("Numerical and Empirical Diagnostics")
    eval_envs2 = list(args.eval_envs) + list(UNSEEN)[:3]
    results_for_theory = evaluate_all({"CCPL": agent}, eval_envs2,
                                       args.eval_episodes, args.max_steps, args.delay)
    checks = run_all_theory_checks(agent,
                                    histories={"main": hist},
                                    eval_results={"CCPL": results_for_theory.get("CCPL", {})},
                                    constraint_d=3.0)
    print_theory_report(checks)

    _sep("Direction 3: Causal Attribution Analysis")
    scm   = EnvironmentSCM()
    rng   = np.random.default_rng(0)
    states = rng.uniform(0.1, 0.9, (100, STATE_DIM)).astype(np.float32)
    safe_actions = [scm.safe_action(s) for s in states]
    causal_actions = [scm.most_causal_action(s) for s in states]
    print("  Most causal action distribution (FULL=2 should dominate):")
    for a, name in enumerate(["DEFER","PARTIAL","FULL","INVEST","REBALANCE"]):
        pct = 100 * causal_actions.count(a) / len(causal_actions)
        print(f"    {name:10s}: {pct:5.1f}% most causal  "
              f"{100*safe_actions.count(a)/len(safe_actions):5.1f}% safest")

    eval_envs = list(args.eval_envs) + list(UNSEEN)[:3]
    results   = evaluate_all({"CCPL": agent}, eval_envs,
                             args.eval_episodes, args.max_steps, args.delay)
    print_benchmark_table(results, eval_envs, compute_transfer_score(results, eval_envs))
    save_eval_results(results, "train_eval", args.out)
    print(f"\n  Params    : {hist['param_count']:,}")
    print(f"  Train time: {_fmt_eta(hist['total_train_s'])}")


def run_benchmark_full(args):
    _sep("CCPL — Full Benchmark vs All Baselines")
    os.makedirs(args.out, exist_ok=True)
    from constrained_baselines import CPOAgent, RCPOAgent, PIDLagrangianAgent, SACLagrangianAgent

    def _build(seed):
        return {
            "CCPL":    make_ccpl(STATE_DIM, ACTION_DIM, seed=seed, pretrain_steps=args.pretrain),
            "CCPL-Base": make_ccpl_base(STATE_DIM, ACTION_DIM, seed=seed),
            "PPO":     PPOAgent(STATE_DIM, ACTION_DIM, seed=seed),
            "A2C":     A2CAgent(STATE_DIM, ACTION_DIM, seed=seed),
            "DQN":     DQNAgent(STATE_DIM, ACTION_DIM, seed=seed),
            "CPO-FO":  CPOAgent(STATE_DIM, ACTION_DIM, seed=seed),
            "RCPO":    RCPOAgent(STATE_DIM, ACTION_DIM, seed=seed),
            "PID-Lag": PIDLagrangianAgent(STATE_DIM, ACTION_DIM, seed=seed),
            "SAC-Lag": SACLagrangianAgent(STATE_DIM, ACTION_DIM, seed=seed),
        }

    eval_envs   = list(args.eval_envs) + list(UNSEEN)[:3]
    all_results = {}
    for seed_i in range(args.seeds):
        seed = args.seed + seed_i * 100
        _sep(f"Seed {seed_i+1}/{args.seeds}")
        agents = _build(seed)
        for name, agent in agents.items():
            train_agent(agent, n_episodes=args.episodes, max_steps=args.max_steps,
                        delay_steps=args.delay, seed=seed, verbose=not args.quiet,
                        log_freq=max(1, args.episodes // 5), env_names=list(args.envs))
        res = evaluate_all(agents, eval_envs, args.eval_episodes, args.max_steps, args.delay)
        for name, r in res.items():
            all_results.setdefault(name, []).append(r)

    def _avg(res_list, envs):
        return {e: {k: float(np.mean([r[e][k] for r in res_list]))
                    for k in res_list[0][e] if isinstance(res_list[0][e][k], (int, float))}
                for e in envs}

    averaged = {n: _avg(v, eval_envs) for n, v in all_results.items()}
    xfer     = compute_transfer_score(averaged, eval_envs)
    print_benchmark_table(averaged, eval_envs, xfer)
    save_eval_results({
        "averaged": averaged,
        "seed_results": all_results,
        "seed_values": [args.seed + i * 100 for i in range(args.seeds)],
    }, "benchmark_eval", args.out)
    print(f"\n  Results: {args.out}/")


def run_theory(args):
    """Historical command name; runs diagnostics, not formal verification."""
    _sep("CCPL — Standalone Diagnostics")
    agent = make_ccpl(STATE_DIM, ACTION_DIM, seed=args.seed, pretrain_steps=100)
    from environments import StandardEnv
    for ep in range(min(args.episodes, 100)):
        env = StandardEnv(max_steps=100, seed=ep)
        run_episode(agent, env, train=True)

    logs = agent.get_theory_logs()
    _sep("D1: Bellman and delay-factor support")
    t1 = logs["theorem1"]
    for k, v in t1.items():
        print(f"  {k:<28}: {v}")

    _sep("D2: State-lambda claim correction")
    print_theorem2()
    t2 = logs["theorem2"]
    for k, v in t2.items():
        print(f"  {k:<28}: {v}")


def run_causal(args):
    """Causal attribution analysis of the environment."""
    _sep("CCPL — Structural Causal Model Analysis")
    scm  = EnvironmentSCM()
    rng  = np.random.default_rng(0)

    print("\n  Action consequences at a high-risk state [rl=0.4, fr=0.8, hpl=0.7]:")
    risky = np.array([0.4, 0.8, 0.5, 0.4, 0.3, 0.7], np.float32)
    for a, name in enumerate(["DEFER","PARTIAL","FULL","INVEST","REBALANCE"]):
        chain = scm.causal_chain(risky, a)
        attr  = scm.causal_attribution(risky, a)
        print(f"  {name:10s}: ΔC={attr:+.4f}  c={chain['c_total']:.4f}  "
              f"F={chain['F']:.3f} U={chain['U']:.3f} D={chain['D']:.3f}  "
              f"dominant={chain['dominant_pathway']}")

    print("\n  Causal attribution across 500 random states:")
    states = rng.uniform(0.1, 0.9, (500, STATE_DIM)).astype(np.float32)
    for a, name in enumerate(["DEFER","PARTIAL","FULL","INVEST","REBALANCE"]):
        attrs = [scm.causal_attribution(s, a) for s in states]
        print(f"  {name:10s}: mean ΔC={np.mean(attrs):+.4f}  std={np.std(attrs):.4f}  "
              f"frac_positive={np.mean(np.array(attrs)>0):.3f}")

    print("\n  In this specified SCM, FULL is the only action with positive mean ΔC.")
    print("  The contrast is relative to a uniform same-state action baseline;")
    print("  it should not be generalized beyond this SCM without identification data.")


def main(argv=None):
    p = argparse.ArgumentParser(description="CCPL — Delayed Causal Consequence-Penalized Learning")
    p.add_argument("mode", nargs="?", default="demo",
                   choices=["demo","train","ablation","benchmark","theory","causal"])
    p.add_argument("--episodes",      type=int,  default=300)
    p.add_argument("--eval-episodes", type=int,  default=50)
    p.add_argument("--max-steps",     type=int,  default=100)
    p.add_argument("--delay",         type=int,  default=5)
    p.add_argument("--pretrain",      type=int,  default=200,
                   help="ICN supervised pretraining steps on SCM labels")
    p.add_argument("--seed",          type=int,  default=42)
    p.add_argument("--seeds",         type=int,  default=3)
    p.add_argument("--out",           type=str,  default="results")
    p.add_argument("--envs",          nargs="+", default=list(TRAIN_ENVS))
    p.add_argument("--eval-envs",     nargs="+", default=list(EVAL_ENVS))
    p.add_argument("--quiet",         action="store_true")
    p.add_argument("--quick",         action="store_true",
                   help="200 episodes, 20 eval episodes, 1 seed")
    args = p.parse_args(argv)
    if args.quick:
        args.episodes, args.eval_episodes, args.seeds, args.pretrain = 200, 20, 1, 50

    {"demo": run_demo, "train": run_train, "ablation": run_ablation,
     "benchmark": run_benchmark_full, "theory": run_theory,
     "causal": run_causal}[args.mode](args)


if __name__ == "__main__":
    main()
