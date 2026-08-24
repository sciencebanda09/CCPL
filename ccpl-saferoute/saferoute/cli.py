from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ccpl import make_ccpl, run_episode

from .env import SafeRouteEnv
from .visualize import save_dashboard


def _agent(seed, state_dim=12):
    return make_ccpl(state_dim=state_dim, action_dim=5, seed=seed,
                     constraint_d=3.0)


def train(args):
    env = SafeRouteEnv(seed=args.seed, delay=args.delay)
    agent = _agent(args.seed)
    history = []
    window = max(1, min(25, args.episodes))
    for episode in range(args.episodes):
        result = run_episode(agent, env, train=True, update_freq=4)
        diagnostics = agent.diagnostics()
        history.append({
            "episode": episode + 1,
            "reward": result["episode_reward"],
            "cost": result["episode_consequence"],
            "steps": result["steps"],
            "lambda": diagnostics.get("mean_lambda", 0.0),
            "lambda_scale": diagnostics.get("lambda_scale", 1.0),
            "lambda_target": diagnostics.get("lambda_target", 0.0),
            "jc_violation": diagnostics.get("jc_violation", 0.0),
            "hit_freq_ema": diagnostics.get("hit_freq_ema", 0.0),
        })
        if (episode + 1) % max(1, args.episodes // 10) == 0:
            recent = history[-window:]
            costs = [item["cost"] for item in recent]
            rewards = [item["reward"] for item in recent]
            csr = sum(cost <= 3.0 for cost in costs) / len(costs)
            print(f"episode={episode + 1} reward={history[-1]['reward']:.3f} "
                  f"cost={history[-1]['cost']:.3f} "
                  f"window_reward={np.mean(rewards):.3f} "
                  f"window_cost={np.mean(costs):.3f} "
                  f"window_CSR={csr:.1%} "
                  f"lambda={diagnostics.get('mean_lambda', 0.0):.3f} "
                  f"lambda_target={diagnostics.get('lambda_target', 0.0):.3f} "
                  f"lambda_scale={diagnostics.get('lambda_scale', 1.0):.3f}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    agent.save(output)
    output.with_suffix(".json").write_text(
        json.dumps({"history": history, "delay": args.delay}, indent=2),
        encoding="utf-8",
    )
    print(f"checkpoint={output}")


def evaluate(args):
    env = SafeRouteEnv(seed=args.seed, delay=args.delay)
    from ccpl import CCPLAgent
    agent = CCPLAgent.load(args.checkpoint)
    results = []
    for index in range(args.episodes):
        result = run_episode(agent, env, train=False)
        stats = env.episode_stats()
        results.append({
            "episode": index + 1,
            "reward": result["episode_reward"],
            "cost": result["episode_consequence"],
            "steps": result["steps"],
            "route_complete": stats["route_complete"],
            "trace": stats["trace"],
            "size": stats["size"],
            "max_steps": stats["max_steps"],
            "delay": stats["delay"],
            "goal": list(env.goal),
            "hazards": stats["hazards"],
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"episodes": results}, indent=2), encoding="utf-8")
    rewards = np.asarray([item["reward"] for item in results], dtype=np.float64)
    costs = np.asarray([item["cost"] for item in results], dtype=np.float64)
    completed = np.asarray([item["route_complete"] for item in results], dtype=bool)
    print(f"mean_reward={rewards.mean():+.3f} mean_cost={costs.mean():.3f} "
          f"CSR={np.mean(costs <= 3.0):.1%} "
          f"complete={completed.mean():.1%} max_cost={costs.max():.3f}")
    print(f"evaluation={output}")


def _reactive_action(state):
    x, y, gx, gy = np.asarray(state[:4], dtype=np.float32)
    hazard_up, hazard_down, hazard_left, hazard_right = state[6:10]
    dx = gx - x
    dy = gy - y
    candidates = [
        (0, -dy, hazard_up),
        (1, dy, hazard_down),
        (2, -dx, hazard_left),
        (3, dx, hazard_right),
        (4, 0.25, 0.0),
    ]
    safe = [item for item in candidates if item[2] < 0.5]
    return int(max(safe, key=lambda item: item[1])[0])


def _oracle_action(env):
    distances = []
    for action, (dx, dy) in enumerate(env.ACTIONS):
        candidate = (env.position[0] + dx, env.position[1] + dy)
        if not env._inside(candidate) or candidate in env.hazards:
            distances.append(float("inf"))
        else:
            distances.append(env._distance(candidate, env.goal))
    return int(np.argmin(distances))


def compare(args):
    rows = []
    for name in ("random", "reactive", "oracle"):
        rewards, costs, completed = [], [], []
        for episode in range(args.episodes):
            env = SafeRouteEnv(seed=args.seed + episode, delay=args.delay)
            total_reward = 0.0
            while not env.done:
                state = env._observation()
                if name == "random":
                    action = int(env.rng.integers(0, 5))
                elif name == "reactive":
                    action = _reactive_action(state)
                else:
                    action = _oracle_action(env)
                _, reward, _, _, _ = env.step(action)
                total_reward += reward
            stats = env.episode_stats()
            rewards.append(total_reward)
            costs.append(stats["total_consequence"])
            completed.append(stats["route_complete"])
        rows.append({
            "agent": name,
            "mean_reward": float(np.mean(rewards)),
            "mean_cost": float(np.mean(costs)),
            "constraint_satisfaction_rate": float(np.mean(np.asarray(costs) <= 3.0)),
            "route_completion_rate": float(np.mean(completed)),
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"episodes": args.episodes, "results": rows}, indent=2), encoding="utf-8")
    for row in rows:
        print(f"{row['agent']:<8} reward={row['mean_reward']:+.3f} "
              f"cost={row['mean_cost']:.3f} CSR={row['constraint_satisfaction_rate']:.1%} "
              f"complete={row['route_completion_rate']:.1%}")
    print(f"comparison={output}")


def render(args):
    env = SafeRouteEnv(seed=args.seed, delay=args.delay)
    env.reset(seed=args.seed)
    print(env.render_text())
    print(f"step={env.step_count} position={env.position} goal={env.goal} delay={env.delay}")


def dashboard(args):
    save_dashboard(args.evaluation, args.output)
    print(f"dashboard={args.output}")


def sweep(args):
    from ccpl import CCPLAgent

    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    root = Path(args.output)
    summaries = []
    for seed in seeds:
        checkpoint = root / f"ccpl_seed{seed}.pkl"
        train_args = argparse.Namespace(
            episodes=args.episodes,
            delay=args.delay,
            seed=seed,
            output=str(checkpoint),
        )
        train(train_args)
        env = SafeRouteEnv(seed=seed, delay=args.delay)
        agent = CCPLAgent.load(checkpoint)
        rewards, costs, completed = [], [], []
        for episode in range(args.eval_episodes):
            result = run_episode(agent, env, train=False)
            stats = env.episode_stats()
            rewards.append(result["episode_reward"])
            costs.append(result["episode_consequence"])
            completed.append(stats["route_complete"])
        summary = {
            "seed": seed,
            "mean_reward": float(np.mean(rewards)),
            "mean_cost": float(np.mean(costs)),
            "csr": float(np.mean(np.asarray(costs) <= 3.0)),
            "completion": float(np.mean(completed)),
            "max_cost": float(np.max(costs)),
        }
        summaries.append(summary)
        print(f"seed={seed} reward={summary['mean_reward']:+.3f} "
              f"cost={summary['mean_cost']:.3f} CSR={summary['csr']:.1%} "
              f"complete={summary['completion']:.1%}")
    aggregate = {
        key: {
            "mean": float(np.mean([row[key] for row in summaries])),
            "std": float(np.std([row[key] for row in summaries], ddof=1))
            if len(summaries) > 1 else 0.0,
        }
        for key in ("mean_reward", "mean_cost", "csr", "completion", "max_cost")
    }
    root.mkdir(parents=True, exist_ok=True)
    output = root / "sweep_summary.json"
    output.write_text(json.dumps({
        "episodes": args.episodes,
        "eval_episodes": args.eval_episodes,
        "delay": args.delay,
        "seeds": seeds,
        "per_seed": summaries,
        "aggregate": aggregate,
    }, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    print(f"sweep={output}")


def main():
    parser = argparse.ArgumentParser(prog="saferoute")
    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser("train")
    train_parser.add_argument("--episodes", type=int, default=300)
    train_parser.add_argument("--delay", type=int, default=3)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--output", default="results/ccpl.pkl")
    train_parser.set_defaults(func=train)

    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("--checkpoint", default="results/ccpl.pkl")
    eval_parser.add_argument("--episodes", type=int, default=20)
    eval_parser.add_argument("--delay", type=int, default=3)
    eval_parser.add_argument("--seed", type=int, default=43)
    eval_parser.add_argument("--output", default="results/evaluation.json")
    eval_parser.set_defaults(func=evaluate)

    view_parser = sub.add_parser("dashboard")
    view_parser.add_argument("--evaluation", default="results/evaluation.json")
    view_parser.add_argument("--output", default="results/dashboard.png")
    view_parser.set_defaults(func=dashboard)

    sweep_parser = sub.add_parser("sweep")
    sweep_parser.add_argument("--episodes", type=int, default=1000)
    sweep_parser.add_argument("--eval-episodes", type=int, default=100)
    sweep_parser.add_argument("--seeds", default="42,43,44")
    sweep_parser.add_argument("--delay", type=int, default=3)
    sweep_parser.add_argument("--output", default="results/sweep")
    sweep_parser.set_defaults(func=sweep)

    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("--episodes", type=int, default=20)
    compare_parser.add_argument("--delay", type=int, default=3)
    compare_parser.add_argument("--seed", type=int, default=42)
    compare_parser.add_argument("--output", default="results/baselines.json")
    compare_parser.set_defaults(func=compare)

    render_parser = sub.add_parser("render")
    render_parser.add_argument("--delay", type=int, default=3)
    render_parser.add_argument("--seed", type=int, default=42)
    render_parser.set_defaults(func=render)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
