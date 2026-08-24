import argparse
import json
from pathlib import Path

import numpy as np

from ccpl import CCPLAgent, ENV_REGISTRY, make_env, run_episode


def _evaluate(agent, env_name, delay, episodes, seeds, max_steps, threshold):
    rows = []
    for seed in seeds:
        env = make_env(env_name, max_steps=max_steps, consequence_delay=delay, seed=seed)
        rewards, costs, completed = [], [], []
        for _ in range(episodes):
            result = run_episode(agent, env, train=False)
            stats = env.episode_stats()
            rewards.append(result["episode_reward"])
            costs.append(result["episode_consequence"])
            if "route_complete" in stats:
                completed.append(bool(stats["route_complete"]))
        row = {
            "seed": seed,
            "mean_reward": float(np.mean(rewards)),
            "mean_cost": float(np.mean(costs)),
            "csr": float(np.mean(np.asarray(costs) <= threshold)),
            "max_cost": float(np.max(costs)),
        }
        if completed:
            row["completion"] = float(np.mean(completed))
        rows.append(row)
    return rows


def audit(args):
    agent = CCPLAgent.load(args.checkpoint)
    env_names = args.environments or list(ENV_REGISTRY)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    delays = [int(value) for value in args.delays.split(",") if value.strip()]
    results = {}
    for env_name in env_names:
        if env_name not in ENV_REGISTRY:
            raise ValueError(f"unknown environment: {env_name}")
        results[env_name] = {}
        for delay in delays:
            rows = _evaluate(agent, env_name, delay, args.episodes, seeds,
                             args.max_steps, args.threshold)
            results[env_name][str(delay)] = rows

    flat = [row for env_data in results.values() for rows in env_data.values() for row in rows]
    completion_values = [row["completion"] for row in flat if "completion" in row]
    summary = {
        "mean_reward": float(np.mean([row["mean_reward"] for row in flat])),
        "mean_cost": float(np.mean([row["mean_cost"] for row in flat])),
        "mean_csr": float(np.mean([row["csr"] for row in flat])),
        "minimum_csr": float(np.min([row["csr"] for row in flat])),
        "maximum_cost": float(np.max([row["max_cost"] for row in flat])),
        "complete_rate": (float(np.mean(completion_values))
                          if completion_values else None),
    }
    report = {
        "protocol": {
            "checkpoint": str(args.checkpoint),
            "environments": env_names,
            "delays": delays,
            "seeds": seeds,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "threshold": args.threshold,
            "claim_scope": "robustness under the tested distribution; not universal safety",
        },
        "summary": summary,
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"report={output}")


def main():
    parser = argparse.ArgumentParser(prog="safety-audit")
    sub = parser.add_subparsers(dest="command", required=True)
    audit_parser = sub.add_parser("audit")
    audit_parser.add_argument("--checkpoint", required=True)
    audit_parser.add_argument("--episodes", type=int, default=100)
    audit_parser.add_argument("--seeds", default="42,43,44,45,46")
    audit_parser.add_argument("--delays", default="0,2,5,10")
    audit_parser.add_argument("--environments", nargs="*", default=None)
    audit_parser.add_argument("--max-steps", type=int, default=100)
    audit_parser.add_argument("--threshold", type=float, default=3.0)
    audit_parser.add_argument("--output", default="results/safety_audit.json")
    audit_parser.set_defaults(func=audit)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
