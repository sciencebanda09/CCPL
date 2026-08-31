from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ccpl-saferoute"))

from ccpl import CCPLAgent, run_episode
from world import FreightWorldEnv


def generate(checkpoint: Path, output: Path, seeds: list[int], delay: int) -> None:
    base_agent = CCPLAgent.load(checkpoint)
    rollouts = []
    for index, seed in enumerate(seeds):
        agent = copy.deepcopy(base_agent)
        env = FreightWorldEnv(seed=seed, delay=delay, max_steps=80)
        result = run_episode(agent, env, train=False)
        stats = env.episode_stats()
        rollouts.append({
            "id": f"CCPL-{index + 1:02d}",
            "policy": "CCPL",
            "seed": seed,
            "reward": float(result["episode_reward"]),
            "cost": float(result["episode_consequence"]),
            "complete": bool(stats["route_complete"]),
            "trace": stats["trace"],
            "hazards": [list(item) for item in stats["hazards"]],
            "goal": list(env.goal),
        })
    payload = {
        "title": "Delayed Horizon",
        "source": "CCPL checkpoint rollout in FreightWorldEnv",
        "checkpoint": str(checkpoint),
        "delay": delay,
        "budget": 3.0,
        "size": 12,
        "rollouts": rollouts,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"generated={output}")
    for rollout in rollouts:
        print(f"{rollout['id']} seed={rollout['seed']} reward={rollout['reward']:+.3f} "
              f"cost={rollout['cost']:.3f} complete={rollout['complete']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Delayed Horizon CCPL rollouts")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", default=Path("data/rollouts.json"), type=Path)
    parser.add_argument("--seeds", default="42,46,146,222,555,777")
    parser.add_argument("--delay", default=10, type=int)
    args = parser.parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise SystemExit("--seeds must contain at least one integer")
    generate(args.checkpoint, args.output, seeds, args.delay)


if __name__ == "__main__":
    main()
