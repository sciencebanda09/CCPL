"""Matched scalar-lambda versus state-lambda experiment."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ccpl" / "algorithms"))
sys.path.insert(0, str(ROOT / "ccpl" / "environments"))

from ccpl_agent import make_ccpl
from environments import StandardEnv
from ccpl_agent import run_episode


def run_comparison(episodes=100, evaluation_episodes=20, seeds=(0, 1, 2), max_steps=50):
    rows = []
    for seed in seeds:
        common = dict(pretrain_steps=0, buffer_capacity=512, batch_size=8,
                      gru_dim=16, hidden_dim=32, n_layers=1)
        state_agent = make_ccpl(6, 5, seed=seed, **common)
        scalar_agent = make_ccpl(6, 5, seed=seed, **common)
        original_forward = scalar_agent.lambda_net.forward
        def scalar_forward(states):
            values = np.asarray(original_forward(states), dtype=np.float32)
            return np.full_like(values, values.mean()) if values.ndim else values
        scalar_agent.lambda_net.forward = scalar_forward
        agents = {"state_lambda": state_agent, "scalar_lambda": scalar_agent}
        for name, agent in agents.items():
            for episode in range(episodes):
                env = StandardEnv(max_steps=max_steps, consequence_delay=2, seed=seed * 10000 + episode)
                run_episode(agent, env, train=True, update_freq=4)
            rewards, costs = [], []
            for episode in range(evaluation_episodes):
                env = StandardEnv(max_steps=max_steps, consequence_delay=2,
                                  seed=seed * 10000 + 100000 + episode)
                result = run_episode(agent, env, train=False)
                rewards.append(result["episode_reward"])
                costs.append(result["episode_consequence"])
            rows.append({"method": name, "seed": seed,
                         "mean_reward": float(np.mean(rewards)),
                         "mean_cost": float(np.mean(costs)),
                         "constraint_satisfaction_rate": float(np.mean(np.asarray(costs) <= 3.0))})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, default=Path("results/lambda_comparison.json"))
    args = parser.parse_args()
    rows = run_comparison(args.episodes, args.eval_episodes, args.seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"rows": rows}, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
