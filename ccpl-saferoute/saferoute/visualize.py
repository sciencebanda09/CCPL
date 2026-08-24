from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_dashboard(evaluation_path, output_path):
    data = json.loads(Path(evaluation_path).read_text(encoding="utf-8"))
    episodes = data["episodes"]
    if not episodes:
        raise ValueError("evaluation contains no episodes")
    trace = episodes[-1]["trace"]
    size = int(episodes[-1].get("size", 12))
    goal = tuple(episodes[-1].get("goal", (size - 2, size - 2)))
    hazards = {tuple(item) for item in episodes[-1].get("hazards", _hazards(size))}
    positions = np.asarray([item["position"] for item in trace], dtype=int)
    rewards = [item["reward"] for item in trace]
    costs = [item["cost"] for item in trace]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].set_title("SafeRoute episode")
    axes[0].set_xlim(-0.5, size - 0.5)
    axes[0].set_ylim(-0.5, size - 0.5)
    axes[0].set_aspect("equal")
    axes[0].set_xticks(range(size))
    axes[0].set_yticks(range(size))
    axes[0].grid(alpha=0.25)
    hazard_x, hazard_y = zip(*hazards)
    axes[0].scatter(hazard_x, hazard_y, marker="s", s=80, color="#F97316",
                    label="hazard")
    axes[0].plot(positions[:, 0], positions[:, 1], color="#2563EB", linewidth=2)
    axes[0].scatter(positions[0, 0], positions[0, 1], color="#16A34A", s=90,
                    label="start")
    axes[0].scatter(goal[0], goal[1], color="#7C3AED", s=90, label="goal")
    axes[0].legend(loc="upper left", fontsize=8)

    axes[1].set_title("Reward")
    axes[1].plot(np.cumsum(rewards), color="#2563EB")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("cumulative reward")
    axes[1].grid(alpha=0.25)

    axes[2].set_title("Delayed consequence")
    axes[2].bar(range(len(costs)), costs, color="#DC2626")
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("emitted cost")
    axes[2].grid(alpha=0.25)

    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def _hazards(size):
    center = size // 2
    return {
        (center - 2, center), (center - 1, center), (center, center),
        (center + 1, center), (center + 2, center),
        (center, center - 1), (center, center + 1),
    }
