"""Generate a summary figure from the stored aggregate JSON files.

Importing this module is side-effect free.  The repository's existing JSON
files predate the audit and contain no seed-level provenance, so figure titles
describe them as stored aggregates rather than validated paper results.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE = Path(__file__).resolve().parent / "results_v7"
COLORS = {
    "CCPL": "#2563EB", "DQN": "#DC2626", "PPO": "#D97706",
    "A2C": "#7C3AED", "CPO": "#059669", "CPO-FO": "#059669", "RCPO": "#0891B2",
    "PID-Lag": "#B45309", "SAC-Lag": "#DB2777",
    "CCPL-Base": "#9CA3AF", "CCPL-NoDelay": "#F97316",
    "CCPL-NoStateλ": "#A855F7", "CCPL-NoCausal": "#EF4444",
    "CCPL-SingleQ": "#10B981",
}


def _load(name: str) -> dict:
    with (BASE / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def _agent_metric(record: dict, key: str) -> float:
    aliases = {
        "reward": ("reward", "mean_reward"),
        "CSR": ("CSR", "constraint_satisfaction_rate"),
    }
    for candidate in aliases[key]:
        if candidate in record:
            return float(record[candidate])
    return float("nan")


def generate_figure(output_path: str | Path | None = None) -> Path:
    main = _load("benchmark_results.json")
    ablation = _load("ablation_results.json")
    safety = _load("safety_gym_results.json")

    figure, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax_main, ax_ablation, ax_safety_reward, ax_safety_csr = axes.ravel()

    for name, values in main.items():
        reward = _agent_metric(values, "reward")
        csr = _agent_metric(values, "CSR")
        ax_main.scatter(
            reward, csr, s=180 if name == "CCPL" else 80,
            marker="*" if name == "CCPL" else "o",
            color=COLORS.get(name, "#6B7280"), label=name,
        )
        ax_main.annotate(name, (reward, csr), xytext=(4, 4),
                         textcoords="offset points", fontsize=8)
    ax_main.set(xlabel="Mean episode reward", ylabel="CSR (%)",
                title="Stored benchmark aggregates: reward vs safety")
    ax_main.grid(alpha=0.25)

    ablation_order = [name for name in (
        "CCPL", "CCPL-NoDelay", "CCPL-NoStateλ", "CCPL-NoCausal",
        "CCPL-SingleQ", "CCPL-Base") if name in ablation]
    x = np.arange(len(ablation_order))
    rewards = [_agent_metric(ablation[name], "reward") for name in ablation_order]
    csrs = [_agent_metric(ablation[name], "CSR") / 20.0 for name in ablation_order]
    ax_ablation.bar(x - 0.18, rewards, 0.36,
                    color=[COLORS.get(name, "#6B7280") for name in ablation_order],
                    label="reward")
    ax_ablation.bar(x + 0.18, csrs, 0.36, color="#94A3B8", alpha=0.65,
                    label="CSR / 20")
    ax_ablation.set_xticks(x, [name.replace("CCPL-", "") for name in ablation_order],
                           rotation=20, ha="right")
    ax_ablation.set_title("Stored ablation aggregates")
    ax_ablation.legend()
    ax_ablation.grid(axis="y", alpha=0.25)

    environments = [key for key, value in safety.items() if isinstance(value, dict)]
    agents = []
    for env_name in environments:
        for agent_name in safety[env_name]:
            if agent_name not in agents:
                agents.append(agent_name)
    width = 0.8 / max(len(environments), 1)
    x = np.arange(len(agents))
    for index, env_name in enumerate(environments):
        offset = (index - (len(environments) - 1) / 2.0) * width
        reward_values = [
            _agent_metric(safety[env_name].get(name, {}), "reward") for name in agents
        ]
        csr_values = [
            _agent_metric(safety[env_name].get(name, {}), "CSR") for name in agents
        ]
        ax_safety_reward.bar(x + offset, reward_values, width, label=env_name)
        ax_safety_csr.bar(x + offset, csr_values, width, label=env_name)

    for axis, metric in ((ax_safety_reward, "reward"), (ax_safety_csr, "CSR (%)")):
        axis.set_xticks(x, agents, rotation=20, ha="right")
        axis.set_ylabel(metric)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)
    ax_safety_reward.set_title("Stored safety-environment reward aggregates")
    ax_safety_csr.set_title("Stored safety-environment CSR aggregates")

    figure.suptitle(
        "CCPL stored aggregates — provenance and paper consistency not validated",
        fontsize=14,
    )
    figure.tight_layout()
    destination = Path(output_path) if output_path else BASE / "ccpl_v7_results.png"
    figure.savefig(destination, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return destination


if __name__ == "__main__":
    print(f"Saved to {generate_figure()}")
