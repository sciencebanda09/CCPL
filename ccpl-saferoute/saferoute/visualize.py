from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
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


def save_3d_episode(trace, size, goal, hazards, output_path, fps=8, azim=-58):
    if not trace:
        raise ValueError("episode trace is empty")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    positions = [tuple(item["position"]) for item in trace]
    costs = [float(item.get("cost", 0.0)) for item in trace]
    hazard_set = {tuple(item) for item in hazards}

    figure = plt.figure(figsize=(8, 6))
    axis = figure.add_subplot(111, projection="3d")
    axis.set_xlim(-0.5, size - 0.5)
    axis.set_ylim(-0.5, size - 0.5)
    axis.set_zlim(0, 1.8)
    axis.set_xticks(range(size))
    axis.set_yticks(range(size))
    axis.set_zticks([0, 1, 2])
    axis.set_xlabel("row")
    axis.set_ylabel("column")
    axis.set_zlabel("risk surface")
    axis.view_init(elev=34, azim=azim)

    grid = np.linspace(0, size - 1, size * 8)
    xx, yy = np.meshgrid(grid, grid)
    risk_surface = np.zeros_like(xx, dtype=float)
    for row, column in hazard_set:
        risk_surface += np.exp(-((xx - row) ** 2 + (yy - column) ** 2) / 1.8)
    risk_surface = np.clip(0.08 + 0.72 * risk_surface, 0.08, 0.95)
    axis.plot_surface(xx, yy, risk_surface, cmap="magma", alpha=0.78,
                      linewidth=0, antialiased=True, shade=True)
    for row, column in hazard_set:
        axis.plot([row - 0.42, row + 0.42, row + 0.42, row - 0.42, row - 0.42],
                  [column - 0.42, column - 0.42, column + 0.42, column + 0.42, column - 0.42],
                  [0.04] * 5, color="#ef476f", linewidth=1.4, alpha=0.9)
    axis.scatter([goal[0]], [goal[1]], [0.95], marker="*", s=220,
                 color="#06d6a0", edgecolors="white", linewidths=0.8)

    path, = axis.plot([], [], [], color="#264653", linewidth=2.5)
    agent = axis.scatter([], [], [], color="#118ab2", edgecolors="white",
                         s=120, depthshade=False)
    pulse, = axis.plot([], [], [], color="#ffd166", linewidth=2.0, alpha=0.0)

    def update(frame):
        current = positions[:frame + 1]
        path.set_data([item[0] for item in current], [item[1] for item in current])
        path.set_3d_properties([0.12] * len(current))
        row, column = positions[frame]
        surface_height = float(risk_surface[min(row * 8, risk_surface.shape[0] - 1),
                                            min(column * 8, risk_surface.shape[1] - 1)])
        agent._offsets3d = ([row], [column], [surface_height + 0.08])
        if costs[frame] > 0:
            radius = 0.35 + 0.08 * min(frame, 8)
            angles = np.linspace(0, 2 * np.pi, 40)
            pulse.set_data(row + radius * np.cos(angles), column + radius * np.sin(angles))
            pulse.set_3d_properties([0.16] * len(angles))
            pulse.set_alpha(min(0.85, costs[frame] / 3.0))
        else:
            pulse.set_data([], [])
            pulse.set_3d_properties([])
            pulse.set_alpha(0.0)
        axis.view_init(elev=34, azim=azim + 0.8 * frame)
        axis.set_title(f"CCPL SafeRoute  |  step {frame}  |  delayed cost {costs[frame]:.3f}")
        return path, agent, pulse

    animation = FuncAnimation(figure, update, frames=len(positions), interval=1000 / max(fps, 1),
                              blit=False, repeat=False)
    if output.suffix.lower() != ".gif":
        raise ValueError("3D recording currently requires a .gif output path")
    animation.save(output, writer=PillowWriter(fps=max(fps, 1)))
    plt.close(figure)


def _save_animation(animation, output, fps):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() != ".gif":
        raise ValueError("animation output must use the .gif suffix")
    animation.save(output, writer=PillowWriter(fps=max(fps, 1)))


def save_topdown_episode(trace, size, goal, hazards, output_path, fps=8):
    positions = [tuple(item["position"]) for item in trace]
    costs = [float(item.get("cost", 0.0)) for item in trace]
    figure, axis = plt.subplots(figsize=(7, 7))
    hazard_set = {tuple(item) for item in hazards}
    axis.set_xlim(-0.5, size - 0.5)
    axis.set_ylim(-0.5, size - 0.5)
    axis.set_aspect("equal")
    axis.set_xticks(range(size))
    axis.set_yticks(range(size))
    for row, column in hazard_set:
        axis.add_patch(plt.Rectangle((row - 0.45, column - 0.45), 0.9, 0.9,
                                     color="#ef476f", alpha=0.55))
    axis.scatter([goal[0]], [goal[1]], marker="*", s=260, color="#06d6a0",
                 edgecolors="white", zorder=4)
    line, = axis.plot([], [], color="#118ab2", linewidth=3)
    point, = axis.plot([], [], "o", color="#073b4c", markersize=9)

    def update(frame):
        current = positions[:frame + 1]
        line.set_data([item[0] for item in current], [item[1] for item in current])
        point.set_data([positions[frame][0]], [positions[frame][1]])
        axis.set_title(f"CCPL SafeRoute | step {frame} | delayed cost {costs[frame]:.3f}")
        return line, point

    animation = FuncAnimation(figure, update, frames=len(positions), interval=1000 / max(fps, 1),
                              blit=False, repeat=False)
    _save_animation(animation, output_path, fps)
    plt.close(figure)


def save_timeline(trace, output_path, fps=8):
    costs = np.asarray([float(item.get("cost", 0.0)) for item in trace])
    immediate = np.asarray([float(item.get("immediate_cost", 0.0)) for item in trace])
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.set_xlim(0, max(len(costs) - 1, 1))
    axis.set_ylim(0, max(float(immediate.max()), float(costs.max()), 1.0) * 1.2)
    axis.set_xlabel("step")
    axis.set_ylabel("cost")
    axis.plot(immediate, color="#ef476f", alpha=0.35, label="immediate risk")
    delayed, = axis.plot([], [], color="#118ab2", linewidth=2.5, label="emitted delayed cost")
    cursor = axis.axvline(0, color="#073b4c", linewidth=1.2)
    axis.legend(loc="upper right")

    def update(frame):
        delayed.set_data(np.arange(frame + 1), costs[:frame + 1])
        cursor.set_xdata([frame, frame])
        axis.set_title(f"Delayed consequence | step {frame}")
        return delayed, cursor

    animation = FuncAnimation(figure, update, frames=len(costs), interval=1000 / max(fps, 1),
                              blit=False, repeat=False)
    _save_animation(animation, output_path, fps)
    plt.close(figure)


def save_policy_comparison(traces, size, goal, hazards, output_path, fps=8):
    names = list(traces)
    figure, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4), squeeze=False)
    axes = axes[0]
    artists = []
    for axis, name in zip(axes, names):
        axis.set_title(name)
        axis.set_xlim(-0.5, size - 0.5)
        axis.set_ylim(-0.5, size - 0.5)
        axis.set_aspect("equal")
        for row, column in hazards:
            axis.add_patch(plt.Rectangle((row - 0.45, column - 0.45), 0.9, 0.9,
                                         color="#ef476f", alpha=0.5))
        axis.scatter([goal[0]], [goal[1]], marker="*", s=160, color="#06d6a0")
        line, = axis.plot([], [], linewidth=2.5)
        point, = axis.plot([], [], "o", markersize=7)
        artists.append((line, point))
    length = max(len(trace) for trace in traces.values())

    def update(frame):
        updated = []
        for name, (line, point) in zip(names, artists):
            positions = [tuple(item["position"]) for item in traces[name]]
            index = min(frame, len(positions) - 1)
            current = positions[:index + 1]
            line.set_data([item[0] for item in current], [item[1] for item in current])
            point.set_data([positions[index][0]], [positions[index][1]])
            updated.extend([line, point])
        return updated

    animation = FuncAnimation(figure, update, frames=length, interval=1000 / max(fps, 1),
                              blit=False, repeat=False)
    _save_animation(animation, output_path, fps)
    plt.close(figure)


def save_metric_animation(history, output_path, fps=6):
    if not history:
        raise ValueError("training history is empty")
    episodes = np.asarray([item["episode"] for item in history])
    rewards = np.asarray([item["reward"] for item in history])
    costs = np.asarray([item["cost"] for item in history])
    lambdas = np.asarray([item.get("lambda", 0.0) for item in history])
    targets = np.asarray([item.get("lambda_target", 0.0) for item in history])
    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].set_ylabel("reward / cost")
    axes[1].set_ylabel("lambda")
    axes[1].set_xlabel("episode")
    lines = [axes[0].plot([], [], color="#118ab2", label="reward")[0],
             axes[0].plot([], [], color="#ef476f", label="cost")[0],
             axes[1].plot([], [], color="#06d6a0", label="lambda")[0],
             axes[1].plot([], [], color="#f4a261", label="target")[0]]
    axes[0].legend(loc="upper left")
    axes[1].legend(loc="upper left")
    axes[0].set_xlim(episodes[0], episodes[-1])
    axes[0].set_ylim(min(rewards.min(), costs.min(), 0) - 0.2,
                     max(rewards.max(), costs.max(), 1) * 1.1)
    axes[1].set_ylim(0, max(lambdas.max(), targets.max(), 1) * 1.1)

    def update(frame):
        index = frame + 1
        lines[0].set_data(episodes[:index], rewards[:index])
        lines[1].set_data(episodes[:index], costs[:index])
        lines[2].set_data(episodes[:index], lambdas[:index])
        lines[3].set_data(episodes[:index], targets[:index])
        figure.suptitle(f"CCPL training diagnostics | episode {episodes[frame]}")
        return lines

    animation = FuncAnimation(figure, update, frames=len(history), interval=1000 / max(fps, 1),
                              blit=False, repeat=False)
    _save_animation(animation, output_path, fps)
    plt.close(figure)


def save_heatmap_episode(trace, size, goal, hazards, output_path, fps=8):
    positions = [tuple(item["position"]) for item in trace]
    hazard_set = {tuple(item) for item in hazards}
    figure, axis = plt.subplots(figsize=(7, 7))
    heat = np.zeros((size, size), dtype=float)
    for row, column in hazard_set:
        for x in range(size):
            for y in range(size):
                heat[x, y] += np.exp(-((x - row) ** 2 + (y - column) ** 2) / 2.0)
    image = axis.imshow(heat.T, origin="lower", cmap="magma", vmin=0, vmax=max(1.0, heat.max()))
    axis.scatter([goal[0]], [goal[1]], marker="*", s=220, color="#06d6a0")
    line, = axis.plot([], [], color="#f8f9fa", linewidth=2.5)
    point, = axis.plot([], [], "o", color="#118ab2", markersize=9)
    figure.colorbar(image, ax=axis, label="hazard influence")

    def update(frame):
        current = positions[:frame + 1]
        line.set_data([item[0] for item in current], [item[1] for item in current])
        point.set_data([positions[frame][0]], [positions[frame][1]])
        axis.set_title(f"Risk heatmap | step {frame}")
        return line, point

    animation = FuncAnimation(figure, update, frames=len(positions), interval=1000 / max(fps, 1),
                              blit=False, repeat=False)
    _save_animation(animation, output_path, fps)
    plt.close(figure)


def save_lambda_dashboard(history, output_path, fps=6):
    if not history:
        raise ValueError("training history is empty")
    episodes = np.asarray([item["episode"] for item in history])
    lambdas = np.asarray([item.get("lambda", 0.0) for item in history])
    targets = np.asarray([item.get("lambda_target", 0.0) for item in history])
    scales = np.asarray([item.get("lambda_scale", 1.0) for item in history])
    costs = np.asarray([item["cost"] for item in history])
    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].set_ylabel("constraint weight")
    axes[1].set_ylabel("cost")
    axes[1].set_xlabel("episode")
    weight_lines = [axes[0].plot([], [], color="#118ab2", label="lambda")[0],
                    axes[0].plot([], [], color="#ef476f", label="target")[0],
                    axes[0].plot([], [], color="#06d6a0", label="warm-up scale")[0]]
    cost_line, = axes[1].plot([], [], color="#f4a261", label="cost")
    axes[0].legend(loc="upper left")
    axes[1].legend(loc="upper left")
    axes[0].set_xlim(episodes[0], episodes[-1])
    axes[0].set_ylim(0, max(lambdas.max(), targets.max(), scales.max(), 1) * 1.1)
    axes[1].set_ylim(0, max(float(costs.max()), 1) * 1.1)

    def update(frame):
        index = frame + 1
        weight_lines[0].set_data(episodes[:index], lambdas[:index])
        weight_lines[1].set_data(episodes[:index], targets[:index])
        weight_lines[2].set_data(episodes[:index], scales[:index])
        cost_line.set_data(episodes[:index], costs[:index])
        figure.suptitle(f"Constraint-weight schedule | episode {episodes[frame]}")
        return (*weight_lines, cost_line)

    animation = FuncAnimation(figure, update, frames=len(history), interval=1000 / max(fps, 1),
                              blit=False, repeat=False)
    _save_animation(animation, output_path, fps)
    plt.close(figure)


def _hazards(size):
    center = size // 2
    return {
        (center - 2, center), (center - 1, center), (center, center),
        (center + 1, center), (center + 2, center),
        (center, center - 1), (center, center + 1),
    }
