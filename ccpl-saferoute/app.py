from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

from saferoute.env import SafeRouteEnv


def _run(mode, delay, seed, steps):
    env = SafeRouteEnv(delay=int(delay), seed=int(seed))
    state = env.reset()
    trace = []
    total_reward = 0.0
    for _ in range(int(steps)):
        if mode == "random":
            action = int(env.rng.integers(0, 5))
        else:
            distances = []
            for candidate_action, (dx, dy) in enumerate(env.ACTIONS):
                candidate = (env.position[0] + dx, env.position[1] + dy)
                if not env._inside(candidate) or candidate in env.hazards:
                    distances.append(float("inf"))
                else:
                    distances.append(env._distance(candidate, env.goal))
            action = int(np.argmin(distances))
        state, reward, cost, done, info = env.step(action)
        total_reward += reward
        trace.append((env.position, reward, cost))
        if done:
            break

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].set_title("SafeRoute")
    axes[0].set_xlim(-0.5, env.size - 0.5)
    axes[0].set_ylim(-0.5, env.size - 0.5)
    axes[0].set_aspect("equal")
    axes[0].set_xticks(range(env.size))
    axes[0].set_yticks(range(env.size))
    axes[0].grid(alpha=0.25)
    hazard_x, hazard_y = zip(*env.hazards)
    axes[0].scatter(hazard_x, hazard_y, marker="s", s=80, color="#F97316")
    positions = np.asarray([item[0] for item in trace], dtype=int)
    if len(positions):
        axes[0].plot(positions[:, 0], positions[:, 1], color="#2563EB", linewidth=2)
    axes[0].scatter(1, 1, color="#16A34A", s=80, label="start")
    axes[0].scatter(*env.goal, color="#7C3AED", s=80, label="goal")
    axes[0].legend(fontsize=8)
    axes[1].set_title("Delayed cost")
    axes[1].bar(range(len(trace)), [item[2] for item in trace], color="#DC2626")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("emitted cost")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    image_path = Path(tempfile.gettempdir()) / "saferoute.png"
    figure.savefig(image_path, dpi=130)
    plt.close(figure)
    stats = env.episode_stats()
    summary = (
        f"mode={mode}\n"
        f"steps={stats['steps']}\n"
        f"reward={total_reward:+.3f}\n"
        f"cost={stats['total_consequence']:.3f}\n"
        f"constraint_satisfied={stats['total_consequence'] <= 3.0}\n"
        f"route_complete={stats['route_complete']}\n"
        f"delay={delay}"
    )
    return str(image_path), summary


with gr.Blocks(title="CCPL SafeRoute") as demo:
    gr.Markdown(
        "# CCPL SafeRoute\n"
        "A delayed-consequence warehouse navigation demo powered by CCPL."
    )
    with gr.Row():
        mode = gr.Radio(["greedy", "random"], value="greedy", label="Policy")
        delay = gr.Slider(0, 8, value=3, step=1, label="Consequence delay")
        seed = gr.Number(value=42, precision=0, label="Seed")
        steps = gr.Slider(10, 120, value=80, step=1, label="Maximum steps")
    run = gr.Button("Run episode", variant="primary")
    image = gr.Image(label="Route and delayed cost", type="filepath")
    summary = gr.Textbox(label="Episode summary", lines=8)
    run.click(_run, [mode, delay, seed, steps], [image, summary])


if __name__ == "__main__":
    demo.launch()
