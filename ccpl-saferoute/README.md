---
title: CCPL SafeRoute
colorFrom: blue
colorTo: orange
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
---

# CCPL SafeRoute

CCPL SafeRoute is a small visual project that demonstrates CCPL on a concrete
control problem. A warehouse robot must reach a delivery point while avoiding
restricted cells. A hazard caused by an action is reported after a configurable
delay, so the cost arriving at the current step may belong to an earlier
decision.

## Install

From this directory:

```bash
python -m pip install -e ".[dev]"
```

The project uses the public `ccpl-rl` package. It does not import CCPL
repository internals.

## Hugging Face Space

The root of this project is ready to upload as a Gradio Space. Create a new
Space at [Hugging Face Spaces](https://huggingface.co/new-space), choose the
Gradio SDK, then upload `app.py`, `requirements.txt`, and the `saferoute/`
directory. The Space runs the interactive route demo without requiring a
checkpoint.

## Run

Train a CCPL agent:

```bash
saferoute train --episodes 300 --output results/ccpl.pkl
```

Evaluate the checkpoint:

```bash
saferoute evaluate --checkpoint results/ccpl.pkl \
  --episodes 20 --output results/evaluation.json
```

Generate the visual dashboard:

```bash
saferoute dashboard \
  --evaluation results/evaluation.json \
  --output results/dashboard.png
```

The dashboard shows the final route, hazard cells, cumulative reward, and the
cost stream emitted after the configured delay. The current version is a
demonstration environment, not a validated warehouse controller.

Each environment reset samples a new hazard layout while keeping the start and
goal cells clear. Evaluation therefore measures performance across layouts, not
repeated copies of one trajectory.

Compare reference policies. `reactive` uses only the observation; `oracle` is
included as an upper-bound diagnostic and has access to the full map:

```bash
saferoute compare --episodes 50 --output results/baselines.json
```

Report the evaluation mean, cost, CSR, completion rate, and maximum cost. A
single checkpoint evaluation is a preliminary generalization test; use
multiple training seeds for a research comparison.

Run a reproducible multi-seed sweep:

```bash
saferoute sweep --episodes 1000 --eval-episodes 100 \
  --seeds 42,43,44,45,46 --output results/sweep
```

The sweep writes one checkpoint per seed and `sweep_summary.json` with
per-seed values plus aggregate mean and sample standard deviation.

The `compare` command evaluates random, reactive, and oracle policies on the
same seeded layout sequence. The oracle has full-map access and is an upper
bound diagnostic, not a learned baseline.

Training prints both the latest episode and a 25-episode window. The latest
cost can be high when a delayed consequence is flushed at termination; the
window cost and window CSR are the better indicators of training trend. The
checkpoint JSON also records `lambda`, `lambda_target`, `lambda_scale`,
`jc_violation`, and `hit_freq_ema`. `lambda_scale` follows the CCPL warm-up
schedule and reaches its full value after 100 episodes. The target is driven
by the previous episode cost, local consequence estimates, and hit frequency,
so it is expected to move faster than the learned state-conditioned lambda.

Inspect the initial layout in the terminal:

```bash
saferoute render
```

Record one evaluation episode as an animated 3D GIF:

```bash
saferoute record3d --checkpoint results/ccpl.pkl \
  --seed 42 --output results/ccpl_saferoute_3d.gif
```

The recording uses the checkpoint policy and shows raised hazard cells, the
goal, the agent path, and the delayed cost at each step. GIF output requires
Matplotlib's Pillow support, which is included by the project dependencies.

Export the complete article-visualization suite:

```bash
saferoute record-suite --checkpoint results/ccpl.pkl \
  --seed 42 --output results/visual_suite
```

The suite writes top-down movement, a risk heatmap, a delayed-consequence
timeline, a policy comparison, training progress, a lambda dashboard, and a
rotating 3D risk-surface animation.

## Environment

The action space is discrete:

$$
\mathcal{A}=\{\text{up},\text{down},\text{left},\text{right},\text{wait}\}.
$$

The observation contains position, goal, remaining horizon, local hazard
signals, distance to goal, and the most recent delayed cost. The environment
returns the CCPL episode interface:

$$
(s_{t+1},r_t,c_t,\mathrm{done}_t,\mathrm{info}_t).
$$

`info["causal_delta"]` records the immediate action-level hazard contribution;
`info["cost"]` records the delayed cost emitted to the agent.

## Next extensions

- Add package-backed CPO-FO, PPO, and SAC-Lag comparisons.
- Add live rendering with pygame.
- Add multiple warehouse layouts and held-out layouts.
- Add a dashboard panel for the learned state-conditioned multiplier.
