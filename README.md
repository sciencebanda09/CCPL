# CCPL

## Causal Consequence-Penalized Learning

CCPL is a research implementation of constrained reinforcement learning for
environments in which an action can produce a safety consequence after a
delay. The package makes the delay explicit in the learning target and keeps
reward learning separate from constraint learning.

The method has four implementation components:

1. **Delay-corrected Bellman targets.** A delay model produces a state-
   dependent effective discount factor instead of assigning every observed
   cost to the most recent action.
2. **State-conditioned multipliers.** The constrained policy uses a multiplier
$\lambda(s)$ rather than one scalar penalty for all states.
3. **Interventional consequence attribution.** The Consequence Net estimates
   action-level contribution from interventional labels supplied by a
   controlled structural causal model (SCM).
4. **Separate critics.** Reward and constraint value functions have separate
parameters and TD targets, so changing $\lambda$ does not change either
   critic's target.

CCPL is research software. The theoretical statements are conditional: the
contraction result requires its stated assumptions, including a positive
minimum delay, and the state-conditioned multiplier result does not claim
universal dominance over a scalar multiplier. Synthetic SCM experiments
evaluate agreement with programmed structural equations; they do not identify
causality from observational data.

## Installation

The PyPI distribution is `ccpl-rl`; the Python import namespace is `ccpl`.

```bash
python -m pip install ccpl-rl
```

For development:

```bash
git clone https://github.com/sciencebanda09/ccpl.git
cd ccpl
python -m pip install -e ".[dev]"
python -m pytest -q
```

## Public API

CCPL currently supports discrete actions and NumPy observations. A minimal
Gymnasium integration is:

```python
import gymnasium as gym
from ccpl import GymnasiumCCPLEnv, make_ccpl

gym_env = gym.make("CartPole-v1")

wrapped_env = GymnasiumCCPLEnv(
    gym_env,
    consequence_key="cost",
    consequence_delay=2,
)
agent = make_ccpl(
    state_dim=gym_env.observation_space.shape[0],
    action_dim=gym_env.action_space.n,
    constraint_d=10.0,
    seed=42,
)

agent.fit(wrapped_env, episodes=1_000)
observation = wrapped_env.reset()
action = agent.predict(observation)
agent.save("checkpoints/ccpl.pkl")
```

The adapter reads the safety cost from `info["cost"]`. The environment must
expose a discrete action space. Pickle checkpoints must only be loaded from
trusted sources.

`SafetyPolicy` can add application-specific action validation, a fallback
action, a consequence budget, and JSONL audit logging. This is a runtime
control and audit mechanism; it is not a safety certificate.

## Mathematical specification

Let $p(\tau\mid h)$ be a delay distribution over
$\tau\in\{0,\ldots,K\}$, conditional on history $h$. CCPL uses

$$
\gamma_{\mathrm{eff}}(h)
  = \sum_{\tau=0}^{K} p(\tau\mid h)\gamma^{\tau}.
$$

Therefore $\gamma^K\leq\gamma_{\mathrm{eff}}(h)\leq 1$. A contraction
modulus strictly below one does not follow from an arbitrary unknown delay
distribution: an assumption excluding zero delay, or an equivalent bound, is
required. The complete scope and assumptions are in
[`docs/MATHEMATICAL_SPEC.md`](docs/MATHEMATICAL_SPEC.md).

## Experiments and results

The repository distinguishes three kinds of evidence:

- **Paper-reported benchmark results:** results from the protocol and version
  stated in the associated paper.
- **Repository reproductions:** outputs produced from a tagged source version,
  recorded configuration, dependency versions, and explicit seed list.
- **Preliminary experiments:** smoke tests or reduced runs, including short
  Safety Gymnasium evaluations. These verify execution and are not benchmark
  evidence.

The E8 experiment compares CCPL with CCPL-Base, CPO-FO, PPO, and SAC-Lag on
selected Safety Gymnasium tasks. It writes reward, constraint-satisfaction,
reward-versus-cost, and learning-curve figures:

```bash
python -m pip install -e ".[dev]"
python -m pip install "gymnasium==0.28.1" "mujoco==2.3.3" \
  "pygame>=2.6.1" "xmltodict" "pyyaml" "imageio"
python -m pip install --no-deps "safety-gymnasium==1.0.0"

export MUJOCO_GL=egl
export MPLBACKEND=Agg
python ccpl_experiments.py --exp E8 \
  --tasks SafetyPointGoal1 \
  --episodes 500 --eval-episodes 100 --seeds 3 \
  --out results_mujoco
```

`CPO-FO` is the repository's first-order constrained policy-optimization
baseline. It is not an exact conjugate-gradient or natural-gradient
implementation of CPO. See [`docs/CPO_COMPARISON.md`](docs/CPO_COMPARISON.md).

## Repository layout

```text
ccpl/                 Installable Python package
  algorithms/         CCPL, baselines, networks, and theory utilities
  environments/       Synthetic environments and Gymnasium adapters
configs/              Versioned experiment configurations
docs/                 Mathematical and research protocols
scripts/              Convenience entry points
tests/                Numerical, theoretical, and regression tests
ccpl_experiments.py   Synthetic and external-environment experiments
run_benchmark_v7.py   Primary synthetic benchmark runner
```

## Reproducibility

Read [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md), [`DATA_SPEC.md`](DATA_SPEC.md),
[`docs/RESEARCH_PROTOCOL.md`](docs/RESEARCH_PROTOCOL.md), and
[`docs/DATA_AND_ARTIFACTS.md`](docs/DATA_AND_ARTIFACTS.md). Every reported
result should include the exact command, configuration, source revision,
dependency versions, environment name, and individual seed values.

## Tests

```bash
python -m pytest -q
```

The last verified run contains **42 passed, 0 failed, and 0 skipped** tests.

## Citation

If you use CCPL in academic or research work, cite the associated paper.
Citation metadata is maintained in [`CITATION.cff`](CITATION.cff). Bibliographic
fields not present in this repository are intentionally not guessed.

## License

CCPL is source-available research software with separate commercial
licensing. Academic, educational, and non-commercial research users may
inspect, run, modify, and use it for experiments and publications, provided
they retain the copyright and license notices and cite the CCPL paper.
Production deployment, commercial products or services, SaaS/API offerings,
proprietary integrations, paid services based substantially on CCPL, and
commercial redistribution require a separate commercial license. See
[`LICENSE`](LICENSE). This is not an OSI-approved open-source license.

## GitHub metadata

Suggested description:

> Causal RL for safety constraints under delayed consequences, combining delay-corrected Bellman targets, causal consequence attribution, state-conditioned Lagrange multipliers, and dual Q-functions.

Suggested topics: `reinforcement-learning`, `safe-reinforcement-learning`,
`constrained-reinforcement-learning`, `causal-reinforcement-learning`,
`causal-inference`, `deep-reinforcement-learning`, `machine-learning`,
`artificial-intelligence`, `neurips`, `research`.
