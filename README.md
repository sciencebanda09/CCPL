# CCPL

## Causal Consequence-Penalized Learning

CCPL is a research implementation for constrained reinforcement learning when
constraint consequences can be delayed. Standard constrained RL can attribute a
delayed consequence to the wrong current action, confuse correlation with
causal contribution, and introduce Bellman-target non-stationarity when the
penalty multiplier changes.

The implementation combines four components:

1. A delay-corrected Bellman operator for stochastic consequence delays.
2. A state-conditioned multiplier `lambda(s)` for the constrained policy.
3. An interventional Consequence Net for action-level causal attribution when
   interventional labels are available from the controlled SCM.
4. Separate reward and constraint Q-functions so multiplier changes do not
   alter either critic's TD target.

This is research software. The causal attribution and state-conditioned
multiplier claims are conditional on the assumptions documented in
[`docs/MATHEMATICAL_SPEC.md`](docs/MATHEMATICAL_SPEC.md); they are not claims
of universal superiority or automatic causal identification in arbitrary
environments.

## Results integrity

**Paper-reported benchmark results.** Results reported in the associated paper
belong to the paper's stated experimental protocol and should be cited from
that paper.

**Current repository results.** The repository contains code, configurations,
tests, and result-generation scripts. Benchmark outputs are generated locally
and are not represented here as paper results. Use the exact configuration,
seed list, environment versions, and command recorded in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

Quick or reduced runs, including short Safety Gymnasium-style runs, are
preliminary checks and must not be presented as the full paper benchmark.

## Installation

From a clean clone:

```bash
python -m pip install -e ".[dev]"
```

The core implementation requires Python 3.10+, NumPy, SciPy, and Matplotlib.
Optional Safety Gymnasium dependencies are available with:

```bash
python -m pip install -e ".[dev,safety]"
```

## Quick start

Install the package and use the public API from another project:

```bash
python -m pip install ccpl-rl
```

```python
from ccpl import make_ccpl, make_env, run_episode

agent = make_ccpl(state_dim=6, action_dim=5, seed=42)
env = make_env("standard", seed=42)
result = run_episode(agent, env, train=False)
print(result)
```

The distribution name on PyPI is `ccpl-rl`; the Python import namespace remains
`ccpl`. Run the repository smoke configuration separately:

```bash
python run_ccpl.py theory
python run_benchmark_v7.py --episodes 200 --eval-eps 30
```

The canonical full synthetic benchmark is configured in
[`configs/main_v7.yaml`](configs/main_v7.yaml). Generated outputs should be
archived with the command and environment metadata when used for a research
claim.

## Reproduction

- [Reproducibility guide](REPRODUCIBILITY.md)
- [Data specification](DATA_SPEC.md)
- [Research protocol](docs/RESEARCH_PROTOCOL.md)
- [Mathematical specification](docs/MATHEMATICAL_SPEC.md)
- [Data and artifact policy](docs/DATA_AND_ARTIFACTS.md)
- [Primary configuration](configs/main_v7.yaml)

The contraction result requires its stated assumptions, including a positive
minimum delay condition. Unknown stochastic delays do not by themselves imply
contraction. Likewise, the state-conditioned multiplier result is conditional
on the assumptions in the mathematical specification and is not a universal
dominance claim.

## Repository structure

```text
ccpl/                 Installable Python package
  algorithms/         CCPL, baselines, networks, theory utilities
  environments/       Synthetic and Safety Gymnasium-style environments
configs/               Versioned experiment configurations
docs/                  Research protocol and artifact documentation
scripts/               Convenience entry points
tests/                 Numerical, theoretical, and regression tests
run_ccpl.py            Legacy-compatible experiment entry point
run_benchmark_v7.py   Benchmark runner
```

## Tests

```bash
python -m pytest -q
```

The current verified suite contains 37 tests: 37 passed, 0 failed, and 0
skipped.

## Citation

If you use CCPL in academic or research work, please cite the associated
paper. Citation metadata is maintained in [`CITATION.cff`](CITATION.cff).
Bibliographic details that are not present in this repository are intentionally
not invented here.

## License

CCPL is source-available research software with separate commercial
licensing. Academic, educational, and non-commercial research users may
inspect, run, modify, and use the software for experiments and publication,
provided they retain the copyright and license notices and cite the associated
paper. Production deployment, commercial products or services, SaaS/API
offerings, proprietary integrations, paid services based substantially on
CCPL, and commercial redistribution require a separate commercial license.
See [`LICENSE`](LICENSE). This is not an OSI-approved open-source license.

## GitHub metadata

Suggested repository description:

> Causal RL for safety constraints under delayed consequences, combining delay-corrected Bellman targets, causal consequence attribution, state-conditioned Lagrange multipliers, and dual Q-functions.

Suggested topics: `reinforcement-learning`, `safe-reinforcement-learning`,
`constrained-reinforcement-learning`, `causal-reinforcement-learning`,
`causal-inference`, `deep-reinforcement-learning`, `machine-learning`,
`artificial-intelligence`, `neurips`, `research`.
