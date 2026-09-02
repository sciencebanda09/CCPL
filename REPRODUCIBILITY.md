# Reproducibility guide

Reproducibility is part of CCPL’s safety contribution. A delayed-consequence
result is only useful if another researcher can determine which action was
credited, which delay was used, and whether the reported policy was evaluated
on held-out episodes. This guide defines the minimum experiment record.

## Environment

Use Python 3.10 or newer and install the pinned project requirements:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Safety Gymnasium dependencies are optional and should be reported separately
from the synthetic benchmark.

## Verification

Run the correctness suite before any experiment:

```bash
python -m pytest -q
```

The suite checks numerical primitives, delayed-cost alignment, causal labels,
environment accounting, and short finite training runs. Passing tests establish
that the implementation behaves as specified; they do not establish that a
policy is safe in an unmodeled environment.

## Experiment record

For every reported result, preserve:

- the exact command;
- the config file and its contents;
- all random seeds;
- Python and dependency versions;
- git commit or source archive identifier;
- training and evaluation episode counts;
- per-seed metrics, not only means;
- environment and task versions.

For MATLAB/Simulink robotics runs, also preserve:

- MATLAB release and toolbox versions;
- robot model source and mesh/support-package status;
- generated `.slx` model or model-generation command;
- simulation stop/fall criteria and contact parameters;
- whether the result is kinematic, physics-simulated, or hardware-validated.

Use `configs/smoke.yaml` for a quick end-to-end check and
`configs/main_v7.yaml` for the primary synthetic benchmark. Do not alter a
config silently after results are generated; create a new experiment version.

## Interpretation

Report reward and constraint cost together. State whether costs are observed,
delayed, or flushed at episode termination, and distinguish held-out transfer
evaluation from environments used for tuning.

Use strong claims only at the level supported by the artifact. A smoke test
proves execution and interface compatibility. A benchmark proves the recorded
comparison under its stated protocol. A physics simulation tests behavior under
its modeled dynamics. None of these alone proves safety on a physical robot;
that requires separate hardware validation and an approved deployment process.
