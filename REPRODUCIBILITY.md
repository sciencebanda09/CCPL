# Reproducibility guide

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
environment accounting, and short finite training runs.

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

Use `configs/smoke.yaml` for a quick end-to-end check and
`configs/main_v7.yaml` for the primary synthetic benchmark. Do not alter a
config silently after results are generated; create a new experiment version.

## Interpretation

Report reward and constraint cost together. State whether costs are observed,
delayed, or flushed at episode termination, and distinguish held-out transfer
evaluation from environments used for tuning.
