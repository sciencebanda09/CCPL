# Contributing to CCPL

This repository is proprietary and evaluation-only. Contributions, patches,
redistribution, and derivative works require written authorization from the
copyright holder.

## Before submitting a change

1. Read [LICENSE](LICENSE) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
2. Keep implementation changes inside `ccpl/` and entry points inside `scripts/`.
3. Add or update regression tests for behavioral or numerical changes.
4. Avoid committing datasets, checkpoints, caches, generated plots, or secrets.
5. Run `python -m pytest -q` from the repository root.

## Research changes

Every experiment-affecting change must document:

- the hypothesis or bug being addressed;
- the configuration and random seeds used;
- the baseline being compared against;
- the metric impact and any known limitations.

Do not tune on adversarial or transfer environments and then report them as
held-out evaluation. Preserve seed-level results rather than only aggregate
means.

## Style

Use clear Python, explicit validation, deterministic seeded tests, and focused
commits. Keep public behavior backward compatible unless the change is called
out in `CHANGELOG.md`.
