# Contributing to CCPL

CCPL is source-available research software under the license in
[`LICENSE`](LICENSE). Academic and non-commercial research use is permitted
under that license. Contributions, redistribution, and derivative releases
must preserve the license and copyright notices.

## Before opening a change

1. Read [`LICENSE`](LICENSE) and [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).
2. Keep reusable implementation in `ccpl/` and command-line entry points in
   the existing root or `scripts/` runners.
3. Add regression tests for numerical or behavioral changes.
4. Do not commit datasets, checkpoints, caches, generated plots, or secrets.
5. Run `python -m pytest -q` from the repository root.

## Research changes

Document the hypothesis or bug, configuration, seed list, baseline, metrics,
and known limitations for every experiment-affecting change. Do not tune on
adversarial or transfer environments and report them as held out. Preserve
seed-level results rather than only aggregate means.

## Style

Use clear Python, explicit validation, deterministic seeded tests, and focused
commits. Keep public behavior compatible unless a breaking change is recorded
in `CHANGELOG.md`.
