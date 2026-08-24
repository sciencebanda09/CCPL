# Data and artifacts

- `results_v7/` is the default location for generated V7 benchmark outputs;
  it is intentionally not part of the source distribution.
- `configs/` contains experiment inputs.
- `tests/` contains correctness and regression checks.
- `docs/` contains protocol and interpretation guidance.

Generated results should include the config, seed list, dependency versions,
and timestamp. Checked-in result files are not authoritative unless their
producing command and config are recorded.
