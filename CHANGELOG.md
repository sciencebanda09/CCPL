# Changelog

This file records user-visible changes to the research artifact. It does not
replace the experiment protocol or the paper's version history.

## [0.7.6] - 2026-08-24

### Changed

- Clarified the public package interface and PyPI distribution name.
- Added the CPO-FO comparison protocol for synthetic and Safety Gymnasium runs.
- Added modern Safety Gymnasium task selection and result visualizations to
  the E8 experiment.
- Reworked research documentation, mathematical scope, and artifact rules.

### Validation

- Verified test suite: 42 passed, 0 failed, 0 skipped.
- Verified package build for the `ccpl-rl` distribution.

## [0.7.0] - 2026-08-21

### Added

- Research package layout under `ccpl/algorithms` and `ccpl/environments`.
- Install metadata through `pyproject.toml`.
- Versioned experiment configurations and canonical entry points.
- License, citation, data, and reproducibility metadata.

### Fixed

- Prioritized causal replay now samples distinct transitions within a batch,
  preserving delayed-cost alignment.
