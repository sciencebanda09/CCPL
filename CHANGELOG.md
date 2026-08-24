# Changelog

All notable changes to this research artifact are recorded here.

## [0.7.0] - 2026-08-21

### Added

- Research-oriented package layout under `ccpl/algorithms` and
  `ccpl/environments`.
- Install metadata through `setup.py` and `pyproject.toml`.
- Versioned experiment configurations and canonical script entry points.
- `LICENSE`, `CITATION.cff`, `.gitignore`, `CONTRIBUTING.md`,
  `DATA_SPEC.md`, and `REPRODUCIBILITY.md`.

### Fixed

- Prioritized causal replay now samples distinct transitions within a batch,
  preserving full-buffer delayed-cost alignment.

### Validation

- Regression suite: 26 tests passing.

## [Unreleased]

- Migrate remaining compatibility runners to package-level imports.
- Add automated result metadata capture for dependency versions and configs.
