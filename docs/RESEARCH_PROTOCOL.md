# CCPL research protocol

## Research question

Can causal action attribution, explicit delay modeling, and state-conditioned
constraint multipliers improve reward while satisfying delayed safety constraints?

## Claims and evidence

| Claim | Required evidence |
| --- | --- |
| Delayed feedback is handled correctly | alignment tests and delay diagnostics |
| Causal attribution improves learning | ICN/SCM MAE and correlation on held-out states |
| State-conditioned lambda is useful | paired seed-level comparison against ablations |
| Improvements transfer | held-out adversarial and Safety Gymnasium results |

## Experiment tiers

1. `smoke`: imports, short training, and finite-value checks.
2. `main_v7`: primary synthetic benchmark and ablations.
3. `adversarial`: held-out stress tests; never tune on these environments.
4. `safety`: optional external-environment evaluation with versions reported.
5. `theory`: contraction, dominance, and causal-label diagnostics.

## Reproducibility checklist

- Record every random seed and the exact command.
- Keep training and evaluation environments separate.
- Report mean, standard deviation, and individual seed values.
- Do not infer independent seeds from episode-level observations.
- Save config and summary JSON beside generated results.
- Run `python -m pytest -q` before publishing a result.

## Implementation boundary

Reusable implementation lives in `ccpl/`. The root runners are retained
as compatibility entry points for published reproduction commands; new code
should import from `ccpl` after installation.
