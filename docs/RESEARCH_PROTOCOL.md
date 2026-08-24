# CCPL Research Protocol

## Research Question

The central question is whether explicit delayed-consequence modeling and
causal action attribution improve constrained learning relative to matched
baselines while maintaining the specified constraint budget.

## Evidence Requirements

| Claim | Required evidence |
| --- | --- |
| Delayed feedback is aligned correctly | Alignment tests and delay diagnostics |
| Causal attribution improves learning | Held-out SCM labels and matched no-causal runs |
| A state-conditioned multiplier is useful | Paired seed-level comparison with a scalar-multiplier ablation |
| The method transfers | Environments held out from tuning, with versions recorded |

An observed reward increase is not by itself evidence of a safety improvement.
Every result must report reward and consequence metrics together.

## Experiment Tiers

1. `smoke`: imports, short training, and finite-value checks.
2. `main_v7`: primary synthetic benchmark and ablations.
3. `adversarial`: held-out stress tests; do not tune on these environments.
4. `safety`: optional external-environment evaluation with package and task versions.
5. `theory`: contraction, multiplier, and synthetic causal-label diagnostics.
6. `E10`: SCM-quality robustness under noisy, misspecified, and observational-only attribution.

The repository includes `CPO-FO`, a first-order approximation of constrained
policy optimization. It must not be reported as an exact conjugate-gradient
or natural-gradient CPO implementation. See
[`CPO_COMPARISON.md`](CPO_COMPARISON.md).

## Required Record

For every reported result, preserve:

- the exact command and configuration;
- all random seeds and per-seed metrics;
- SCM-quality mode and whether causal labels were available;
- Python, package, MuJoCo, and environment versions;
- the source revision or archive hash;
- training and evaluation episode counts;
- definitions of reward, cost, discounting, and constraint satisfaction;
- training failures, safety trips, and excluded runs.

Means and standard deviations must be computed across independent training
seeds. Episodes from one run are not independent experimental replicates.

## Interpretation Boundary

The contraction theorem is conditional on its assumptions, including the
positive minimum delay condition. State-conditioned multipliers are not
universally superior to scalar multipliers. External Safety Gymnasium results
test the implementation on those tasks; they do not validate synthetic SCM
labels or provide a deployment certification.
