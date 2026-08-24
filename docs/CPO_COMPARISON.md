# CCPL and CPO-FO Comparison

## Purpose

This comparison tests whether CCPL's delayed-consequence correction and causal
attribution add value beyond a constrained policy-optimization baseline. It is
a research comparison, not a certification of either algorithm.

## Baseline Definition

The repository baseline is named `CPO-FO` because it is a first-order
approximation. It is not an exact reproduction of the conjugate-gradient and
natural-gradient CPO implementation. Keep the `CPO-FO` label in tables and
figures.

The original CPO work uses local trust-region policy updates and approximate
constraint enforcement. See the [BAIR overview](https://bair.berkeley.edu/blog/2017/07/06/cpo/)
and the [reference implementation](https://github.com/jachiam/cpo).

## Controlled Protocol

Use the same values for CCPL and CPO-FO wherever the algorithms support them:

- state and action spaces;
- environment class and consequence-delay settings;
- constraint budget;
- training and evaluation episode counts;
- random seed list;
- held-out environments and evaluation procedure;
- hardware and dependency versions.

Both methods receive the consequence stream emitted by the environment. Do not
give CPO-FO an undelayed cost stream unless that is a separate named
experiment.

Synthetic comparison:

```bash
python run_benchmark_v7.py --episodes 1000 --eval-eps 100 --seeds 3
```

Smoke comparison:

```bash
python run_benchmark_v7.py --episodes 200 --eval-eps 30 --seeds 1
```

Modern Safety Gymnasium comparison:

```bash
python ccpl_experiments.py --exp E8 \
  --tasks SafetyPointGoal1 \
  --episodes 500 --eval-episodes 100 --seeds 3 \
  --out results_mujoco
```

Use the official task name. `SafetyPointGoal1` is not the historical CPO
`PointGather` task.

## Metrics

Report per-seed values and mean $\pm$ standard deviation for:

- episode reward;
- discounted and undiscounted consequence;
- constraint satisfaction rate;
- delayed-hit frequency;
- inference latency;
- training failures and safety trips.

Use paired seed-level comparisons. Episodes from one run are not independent
replicates.

## Interpretation

An improvement can be attributed only to the tested protocol. Use
`CCPL-NoDelay`, `CCPL-NoCausal`, `CCPL-NoStateLambda`, and `CCPL-SingleQ` to
study individual components. Neither CCPL nor CPO-FO provides a formal
deployment safety certificate; application-level action validation remains
necessary for safety-sensitive use.
