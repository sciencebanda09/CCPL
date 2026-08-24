# Data specification

CCPL currently uses synthetic controlled Markov decision processes and
optional Safety Gymnasium-compatible environments. No external dataset is
required for the default benchmark.

## Transition record

Each environment step produces:

| Field | Type | Meaning |
| --- | --- | --- |
| `state` | `float32[state_dim]` | Observation before the action |
| `action` | `int` | Discrete controller action |
| `reward` | `float` | Reward emitted at this step |
| `next_state` | `float32[state_dim]` | Observation after the action |
| `consequence` | `float` | Cost emitted after the configured delay |
| `done` | `bool` | Episode termination flag |
| `info` | mapping | Delay, causal-label, and episode diagnostics |

The source consequence and emitted consequence are deliberately distinct.
Delayed feedback must be aligned back to the transition that caused it before
it is used as a supervised causal target.

## State conventions

The canonical synthetic environment uses six normalized state variables. Safety
Gymnasium adapters may use higher-dimensional observations and therefore do not
receive synthetic SCM labels. Action indices are zero-based and environment
specific; configurations must record `action_dim`.

## Artifact rules

Generated results belong under a named results directory and must include the
experiment configuration, seeds, environment names, code version, and metric
definitions. Large local data and model checkpoints are ignored by Git; do not
commit private or externally sourced data without authorization.
