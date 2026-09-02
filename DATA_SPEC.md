# Data Specification

The default benchmark generates data from synthetic controlled Markov decision
processes. No external dataset is required. Optional Safety Gymnasium runs
obtain observations and costs from the environment.

## Transition Record

At step (t), the logical transition is

$$
\mathcal{T}_t=(s_t,a_t,r_t,s_{t+1},c_t,\mathrm{done}_t,\mathrm{info}_t).
$$

| Field | Type | Meaning |
| --- | --- | --- |
| `state` | `float32[state_dim]` | Observation before the action |
| `action` | `int` | Zero-based discrete action |
| `reward` | `float` | Reward emitted by the environment |
| `next_state` | `float32[state_dim]` | Observation after the action |
| `consequence` | `float` | Cost aligned with the causal transition |
| `done` | `bool` | Episode termination flag |
| `info` | mapping | Delay, causal-label, and episode diagnostics |

The source consequence and emitted consequence are distinct. Delayed feedback
must be aligned with the transition that caused it before it is used as a
supervised causal target.

## State and Action Conventions

The canonical synthetic environment uses six normalized state variables. The
Safety Gymnasium adapter accepts higher-dimensional observations but does not
receive synthetic SCM labels. Action indices are zero-based and environment
specific; every configuration must record `action_dim`.

## Real Logged Data

`ccpl.real_data.SWaTWADI` accepts local SWaT/WADI-style CSV exports and maps
numeric sensor columns to `state`. Actuator columns are encoded as observed
action categories; original actuator values remain in metadata. An anomaly
label becomes an observed consequence label when no explicit consequence
column is configured. The adapter never creates a `source_timestep` from an
anomaly label. Source attribution is available only when the log explicitly
provides an intervention/source field.

Use time-ordered episode splits and retain input file hashes with every report.
These results are offline logged-data evidence, not proof of causal
identification or online safety.

## Artifact Requirements

Generated results belong under a named results directory. Store the command,
configuration, seed list, source revision, dependency versions, environment
identifiers, metric definitions, and timestamp beside the results. Large local
data, caches, and checkpoints are ignored by Git. Do not commit private or
externally licensed data without authorization.
