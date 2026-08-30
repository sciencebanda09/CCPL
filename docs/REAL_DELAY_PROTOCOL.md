# CCPL Real-Delay Protocol

This protocol defines the first step for evaluating CCPL on logged data. It
does not turn an arbitrary log into a causal dataset and it does not perform
offline policy evaluation by itself.

## Required record

Store one JSON object per transition in a `.jsonl` file:

```json
{"episode_id":"case-001","timestep":0,"state":[0.1,0.2],"action":1,"reward":1.0,"consequence":0.0,"timestamp":1700000000.0,"done":false}
{"episode_id":"case-001","timestep":1,"state":[0.2,0.3],"action":0,"reward":0.0,"consequence":2.0,"timestamp":1700000001.0,"done":true,"delay":1,"causal_label":2.0}
```

Required fields are `episode_id`, `timestep`, `state`, `action`, `reward`,
`consequence`, `timestamp`, and `done`. Optional fields are:

- `delay`: observed consequence delay in transitions.
- `source_timestep`: explicit source-action index. When both fields are
  present, `source_timestep = timestep - delay` is required.
- `causal_label`: an externally justified action-level causal label. It must
  not be filled with an observational correlation and should be absent when
  no valid intervention or causal model is available.

## Audit

From the repository root:

```bash
python scripts/audit_logged_dataset.py data/trajectories.jsonl \
  --output results/real_delay_audit.json
```

The audit checks episode ordering, state dimensions, timestamps, terminal
records, delay ranges, and source-action alignment. Its report includes the
fraction of transitions with aligned consequences and causal labels.

## Evaluation stages

1. Validate the log and freeze the data split before training.
2. Run behavior-policy and support diagnostics; do not evaluate actions far
   outside the logged action distribution.
3. Compare CCPL with PPO-Lagrangian, SAC-Lagrangian, CPO, and a non-causal
   delay-aware ablation under identical splits and seeds.
4. Repeat with exact, noisy, misspecified, and absent causal labels.
5. Use an offline policy-evaluation estimator with bootstrap intervals before
   considering a shadow-mode test.

The current repository's synthetic SCM labels remain a controlled diagnostic.
They are not evidence that CCPL identifies causality from observational logs.
