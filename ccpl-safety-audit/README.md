# CCPL Safety Audit

This project evaluates a trained CCPL checkpoint across multiple environments,
delay settings, and random seeds. It reports average performance and worst-case
constraint behavior.

The audit does not prove universal safety. It measures robustness under the
explicit test distribution and records failures that require further analysis.

## Run

From the repository root:

```powershell
py -3.11 -m pip install -e .
py -3.11 -m pip install -e ccpl-safety-audit
py -3.11 -m safety_audit audit `
  --checkpoint ccpl-saferoute\results\local_sweep_1000\ccpl_seed42.pkl `
  --episodes 100 `
  --seeds 42,43,44,45,46 `
  --delays 0,2,5,10 `
  --output results/safety_audit.json
```

The report contains per-environment, per-delay, and per-seed results together
with aggregate mean, standard deviation, minimum CSR, and maximum cost.

## Interpretation

Use the report to identify failure modes, not to claim universal safety. A
credible safety statement must specify the environments, delays, disturbances,
seed set, constraint threshold, and evaluation horizon.
