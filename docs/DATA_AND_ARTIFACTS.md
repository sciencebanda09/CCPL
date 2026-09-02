# Data and Artifacts

CCPL treats provenance as a first-class artifact. Delayed safety experiments
are difficult to audit after the fact because the visible violation may occur
many steps after its source action. The files below preserve enough context to
reconstruct what was measured and what was only estimated.

## Source data

The default experiments generate synthetic transitions at runtime. Safety
Gymnasium experiments use the installed external environments and do not add
their assets to this repository.

## Generated outputs

Store outputs under a named directory such as `results_v7/` or
`results_mujoco/`. A result directory should contain, where applicable:

- the exact command;
- the configuration file;
- source revision;
- dependency and environment versions;
- random seed list;
- per-seed summary data;
- aggregate metrics and figures.

The directory name alone is not metadata. A checked-in figure is not
authoritative unless its producing command and configuration are recorded.
Generated caches, logs, checkpoints, and local plots are ignored unless they
are intentionally curated as research artifacts.

## What the artifacts establish

- Test reports establish implementation correctness for covered numerical,
  causal-label, and delay-accounting cases.
- Benchmark results establish performance only for the named environment,
  configuration, seeds, and evaluation protocol.
- Offline audit reports establish data integrity, support coverage, and the
  limits of what can be estimated from logged trajectories.
- MATLAB robotics outputs establish behavior of the stated simulated robot
  model and controller configuration. A Simscape model must include documented
  contact, actuation, sensing, and termination interfaces before it can be
  described as a physics-validated walking experiment.

This distinction makes the repository useful for research review and system
design: readers can see what was solved, what evidence supports it, and which
engineering validation step remains before deployment.
