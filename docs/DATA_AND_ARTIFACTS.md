# Data and Artifacts

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
