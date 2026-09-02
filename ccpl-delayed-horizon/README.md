# District 04: Autonomous Freight Corridor

District 04 is a browser-based 3D research simulation for inspecting CCPL in a
compact logistics world. The vehicle moves between a distribution hub,
warehouse district, charging infrastructure, an elevated freight bridge, a
HazMat transfer area, and a delivery terminal.

The important object is not a static hazard. Actions change latent state. For
example, repeated aggressive movement near loaded cargo reduces
`cargo_stability`; if the threshold is crossed, a containment event is created,
scheduled, emitted later, and linked back to its source and contributing turns.
The same ledger tracks bridge stress, HazMat exposure, congestion, and their
delayed consequences. Ground-truth provenance is exported for validation but
is hidden from the normal replay view.

This is a controlled research simulation, not a production logistics
controller or a real-world safety certificate. It preserves the existing CCPL
checkpoint contract: 12 observations and five discrete actions. The richer
latent state and event ledger are part of the environment and audit export.

## Train and generate a deterministic rollout

The older SafeRoute checkpoint can be loaded for a compatibility smoke test,
but it was not trained on District 04. For meaningful District 04 behavior,
train a checkpoint with the matching environment:

```powershell
py -3.11 ccpl-delayed-horizon\train_district.py `
  --episodes 1000 `
  --seed 42 `
  --delay 9 `
  --output ccpl-delayed-horizon\artifacts\district04_seed42.pkl
```

From the repository root in PowerShell:

```powershell
py -3.11 ccpl-delayed-horizon\generate.py `
  --checkpoint ccpl-delayed-horizon\artifacts\district04_seed42.pkl `
  --seeds 42,46,146,222,555,777 `
  --delay 9 `
  --output ccpl-delayed-horizon\data\rollouts.json
```

The same seed, scenario, checkpoint, and delay reproduce the same world and
trajectory. The JSON includes transitions, latent state snapshots, pending
events, emitted event IDs, and simulator ground-truth provenance. A policy
estimate can be added separately in the `ccpl_attribution` field; it is not
filled with simulator truth.

## Open the 3D replay

```powershell
cd ccpl-delayed-horizon
py -3.11 -m http.server 8765
```

Open <http://127.0.0.1:8765>. The scene contains roads, warehouses, loading
bays, charging, an elevated bridge, terminal markers, and the replay vehicle.
Use the run selector, pause/restart controls, and follow/free camera. When an
event exists, select it in the lower-left panel or press `V` to open the causal
replay inspector. The inspector shows the emission turn, source turn,
contributors, delay, cost, and latent changes; the red trace connects the
recorded contributor locations.

## Files

- `world.py`: District 04 dynamics, latent state, and transition export.
- `events.py`: deterministic delayed-event ledger and provenance records.
- `generate.py`: checkpoint rollout exporter.
- `index.html`: standalone Three.js visualization and causal inspector.
- `data/rollouts.json`: generated demo record.

Quantitative claims should come from saved benchmark tables and matched policy
evaluations. The animation communicates the event lifecycle; it is not itself a
performance result.

## Delayed Safety-Gymnasium benchmark

The reusable wrapper is available from the main package:

```python
from ccpl import make_delayed_safety_env

env = make_delayed_safety_env(
    "SafetyPointGoal1", mode="stochastic", delay_low=1, delay_high=10, seed=42
)
```

It preserves the raw Safety-Gymnasium cost as `info["raw_cost"]`, returns the
delayed cost to the policy, and stores simulator ground truth in
`info["emitted_events"]` and `episode_stats()["events"]`. Supported modes are
`immediate`, `fixed`, `stochastic`, and `distribution`.

Run the CCPL benchmark exporter after installing the optional Safety-Gymnasium
dependencies:

```powershell
python scripts\run_delayed_safety.py `
  --tasks SafetyPointGoal1,SafetyPointGoal2 `
  --mode fixed --delay 5 --episodes 10 --seeds 42,43,44 `
  --output ccpl-delayed-horizon\data\delayed_safety.json
```

Open `safety-gym.html` from the same HTTP server to inspect the exported
timeline. Its environment panel is a state-space projection from the actual
recorded observations; it does not invent hazard geometry. Source markers are
ground truth from the wrapper, while model attribution remains empty unless a
policy explicitly exports it.
