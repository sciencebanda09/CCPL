from __future__ import annotations

import math
from pathlib import Path
import sys

from collections import deque
import numpy as np

from events import EventLedger


class District04Env:
    """Deterministic District 04 logistics world using the CCPL 12/5 contract."""

    ACTIONS = ("forward", "turn_left", "turn_right", "slow", "stop")
    SCENARIO = "north_freight_corridor"

    def __init__(self, max_steps: int = 180, delay: int = 9, seed: int = 42):
        if max_steps < 1 or delay < 0:
            raise ValueError("max_steps and delay must be non-negative")
        self.max_steps = int(max_steps)
        self.consequence_delay = int(delay)
        self.seed = int(seed)
        self.action_space = type("Discrete", (), {"n": 5})()
        self.observation_space = type("Box", (), {"shape": (12,)})()
        self.reset()

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        offset = float(self.seed % 5 - 2)
        self.size = 100.0
        self.position = np.array([-42.0, -34.0], dtype=np.float64)
        self.start = self.position.copy()
        self.heading = 0.70
        self.speed = 0.0
        self.energy = 1.0
        self.cargo_load = 1.0
        self.cargo_stability = 1.0
        self.bridge_stress = 0.0
        self.hazard_exposure = 0.0
        self.congestion = 0.0
        self.step_count = 0
        self.total_cost = 0.0
        self.delayed_hits = 0
        self.done = False
        self.last_event = "dispatch"
        self.route_complete = False
        self.trace = []
        self.ledger = EventLedger(self.consequence_delay)
        self.pending_costs = deque([(0.0, None, None)] * self.consequence_delay,
                                   maxlen=max(1, self.consequence_delay))
        self.locations = {
            "hub": [-42.0, -34.0], "warehouse": [-15.0, -9.0],
            "charging": [6.0, -25.0], "bridge": [12.0, 4.0],
            "hazmat": [25.0, 17.0], "terminal": [39.0, 35.0],
        }
        self.warehouses = [[-27.0, -16.0], [-13.0, -4.0], [4.0, 7.0], [22.0, 27.0]]
        self.roads = [
            [[-45.0, -34.0], [-16.0, -12.0], [12.0, 4.0], [40.0, 35.0]],
            [[-16.0, -12.0], [-8.0, 19.0], [12.0, 4.0], [25.0, 17.0]],
        ]
        self.scenario_config = {"district": "04", "name": self.SCENARIO,
                                "seed_offset": offset, "delay": self.consequence_delay}
        return self._observation()

    def _distance(self, point):
        return float(np.linalg.norm(self.position - np.asarray(point, dtype=np.float64)))

    def _near(self, name, radius):
        return self._distance(self.locations[name]) <= radius

    def _road_alignment(self):
        return float(np.clip(1.0 - min(self._distance(self.locations["bridge"]) / 28.0, 1.0), 0.0, 1.0))

    def _progress(self):
        total = np.linalg.norm(np.asarray(self.locations["terminal"]) - self.start)
        left = self._distance(self.locations["terminal"])
        return float(np.clip(1.0 - left / total, 0.0, 1.0))

    def _observation(self):
        terminal = np.asarray(self.locations["terminal"])
        delta = terminal - self.position
        wind_angle = 0.3 - self.heading
        queued = float(self.pending_costs[-1][0]) if self.consequence_delay else 0.0
        return np.asarray([
            np.clip((self.position[0] + 50) / 100, 0, 1),
            np.clip((self.position[1] + 50) / 100, 0, 1),
            np.clip(delta[0] / 100, -1, 1), np.clip(delta[1] / 100, -1, 1),
            max(0.0, 1.0 - self.step_count / self.max_steps),
            self.cargo_stability, self.bridge_stress, self.hazard_exposure,
            self.energy, np.clip(self._distance(terminal) / 120, 0, 1),
            queued, self.congestion,
        ], dtype=np.float32)

    def _add_latent_event(self, event_type, contributors, cost, changes):
        return self.ledger.create(event_type, self.step_count, contributors, cost, changes)

    def step(self, action: int):
        if self.done:
            raise RuntimeError("reset must be called before step")
        action = int(action)
        if not 0 <= action < len(self.ACTIONS):
            raise ValueError("action outside the discrete action space")
        old_distance = self._distance(self.locations["terminal"])
        if action == 1:
            self.heading -= 0.22
        elif action == 2:
            self.heading += 0.22
        elif action == 0:
            self.speed = min(1.55, self.speed + 0.18)
        elif action == 3:
            self.speed *= 0.55
        else:
            self.speed *= 0.15
        turn_stress = 0.0 if action in (0, 3, 4) else 0.05 * max(self.speed, 0.18)
        self.speed = max(self.speed, 0.18 if action in (1, 2) else 0.0)
        movement = np.array([math.cos(self.heading), math.sin(self.heading)]) * self.speed
        candidate = self.position + movement
        boundary = bool(np.any(np.abs(candidate) > 48.0))
        if boundary:
            candidate = np.clip(candidate, -48.0, 48.0)
            self.speed *= 0.2
        self.position = candidate
        near_bridge = self._near("bridge", 12.0)
        near_hazmat = self._near("hazmat", 11.0)
        near_warehouse = self._near("warehouse", 11.0) or self._near("hub", 8.0)
        if near_warehouse and action in (0, 1, 2):
            self.cargo_stability = max(0.0, self.cargo_stability - (0.018 + turn_stress))
        if near_bridge:
            self.bridge_stress = min(1.0, self.bridge_stress + 0.012 + 0.012 * (self.speed / 1.55))
        if near_hazmat:
            self.hazard_exposure = min(1.0, self.hazard_exposure + 0.025 + 0.02 * (self.speed / 1.55))
        self.congestion = float(np.clip(self.congestion * 0.96 + (0.035 if action == 4 and self._road_alignment() > .25 else 0), 0, 1))
        self.energy = float(np.clip(self.energy - .0025 - .002 * self.speed - .001 * near_hazmat, 0, 1))
        contributors = [self.step_count]
        if near_warehouse and self.cargo_stability < .88 and action in (1, 2):
            contributors += [max(0, self.step_count - 2)]
        if self.cargo_stability < .62 and not any(e.event_type == "cargo_containment_failure" for e in self.ledger.events):
            event = self._add_latent_event("cargo_containment_failure", contributors, 1.2,
                [{"field": "cargo_stability", "delta": -0.38}, {"field": "cargo_load", "delta": -0.2}])
            self.cargo_load = .8
        if self.bridge_stress > .72 and not any(e.event_type == "bridge_platform_failure" for e in self.ledger.events):
            self._add_latent_event("bridge_platform_failure", contributors, 1.0,
                                   [{"field": "bridge_stress", "delta": .28}])
        if self.hazard_exposure > .55 and not any(e.event_type == "hazmat_containment_breach" for e in self.ledger.events):
            self._add_latent_event("hazmat_containment_breach", contributors, 1.4,
                                   [{"field": "hazard_exposure", "delta": .45}])
        if self.congestion > .64 and not any(e.event_type == "route_cascade" for e in self.ledger.events):
            self._add_latent_event("route_cascade", contributors, .7,
                                   [{"field": "congestion", "delta": .25}])
        source_turn = self.step_count
        if self.consequence_delay:
            self.pending_costs.append((0.0, action, source_turn))
            emitted_cost, source_action, source_index = self.pending_costs.popleft()
        else:
            emitted_cost, source_action, source_index = 0.0, action, source_turn
        emitted_events = self.ledger.emit(self.step_count + 1)
        emitted_cost += sum(event.cost for event in emitted_events)
        self.step_count += 1
        if self.done and self.consequence_delay:
            self.pending_costs.clear()
            self.pending_costs.extend([(0.0, None, None)] * self.consequence_delay)
        new_distance = self._distance(self.locations["terminal"])
        reached = new_distance < 4.0
        self.route_complete = reached
        self.done = reached or self.step_count >= self.max_steps or self.energy <= 0
        if emitted_cost > 0:
            self.delayed_hits += len(emitted_events) or 1
        self.total_cost += emitted_cost
        self.last_event = emitted_events[-1].event_type if emitted_events else ("boundary correction" if boundary else "corridor clear")
        reward = .045 * (old_distance - new_distance) + .018 * (self._progress()) - .003 * self.speed
        if reached:
            reward += 3.0
        emitted_ids = [event.event_id for event in emitted_events]
        event_refs = [event.to_dict() for event in emitted_events]
        transition = {
            "episode_id": f"district04-seed-{self.seed}", "seed": self.seed,
            "scenario": self.SCENARIO, "turn": self.step_count - 1,
            "state": self._observation().tolist(), "action": action,
            "action_name": self.ACTIONS[action], "reward": float(reward),
            "immediate_cost": 0.0, "delayed_cost_emitted": float(emitted_cost),
            "energy": self.energy, "route_progress": self._progress(),
            "latent_state": {"cargo_stability": self.cargo_stability, "bridge_stress": self.bridge_stress,
                             "hazard_exposure": self.hazard_exposure, "congestion": self.congestion},
            "pending_delayed_events": self.ledger.pending(), "emitted_event_ids": emitted_ids,
            "ground_truth_events": event_refs, "ccpl_attribution": None,
        }
        self.trace.append({
            "position": self.position.tolist(), "action": action, "action_name": self.ACTIONS[action],
            "reward": float(reward), "cost": float(emitted_cost), "immediate_cost": 0.0,
            "event": self.last_event, "heading": self.heading, "speed": self.speed,
            "energy": self.energy, "cargo_stability": self.cargo_stability, "bridge_stress": self.bridge_stress,
            "hazard_exposure": self.hazard_exposure, "congestion": self.congestion,
            "route_progress": self._progress(), "actual_tau": None if not emitted_events else self.consequence_delay,
            "emitted_events": event_refs, "transition": transition,
        })
        return self._observation(), float(reward), float(emitted_cost), self.done, {
            "cost": float(emitted_cost), "immediate_cost": 0.0, "event": self.last_event,
            "position": self.position.tolist(), "route_complete": reached, "delay": self.consequence_delay,
            "emitted_events": event_refs, "ledger": self.ledger.all(), "route_progress": self._progress(),
        }

    def episode_stats(self):
        return {"scenario": self.SCENARIO, "size": self.size, "max_steps": self.max_steps,
                "delay": self.consequence_delay, "steps": self.step_count,
                "delayed_hits": self.delayed_hits, "total_consequence": self.total_cost,
                "route_complete": self.route_complete, "trace": self.trace,
                "events": self.ledger.all(), "start": self.start.tolist(),
                "locations": self.locations, "warehouses": self.warehouses, "roads": self.roads,
                "scenario_config": self.scenario_config}
