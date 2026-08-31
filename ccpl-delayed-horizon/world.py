from __future__ import annotations

from collections import deque

import numpy as np


class FreightWorldEnv:
    ACTIONS = ((0, -1), (0, 1), (-1, 0), (1, 0), (0, 0))

    def __init__(self, size: int = 12, max_steps: int = 80, delay: int = 10,
                 seed: int = 42, hazard_count: int = 10):
        if size < 8 or max_steps < 1 or delay < 0:
            raise ValueError("size, max_steps, and delay are invalid")
        self.size = int(size)
        self.max_steps = int(max_steps)
        self.consequence_delay = int(delay)
        self.hazard_count = int(hazard_count)
        self.action_space = type("Discrete", (), {"n": len(self.ACTIONS)})()
        self.observation_space = type("Box", (), {"shape": (12,)})()
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.position = (1, 1)
        self.goal = (self.size - 2, self.size - 2)
        candidates = [(x, y) for x in range(1, self.size - 1)
                      for y in range(1, self.size - 1)
                      if (x, y) not in {self.position, self.goal}]
        chosen = self.rng.choice(len(candidates), self.hazard_count, replace=False)
        self.hazards = {candidates[int(index)] for index in chosen}
        self.step_count = 0
        self.total_cost = 0.0
        self.delayed_hits = 0
        self.done = False
        self.last_event = "start"
        self.pending = deque([(0.0, None, None)] * self.consequence_delay,
                             maxlen=max(1, self.consequence_delay))
        self.trace = []
        return self._observation()

    def _inside(self, position):
        return 0 <= position[0] < self.size and 0 <= position[1] < self.size

    def _distance(self, first, second):
        return abs(first[0] - second[0]) + abs(first[1] - second[1])

    def _observation(self):
        x, y = self.position
        gx, gy = self.goal
        adjacent = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]
        return np.asarray([
            x / (self.size - 1), y / (self.size - 1),
            gx / (self.size - 1), gy / (self.size - 1),
            max(0.0, 1.0 - self.step_count / self.max_steps),
            float(self.position in self.hazards),
            *[float(cell in self.hazards) for cell in adjacent],
            self._distance(self.position, self.goal) / (2 * (self.size - 1)),
            float(self.pending[-1][0]) if self.consequence_delay else 0.0,
        ], dtype=np.float32)

    def step(self, action: int):
        if self.done:
            raise RuntimeError("reset must be called before step")
        action = int(action)
        if not 0 <= action < len(self.ACTIONS):
            raise ValueError("action outside the discrete action space")
        dx, dy = self.ACTIONS[action]
        candidate = (self.position[0] + dx, self.position[1] + dy)
        old_distance = self._distance(self.position, self.goal)
        immediate_cost = 0.0
        event = "clear"
        if not self._inside(candidate):
            immediate_cost, event = 0.4, "boundary"
        elif candidate in self.hazards:
            immediate_cost, event = 1.0, "loading_zone_hazard"
        else:
            self.position = candidate
        new_distance = self._distance(self.position, self.goal)
        source_step = self.step_count
        emitted_cost, source_action, source_index = (0.0, None, None)
        if self.consequence_delay:
            self.pending.append((immediate_cost, action, source_step))
            emitted_cost, source_action, source_index = self.pending.popleft()
        else:
            emitted_cost, source_action, source_index = immediate_cost, action, source_step
        self.step_count += 1
        reached = self.position == self.goal
        self.done = reached or self.step_count >= self.max_steps
        if self.done and self.consequence_delay:
            emitted_cost += sum(item[0] for item in self.pending)
            self.pending.clear()
            self.pending.extend([(0.0, None, None)] * self.consequence_delay)
        if emitted_cost > 0:
            self.delayed_hits += 1
        self.total_cost += emitted_cost
        self.last_event = event if immediate_cost else ("delivery" if reached else "clear")
        reward = 0.045 * (old_distance - new_distance) - 0.012
        if reached:
            reward += 2.4
        info = {
            "cost": float(emitted_cost),
            "immediate_cost": float(immediate_cost),
            "causal_delta": float(immediate_cost),
            "source_action": source_action,
            "source_timestep": source_index,
            "actual_tau": None if source_index is None else self.step_count - 1 - source_index,
            "delayed_hits": self.delayed_hits,
            "event": self.last_event,
            "position": self.position,
            "goal": self.goal,
            "delay": self.consequence_delay,
            "hazards": sorted(self.hazards),
            "budget": 3.0,
        }
        self.trace.append({
            "position": self.position,
            "action": action,
            "reward": float(reward),
            "cost": float(emitted_cost),
            "immediate_cost": float(immediate_cost),
            "event": self.last_event,
        })
        return self._observation(), float(reward), float(emitted_cost), self.done, info

    def episode_stats(self):
        return {
            "size": self.size,
            "max_steps": self.max_steps,
            "delay": self.consequence_delay,
            "hazards": sorted(self.hazards),
            "steps": self.step_count,
            "delayed_hits": self.delayed_hits,
            "total_consequence": self.total_cost,
            "route_complete": self.position == self.goal,
            "trace": self.trace,
        }
