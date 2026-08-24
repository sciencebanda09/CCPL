from __future__ import annotations

from collections import deque

import numpy as np


class SafeRouteEnv:
    """Discrete warehouse navigation with delayed safety consequences."""

    ACTIONS = ((0, -1), (0, 1), (-1, 0), (1, 0), (0, 0))

    def __init__(self, size: int = 12, max_steps: int = 120, delay: int = 3,
                 seed: int = 42, hazard_count: int = 7):
        if size < 8 or max_steps < 1 or delay < 0:
            raise ValueError("size, max_steps, and delay are invalid")
        self.size = int(size)
        self.max_steps = int(max_steps)
        self.delay = int(delay)
        self.hazard_count = int(hazard_count)
        if self.hazard_count < 1 or self.hazard_count >= (self.size - 2) ** 2:
            raise ValueError("hazard_count is invalid")
        self.rng = np.random.default_rng(seed)
        self.action_space = type("Discrete", (), {"n": len(self.ACTIONS)})()
        self.observation_space = type("Box", (), {"shape": (12,)})()
        self.reset()

    def _build_hazards(self):
        candidates = [
            (x, y)
            for x in range(1, self.size - 1)
            for y in range(1, self.size - 1)
            if (x, y) not in {(1, 1), self.goal}
        ]
        selected = self.rng.choice(len(candidates), self.hazard_count, replace=False)
        return {candidates[int(index)] for index in selected}

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.position = (1, 1)
        self.goal = (self.size - 2, self.size - 2)
        self.hazards = self._build_hazards()
        self.step_count = 0
        self.total_cost = 0.0
        self.delayed_hits = 0
        self.delayed_queue = deque([0.0] * self.delay, maxlen=self.delay or 1)
        self.trace = []
        self.done = False
        self.last_event = "start"
        return self._observation()

    def _observation(self):
        x, y = self.position
        gx, gy = self.goal
        adjacent = [
            (x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)
        ]
        values = [
            x / (self.size - 1), y / (self.size - 1),
            gx / (self.size - 1), gy / (self.size - 1),
            max(0.0, 1.0 - self.step_count / self.max_steps),
            float(self.position in self.hazards),
            *[float(cell in self.hazards) for cell in adjacent],
            self._distance(self.position, self.goal) / (2 * (self.size - 1)),
            float(self.delayed_queue[-1]) if self.delay else 0.0,
        ]
        return np.asarray(values, dtype=np.float32)

    def _distance(self, first, second):
        return abs(first[0] - second[0]) + abs(first[1] - second[1])

    def _inside(self, position):
        x, y = position
        return 0 <= x < self.size and 0 <= y < self.size

    def _risk(self, position, action):
        if not self._inside(position):
            return 1.0, "boundary"
        if position in self.hazards:
            return 1.0, "hazard"
        if action == 4:
            return 0.0, "wait"
        return 0.0, "clear"

    def step(self, action: int):
        if self.done:
            raise RuntimeError("reset must be called before step")
        action = int(action)
        if not 0 <= action < len(self.ACTIONS):
            raise ValueError("action outside the discrete action space")

        dx, dy = self.ACTIONS[action]
        candidate = (self.position[0] + dx, self.position[1] + dy)
        risk, event = self._risk(candidate, action)
        old_distance = self._distance(self.position, self.goal)
        if self._inside(candidate):
            self.position = candidate
        new_distance = self._distance(self.position, self.goal)
        immediate_cost = risk + (0.15 if not self._inside(candidate) else 0.0)
        emitted_cost = self.delayed_queue.popleft() if self.delay else immediate_cost
        if self.delay:
            self.delayed_queue.append(immediate_cost)
        self.step_count += 1
        reached = self.position == self.goal
        self.done = reached or self.step_count >= self.max_steps
        if self.done and self.delay:
            emitted_cost += float(sum(self.delayed_queue))
            self.delayed_queue.clear()
            self.delayed_queue.extend([0.0] * self.delay)
        reward = 0.03 * (old_distance - new_distance) - 0.01
        if reached:
            reward += 2.0
        self.total_cost += emitted_cost
        self.delayed_hits += int(emitted_cost > 0.0)
        self.last_event = event if risk else ("goal" if reached else "clear")
        info = {
            "cost": float(emitted_cost),
            "immediate_cost": float(immediate_cost),
            "causal_delta": float(risk),
            "source_action": action,
            "event": self.last_event,
            "position": self.position,
            "goal": self.goal,
            "delay": self.delay,
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
            "lambda_state": None,
        })
        return self._observation(), float(reward), float(emitted_cost), self.done, info

    def episode_stats(self):
        return {
            "size": self.size,
            "max_steps": self.max_steps,
            "delay": self.delay,
            "hazards": sorted(self.hazards),
            "steps": self.step_count,
            "delayed_hits": self.delayed_hits,
            "total_consequence": self.total_cost,
            "route_complete": self.position == self.goal,
            "trace": self.trace,
        }

    def render_text(self):
        cells = []
        for y in range(self.size - 1, -1, -1):
            row = []
            for x in range(self.size):
                cell = (x, y)
                if cell == self.position:
                    row.append("R")
                elif cell == self.goal:
                    row.append("G")
                elif cell in self.hazards:
                    row.append("X")
                else:
                    row.append(".")
            cells.append(" ".join(row))
        return "\n".join(cells)
