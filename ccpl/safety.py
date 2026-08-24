"""Runtime safety controls for deploying a CCPL policy.

This module provides enforcement and observability hooks. It is not a safety
certificate and does not replace a domain hazard analysis or independent
assessment.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Iterable, Optional


class SafetyTrip(RuntimeError):
    """Raised when the configured safety policy cannot produce a safe action."""


class SafetyPolicy:
    """Guard a discrete-action agent with application-defined safety rules."""

    def __init__(self, agent, action_is_safe: Callable[[object, int], bool],
                 action_dim: int, fallback_action: int = 0,
                 allowed_actions: Optional[Iterable[int]] = None,
                 consequence_budget: Optional[float] = None, audit_path=None,
                 fail_closed: bool = True):
        if not callable(action_is_safe):
            raise TypeError("action_is_safe must be callable")
        if int(action_dim) <= 0:
            raise ValueError("action_dim must be positive")
        if not 0 <= int(fallback_action) < int(action_dim):
            raise ValueError("fallback_action must be within the action space")
        if consequence_budget is not None and float(consequence_budget) < 0:
            raise ValueError("consequence_budget must be non-negative")
        actions = list(range(int(action_dim))) if allowed_actions is None else list(allowed_actions)
        if not actions or any(not 0 <= int(a) < int(action_dim) for a in actions):
            raise ValueError("allowed_actions must contain valid discrete actions")
        self.agent = agent
        self.action_is_safe = action_is_safe
        self.action_dim = int(action_dim)
        self.fallback_action = int(fallback_action)
        self.allowed_actions = tuple(dict.fromkeys(int(a) for a in actions))
        self.consequence_budget = None if consequence_budget is None else float(consequence_budget)
        self.fail_closed = bool(fail_closed)
        self.audit_path = None if audit_path is None else Path(audit_path)
        self.episode_consequence = 0.0
        self.tripped = False

    def reset_episode(self):
        self.episode_consequence = 0.0
        self.tripped = False

    def _audit(self, event: str, **fields):
        if self.audit_path is None:
            return
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"time": time.time(), "event": event, **fields}) + "\n")

    def predict(self, observation) -> int:
        if self.tripped:
            raise SafetyTrip("safety policy is tripped; reset_episode() is required")
        proposed = int(self.agent.predict(observation))
        candidates = (proposed,) + tuple(a for a in self.allowed_actions if a != proposed)
        for action in candidates:
            if self.action_is_safe(observation, action):
                self._audit("action", proposed=proposed, selected=action,
                            fallback=action != proposed)
                return action
        self._audit("trip", proposed=proposed)
        self.tripped = True
        if self.fail_closed:
            raise SafetyTrip("no safe action is available")
        return self.fallback_action

    act = predict

    def observe_consequence(self, consequence: float, done: bool = False):
        value = float(consequence)
        if value < 0:
            raise ValueError("consequence must be non-negative")
        self.episode_consequence += value
        self._audit("consequence", value=value, cumulative=self.episode_consequence)
        if self.consequence_budget is not None and self.episode_consequence > self.consequence_budget:
            self.tripped = True
            self._audit("budget_exceeded", budget=self.consequence_budget)
            if self.fail_closed:
                raise SafetyTrip("episode consequence budget exceeded")
        if done:
            self._audit("episode_end", cumulative=self.episode_consequence)
