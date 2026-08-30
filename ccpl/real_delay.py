"""Validation utilities for real logged delayed-consequence trajectories."""

from __future__ import annotations

import json
import math
import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class LoggedTransition:
    episode_id: str
    timestep: int
    state: tuple[float, ...]
    action: int
    reward: float
    consequence: float
    timestamp: float
    done: bool
    delay: Optional[int] = None
    source_timestep: Optional[int] = None
    causal_label: Optional[float] = None

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> "LoggedTransition":
        required = (
            "episode_id", "timestep", "state", "action", "reward",
            "consequence", "timestamp", "done",
        )
        missing = [name for name in required if name not in row]
        if missing:
            raise ValueError(f"missing fields: {', '.join(missing)}")
        state = tuple(float(value) for value in row["state"])
        if not state or not all(math.isfinite(value) for value in state):
            raise ValueError("state must be a non-empty finite vector")
        values = (row["reward"], row["consequence"], row["timestamp"])
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("reward, consequence, and timestamp must be finite")
        delay = None if row.get("delay") is None else int(row["delay"])
        source = None if row.get("source_timestep") is None else int(row["source_timestep"])
        if delay is not None and delay < 0:
            raise ValueError("delay must be non-negative")
        if source is not None and source < 0:
            raise ValueError("source_timestep must be non-negative")
        causal_label = row.get("causal_label")
        if causal_label is not None and not math.isfinite(float(causal_label)):
            raise ValueError("causal_label must be finite")
        return cls(
            episode_id=str(row["episode_id"]),
            timestep=int(row["timestep"]),
            state=state,
            action=int(row["action"]),
            reward=float(row["reward"]),
            consequence=float(row["consequence"]),
            timestamp=float(row["timestamp"]),
            done=bool(row["done"]),
            delay=delay,
            source_timestep=source,
            causal_label=None if causal_label is None else float(causal_label),
        )


class LoggedTrajectoryDataset:
    def __init__(self, transitions: Iterable[LoggedTransition]):
        self.episodes: dict[str, tuple[LoggedTransition, ...]] = self._group(transitions)
        if not self.episodes:
            raise ValueError("dataset contains no transitions")
        self._validate()

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "LoggedTrajectoryDataset":
        rows = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON on line {line_number}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"line {line_number} must contain a JSON object")
                rows.append(LoggedTransition.from_mapping(row))
        return cls(rows)

    @staticmethod
    def _group(transitions: Iterable[LoggedTransition]) -> dict[str, tuple[LoggedTransition, ...]]:
        grouped: dict[str, list[LoggedTransition]] = defaultdict(list)
        for transition in transitions:
            grouped[transition.episode_id].append(transition)
        return {
            episode_id: tuple(sorted(rows, key=lambda row: row.timestep))
            for episode_id, rows in grouped.items()
        }

    def _validate(self) -> None:
        state_dims = {len(row.state) for rows in self.episodes.values() for row in rows}
        if len(state_dims) != 1:
            raise ValueError("all states must have the same dimension")
        for episode_id, rows in self.episodes.items():
            expected = list(range(len(rows)))
            actual = [row.timestep for row in rows]
            if actual != expected:
                raise ValueError(f"episode {episode_id!r} must have contiguous timesteps from 0")
            timestamps = [row.timestamp for row in rows]
            if any(later < earlier for earlier, later in zip(timestamps, timestamps[1:])):
                raise ValueError(f"episode {episode_id!r} timestamps are not monotonic")
            if sum(row.done for row in rows) > 1 or rows[-1].done is not True:
                raise ValueError(f"episode {episode_id!r} must end with exactly one done transition")
            for row in rows:
                if row.delay is not None:
                    expected_source = row.timestep - row.delay
                    if expected_source < 0:
                        raise ValueError(f"episode {episode_id!r} has delay before episode start")
                    if row.source_timestep is not None and row.source_timestep != expected_source:
                        raise ValueError(f"episode {episode_id!r} has inconsistent delay alignment")
                if row.source_timestep is not None and row.source_timestep > row.timestep:
                    raise ValueError(f"episode {episode_id!r} points to a future source timestep")

    @property
    def state_dim(self) -> int:
        first_episode = next(iter(self.episodes.values()))
        return len(first_episode[0].state)

    def alignment_records(self) -> list[dict[str, Any]]:
        records = []
        for episode_id, rows in self.episodes.items():
            for row in rows:
                if row.source_timestep is None and row.delay is None:
                    continue
                source = row.source_timestep
                if source is None:
                    source = row.timestep - int(row.delay)
                records.append({
                    "episode_id": episode_id,
                    "source_timestep": source,
                    "consequence_timestep": row.timestep,
                    "delay": row.timestep - source,
                    "consequence": row.consequence,
                })
        return records

    def summary(self) -> dict[str, Any]:
        rows = [row for episode in self.episodes.values() for row in episode]
        delays = [record["delay"] for record in self.alignment_records()]
        return {
            "episodes": len(self.episodes),
            "transitions": len(rows),
            "state_dim": self.state_dim,
            "causal_label_rate": sum(row.causal_label is not None for row in rows) / len(rows),
            "aligned_consequence_rate": len(self.alignment_records()) / len(rows),
            "observed_delay_count": len(delays),
            "observed_delay_mean": None if not delays else sum(delays) / len(delays),
            "observed_delay_min": None if not delays else min(delays),
            "observed_delay_max": None if not delays else max(delays),
        }


def audit_main() -> None:
    parser = argparse.ArgumentParser(description="Validate delayed-consequence JSONL data")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = LoggedTrajectoryDataset.from_jsonl(args.dataset).summary()
    payload = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
