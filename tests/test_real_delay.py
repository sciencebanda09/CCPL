import json

import pytest

from ccpl.real_delay import LoggedTrajectoryDataset


def _rows():
    return [
        {"episode_id": "a", "timestep": 0, "state": [0.1, 0.2], "action": 1,
         "reward": 1.0, "consequence": 0.0, "timestamp": 10.0, "done": False},
        {"episode_id": "a", "timestep": 1, "state": [0.2, 0.3], "action": 0,
         "reward": 0.0, "consequence": 2.0, "timestamp": 11.0, "done": True,
         "delay": 1, "causal_label": 2.0},
    ]


def test_logged_dataset_validates_delay_alignment(tmp_path):
    path = tmp_path / "trajectory.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in _rows()) + "\n", encoding="utf-8")
    dataset = LoggedTrajectoryDataset.from_jsonl(path)
    assert dataset.summary()["episodes"] == 1
    assert dataset.summary()["observed_delay_mean"] == 1.0
    assert dataset.alignment_records()[0]["source_timestep"] == 0


def test_logged_dataset_rejects_inconsistent_source(tmp_path):
    rows = _rows()
    rows[1]["source_timestep"] = 1
    path = tmp_path / "invalid.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent delay"):
        LoggedTrajectoryDataset.from_jsonl(path)
