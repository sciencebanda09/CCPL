import numpy as np

from interventions import evaluate_intervention_predictions, make_intervention_dataset
from stats import paired_effect_summary


def test_intervention_dataset_is_reproducible_and_held_out():
    first = make_intervention_dataset(20, seed=4)
    second = make_intervention_dataset(20, seed=4)
    assert np.array_equal(first["train"]["states"], second["train"]["states"])
    assert len(first["train"]["states"]) + len(first["test"]["states"]) == 20


def test_intervention_metrics_are_finite():
    result = evaluate_intervention_predictions([1.0, -1.0], [0.5, -0.5])
    assert result["mae"] == 0.5
    assert result["sign_agreement"] == 1.0


def test_paired_effect_summary_reports_seed_uncertainty():
    result = paired_effect_summary([3.0, 4.0, 5.0], [1.0, 2.0, 4.0])
    assert result["valid"]
    assert result["n"] == 3
    assert result["mean_difference"] == 5.0 / 3.0
    assert len(result["ci95"]) == 2
