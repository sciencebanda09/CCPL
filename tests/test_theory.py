import numpy as np
import pytest

from ccpl_theory import effective_discount, effective_discount_bounds


def test_effective_discount_is_a_convex_combination_of_delay_discounts():
    probabilities = np.array([[1.0, 0.0, 0.0], [0.0, 0.25, 0.75]])
    values = effective_discount(probabilities, gamma=0.9)
    assert np.allclose(values, [1.0, 0.25 * 0.9 + 0.75 * 0.9 ** 2])
    bounds = effective_discount_bounds(probabilities, gamma=0.9)
    assert bounds["support_valid"]
    assert not bounds["strict_contraction_if_min_delay_positive"]


def test_effective_discount_requires_a_probability_simplex():
    with pytest.raises(ValueError):
        effective_discount([[0.5, 0.25]], gamma=0.9)
    with pytest.raises(ValueError):
        effective_discount([[-0.1, 1.1]], gamma=0.9)


def test_positive_delays_give_a_strict_modulus_bound():
    probabilities = np.array([[0.0, 1.0, 0.0], [0.0, 0.2, 0.8]])
    bounds = effective_discount_bounds(probabilities, gamma=0.9)
    assert bounds["strict_contraction_if_min_delay_positive"]
    assert bounds["max"] <= 0.9 + 1e-12
