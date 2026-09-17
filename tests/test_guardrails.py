"""
Unit tests for AnomalyGuardrail.
"""

import numpy as np

from src.core.guardrails import AnomalyGuardrail


def test_fold_stability():
    guard = AnomalyGuardrail(max_fold_std=0.04)

    # Stable folds
    stable_losses = [1.055, 1.061, 1.058, 1.052, 1.060]
    res_stable = guard.check_fold_stability(stable_losses)
    assert res_stable.passed is True

    # High variance folds
    volatile_losses = [1.050, 1.250, 0.980, 1.180, 1.020]
    res_volatile = guard.check_fold_stability(volatile_losses)
    assert res_volatile.passed is False
    assert "High fold variance" in res_volatile.message


def test_entropy_and_extremes():
    guard = AnomalyGuardrail(min_probability_bound=1e-5, max_probability_bound=0.999)

    # Well-behaved probabilities
    good_preds = np.array(
        [
            [0.45, 0.35, 0.20],
            [0.30, 0.50, 0.20],
            [0.33, 0.33, 0.34],
        ]
    )
    res_good = guard.check_entropy_and_extremes(good_preds)
    assert res_good.passed is True

    # Dangerous extreme probabilities
    extreme_preds = np.array(
        [
            [0.99999, 0.000005, 0.000005],
            [0.40, 0.40, 0.20],
        ]
    )
    res_extreme = guard.check_entropy_and_extremes(extreme_preds)
    assert res_extreme.passed is False
    assert "extreme probabilities" in res_extreme.message.lower()


def test_check_all_integration():
    guard = AnomalyGuardrail()
    good_preds = np.full((10, 3), 1.0 / 3.0)
    good_losses = [1.06, 1.07, 1.06, 1.07]

    res = guard.check_all(good_preds, good_losses)
    assert res.passed is True
