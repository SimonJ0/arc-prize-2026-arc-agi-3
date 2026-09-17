"""
Unit tests for convex ensemble blending and nested CV optimization.
"""

import numpy as np

from src.models.ensemble import EnsembleBlender


def test_ensemble_blender_weights_sum_to_one():
    np.random.seed(42)
    n = 100
    y_true = np.zeros((n, 3))
    for i, c in enumerate(np.random.choice([0, 1, 2], size=n)):
        y_true[i, c] = 1.0

    oof1 = np.clip(y_true + np.random.normal(0, 0.2, (n, 3)), 0.01, 0.99)
    oof1 = oof1 / oof1.sum(axis=1, keepdims=True)
    oof2 = np.full((n, 3), 1.0 / 3.0)

    blender = EnsembleBlender(names=["m1", "m2"])
    blender.fit([oof1, oof2], y_true)

    assert blender.weights is not None
    assert len(blender.weights) == 2
    assert np.isclose(np.sum(blender.weights), 1.0)
    assert (blender.weights >= 0).all()
    # m1 has signal, m2 is random, so m1 should get dominant weight
    assert blender.weights[0] > blender.weights[1]


def test_ensemble_nested_cv():
    np.random.seed(42)
    n = 150
    y_true = np.zeros((n, 3))
    for i, c in enumerate(np.random.choice([0, 1, 2], size=n)):
        y_true[i, c] = 1.0

    oof1 = np.clip(y_true + np.random.normal(0, 0.3, (n, 3)), 0.01, 0.99)
    oof1 = oof1 / oof1.sum(axis=1, keepdims=True)
    oof2 = np.clip(y_true + np.random.normal(0, 0.4, (n, 3)), 0.01, 0.99)
    oof2 = oof2 / oof2.sum(axis=1, keepdims=True)

    blender = EnsembleBlender(names=["m1", "m2"])
    blender.fit_with_nested_cv([oof1, oof2], y_true, n_blend_folds=3)

    assert blender.weights is not None
    assert len(blender.weights) == 2
    assert np.isclose(np.sum(blender.weights), 1.0)
    assert (blender.weights >= 0).all()
    assert blender.optimal_log_loss is not None
