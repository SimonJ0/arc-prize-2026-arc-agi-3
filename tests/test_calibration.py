"""
Unit tests for post-hoc temperature calibration.
"""

import numpy as np
import pytest
from src.core.calibration import TemperatureCalibrator, optimize_temperature
from src.core.metrics import compute_log_loss


def test_temperature_calibration_overconfidence():
    """Verify temperature scaling reduces loss on overconfident predictions."""
    np.random.seed(42)
    n = 200
    y_true = np.zeros((n, 3))
    # True class distribution
    true_labels = np.random.choice([0, 1, 2], size=n, p=[0.4, 0.4, 0.2])
    for i, label in enumerate(true_labels):
        y_true[i, label] = 1.0

    # Overconfident predictions (e.g. 0.99 for predicted class, 0.005 for others)
    probs = np.full((n, 3), 0.05)
    for i, label in enumerate(true_labels):
        # 30% error rate but predicting 0.9 confidence
        if np.random.rand() > 0.3:
            probs[i, label] = 0.9
        else:
            wrong_label = (label + 1) % 3
            probs[i, wrong_label] = 0.9
    probs = probs / probs.sum(axis=1, keepdims=True)

    uncalibrated_loss = compute_log_loss(y_true, probs)

    calibrator = TemperatureCalibrator()
    calibrator.fit(y_true, probs)
    calibrated_probs = calibrator.calibrate(probs)
    calibrated_loss = compute_log_loss(y_true, calibrated_probs)

    # Temperature should be > 1.0 to soften overconfidence
    assert calibrator.temperature > 1.0
    # Calibrated loss must be lower than uncalibrated loss
    assert calibrated_loss < uncalibrated_loss


def test_temperature_calibration_perfect():
    """If already well-calibrated, temperature should remain near 1.0."""
    y_true = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    probs = np.array([
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
    ])
    T = optimize_temperature(y_true, probs)
    assert 0.1 <= T <= 5.0
