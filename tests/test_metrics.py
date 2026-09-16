"""
Unit tests for metric computations.
"""

import numpy as np
import pytest
from src.core.metrics import (
    compute_log_loss,
    compute_brier_score,
    compute_symmetry_divergence,
    normalize_probabilities,
)


def test_normalize_probabilities():
    raw = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    norm = normalize_probabilities(raw)
    assert norm.shape == (3, 3)
    assert np.allclose(norm.sum(axis=1), 1.0)
    assert (norm >= 0.0).all()
    assert (norm <= 1.0).all()


def test_compute_log_loss_uniform():
    # 3 classes uniform: log(1/3) = ln(3) ~ 1.098612
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y_pred = np.array([[1/3, 1/3, 1/3], [1/3, 1/3, 1/3], [1/3, 1/3, 1/3]])
    loss = compute_log_loss(y_true, y_pred)
    assert abs(loss - np.log(3)) < 1e-4


def test_compute_log_loss_perfect():
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y_pred = np.array([[0.9999, 0.00005, 0.00005], [0.00005, 0.9999, 0.00005], [0.00005, 0.00005, 0.9999]])
    loss = compute_log_loss(y_true, y_pred)
    assert loss < 0.01


def test_symmetry_divergence():
    # Perfectly symmetric
    p_norm = np.array([[0.6, 0.2, 0.2]])
    p_swap = np.array([[0.2, 0.6, 0.2]])
    div = compute_symmetry_divergence(p_norm, p_swap)
    assert div == 0.0

    # Highly asymmetric
    p_bad_swap = np.array([[0.6, 0.2, 0.2]])
    div_bad = compute_symmetry_divergence(p_norm, p_bad_swap)
    assert div_bad > 0.1
