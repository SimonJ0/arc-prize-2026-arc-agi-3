"""
Unit tests for validation gates.
"""

import numpy as np
import pandas as pd
import pytest
from src.core.gates import ValidationGatekeeper


def test_leakage_gate_overlap():
    gk = ValidationGatekeeper()
    train_ids = np.array([1, 2, 3, 4])
    val_ids = np.array([4, 5, 6])  # 4 overlaps!
    preds = np.array([[0.33, 0.33, 0.34], [0.33, 0.33, 0.34], [0.33, 0.33, 0.34]])
    res = gk.check_leakage(train_ids, val_ids, preds)
    assert not res.passed
    assert "CRITICAL: Found 1 overlapping IDs" in res.message


def test_leakage_gate_clean():
    gk = ValidationGatekeeper()
    train_ids = np.array([1, 2, 3])
    val_ids = np.array([4, 5, 6])
    preds = np.array([[0.33, 0.33, 0.34], [0.33, 0.33, 0.34], [0.33, 0.33, 0.34]])
    res = gk.check_leakage(train_ids, val_ids, preds)
    assert res.passed


def test_cv_gate():
    gk = ValidationGatekeeper(min_loss_improvement=0.001)
    # Improvement
    res = gk.check_cv_improvement(candidate_log_loss=1.050, best_log_loss=1.060)
    assert res.passed

    # Regression
    res_reg = gk.check_cv_improvement(candidate_log_loss=1.070, best_log_loss=1.060)
    assert not res_reg.passed


def test_submission_format_gate():
    gk = ValidationGatekeeper()
    test_df = pd.DataFrame({"id": [101, 102], "prompt": ["p1", "p2"]})

    # Valid submission
    sub_df = pd.DataFrame({
        "id": [101, 102],
        "winner_model_a": [0.4, 0.3],
        "winner_model_b": [0.4, 0.3],
        "winner_tie": [0.2, 0.4],
    })
    res = gk.verify_submission_format(sub_df, test_df)
    assert res.passed

    # Invalid probability sum
    sub_bad = sub_df.copy()
    sub_bad["winner_tie"] = [0.9, 0.9]
    res_bad = gk.verify_submission_format(sub_bad, test_df)
    assert not res_bad.passed


def test_diversity_gate():
    gk = ValidationGatekeeper(max_ensemble_correlation=0.95)
    
    # First model always passes
    m1 = np.array([[0.6, 0.2, 0.2], [0.1, 0.8, 0.1]])
    res_first = gk.check_diversity(m1, [])
    assert res_first.passed

    # Nearly identical model fails
    m2 = m1 + 0.0001
    res_redundant = gk.check_diversity(m2, [m1])
    assert not res_redundant.passed

    # Diverse model passes
    m3 = np.array([[0.1, 0.8, 0.1], [0.7, 0.1, 0.2]])
    res_diverse = gk.check_diversity(m3, [m1])
    assert res_diverse.passed


def test_leakage_gate_nan_and_inf():
    gk = ValidationGatekeeper()
    train_ids = np.array([1, 2])
    val_ids = np.array([3, 4])
    
    # NaN
    preds_nan = np.array([[np.nan, 0.5, 0.5], [0.3, 0.3, 0.4]])
    assert not gk.check_leakage(train_ids, val_ids, preds_nan).passed

    # Inf
    preds_inf = np.array([[np.inf, 0.5, 0.5], [0.3, 0.3, 0.4]])
    assert not gk.check_leakage(train_ids, val_ids, preds_inf).passed
