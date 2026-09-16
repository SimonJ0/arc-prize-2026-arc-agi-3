"""
Evaluation metrics and diagnostics for Kaggle LLM Classification Finetuning.
Includes exact Kaggle multi-class log loss, Brier score, ECE, and Position Symmetry divergence.
"""

from typing import Dict, Tuple, Union
import numpy as np
import pandas as pd


CLASSES = ["winner_model_a", "winner_model_b", "winner_tie"]
EPSILON = 1e-15


def normalize_probabilities(preds: np.ndarray, eps: float = EPSILON) -> np.ndarray:
    """
    Normalizes, clips, and re-normalizes predicted probabilities according
    to the official Kaggle log loss evaluation specification.
    """
    preds = np.asarray(preds, dtype=np.float64)
    # Avoid division by zero or NaN rows
    row_sums = preds.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    preds = preds / row_sums

    # Clip to avoid infinite log
    preds = np.clip(preds, eps, 1.0 - eps)

    # Re-normalize after clipping
    preds = preds / preds.sum(axis=1, keepdims=True)
    return preds


def compute_log_loss(
    y_true: Union[np.ndarray, pd.DataFrame],
    y_pred: Union[np.ndarray, pd.DataFrame],
    eps: float = EPSILON,
) -> float:
    """
    Computes exact Kaggle multi-class log loss.
    
    Args:
        y_true: Ground truth array of shape (N, 3) (one-hot or probabilities).
        y_pred: Predicted probabilities array of shape (N, 3).
        eps: Clipping threshold (default: 1e-15).
        
    Returns:
        float: Multi-class log loss.
    """
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true[CLASSES].values
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred[CLASSES].values

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = normalize_probabilities(y_pred, eps=eps)

    # Multi-class cross-entropy
    loss = -np.sum(y_true * np.log(y_pred)) / len(y_true)
    return float(loss)


def compute_brier_score(
    y_true: Union[np.ndarray, pd.DataFrame],
    y_pred: Union[np.ndarray, pd.DataFrame],
) -> float:
    """Computes multi-class Brier score (mean squared error of probabilities)."""
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true[CLASSES].values
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred[CLASSES].values

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = normalize_probabilities(y_pred)
    return float(np.mean(np.sum((y_pred - y_true) ** 2, axis=1)))


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Computes Expected Calibration Error (ECE) across confidence bins.
    """
    y_true_labels = np.argmax(y_true, axis=1)
    confidences = np.max(y_pred, axis=1)
    predictions = np.argmax(y_pred, axis=1)
    accuracies = (predictions == y_true_labels)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin

    return float(ece)


def compute_symmetry_divergence(
    preds_normal: np.ndarray,
    preds_swapped: np.ndarray,
) -> float:
    """
    Computes position symmetry divergence.
    When response A and B are swapped, the model's prediction for winner_model_a
    should match the original prediction for winner_model_b, and winner_tie
    should remain identical.
    
    Returns:
        float: Mean absolute difference between expected symmetric probabilities.
    """
    p_norm = normalize_probabilities(preds_normal)
    p_swap = normalize_probabilities(preds_swapped)

    # Under swap: expected p_swap[:, 0] == p_norm[:, 1]
    #             expected p_swap[:, 1] == p_norm[:, 0]
    #             expected p_swap[:, 2] == p_norm[:, 2]
    diff_a = np.abs(p_norm[:, 0] - p_swap[:, 1])
    diff_b = np.abs(p_norm[:, 1] - p_swap[:, 0])
    diff_tie = np.abs(p_norm[:, 2] - p_swap[:, 2])

    mean_div = float(np.mean(diff_a + diff_b + diff_tie) / 3.0)
    return mean_div


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    preds_swapped: np.ndarray = None,
) -> Dict[str, float]:
    """Computes a complete suite of evaluation metrics for validation reporting."""
    y_pred_norm = normalize_probabilities(y_pred)
    log_loss = compute_log_loss(y_true, y_pred_norm)
    brier = compute_brier_score(y_true, y_pred_norm)
    ece = compute_expected_calibration_error(y_true, y_pred_norm)

    # Accuracy
    acc = float(np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred_norm, axis=1)))

    metrics = {
        "log_loss": round(log_loss, 5),
        "brier_score": round(brier, 5),
        "ece": round(ece, 5),
        "accuracy": round(acc, 5),
    }

    if preds_swapped is not None:
        sym_div = compute_symmetry_divergence(y_pred, preds_swapped)
        metrics["symmetry_divergence"] = round(sym_div, 5)

    return metrics
