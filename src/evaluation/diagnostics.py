"""
Deep Diagnostic Engine for LLM Preference Evaluation.
Analyzes verbosity bias, calibration curves, confusion matrices, and fold breakdowns.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.core.metrics import (
    compute_brier_score,
    compute_expected_calibration_error,
    compute_log_loss,
    normalize_probabilities,
)


class DiagnosticEngine:
    """Computes comprehensive diagnostic metrics for an experiment."""

    @staticmethod
    def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
        """Computes 3x3 confusion matrix and normalized percentages."""
        y_true_labels = np.argmax(y_true, axis=1)
        y_pred_labels = np.argmax(y_pred, axis=1)
        cm = confusion_matrix(y_true_labels, y_pred_labels, labels=[0, 1, 2])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        return {
            "matrix": cm.tolist(),
            "normalized": np.round(cm_norm, 3).tolist(),
            "class_names": ["Model A", "Model B", "Tie"],
        }

    @staticmethod
    def compute_verbosity_bias_curve(
        df: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bins: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Analyzes win rates as a function of length difference (len_a - len_b).
        Crucial for detecting if model captures genuine quality vs blind verbosity bias.
        """
        char_diff = (df["response_a"].str.len() - df["response_b"].str.len()).values

        # Bin boundaries
        bins = np.quantile(char_diff, np.linspace(0, 1, n_bins + 1))
        # Ensure unique bin edges
        bins = np.unique(bins)
        if len(bins) < 2:
            return []

        curve = []
        for i in range(len(bins) - 1):
            low, high = bins[i], bins[i + 1]
            if i == len(bins) - 2:
                mask = (char_diff >= low) & (char_diff <= high)
            else:
                mask = (char_diff >= low) & (char_diff < high)

            count = int(np.sum(mask))
            if count == 0:
                continue

            sub_true = y_true[mask]
            sub_pred = y_pred[mask]

            actual_a_win = float(np.mean(sub_true[:, 0]))
            actual_b_win = float(np.mean(sub_true[:, 1]))
            actual_tie = float(np.mean(sub_true[:, 2]))

            pred_a_win = float(np.mean(sub_pred[:, 0]))
            pred_b_win = float(np.mean(sub_pred[:, 1]))
            pred_tie = float(np.mean(sub_pred[:, 2]))

            curve.append(
                {
                    "bin_range": f"[{int(low):+d}, {int(high):+d}]",
                    "sample_count": count,
                    "actual_win_a": round(actual_a_win, 3),
                    "pred_win_a": round(pred_a_win, 3),
                    "actual_win_b": round(actual_b_win, 3),
                    "pred_win_b": round(pred_b_win, 3),
                    "actual_tie": round(actual_tie, 3),
                    "pred_tie": round(pred_tie, 3),
                }
            )

        return curve

    @staticmethod
    def compute_calibration_curve(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bins: int = 10,
    ) -> list[dict[str, Any]]:
        """Computes reliability diagram points (confidence vs observed frequency)."""
        confidences = np.max(y_pred, axis=1)
        accuracies = np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1)

        bin_edges = np.linspace(0.3, 1.0, n_bins + 1)
        points = []

        for i in range(n_bins):
            low, high = bin_edges[i], bin_edges[i + 1]
            mask = (confidences >= low) & (
                confidences < high if i < n_bins - 1 else confidences <= high
            )
            count = int(np.sum(mask))
            if count > 0:
                mean_conf = float(np.mean(confidences[mask]))
                mean_acc = float(np.mean(accuracies[mask]))
                points.append(
                    {
                        "bin": f"{low:.2f}-{high:.2f}",
                        "mean_confidence": round(mean_conf, 3),
                        "empirical_accuracy": round(mean_acc, 3),
                        "count": count,
                    }
                )

        return points

    def run_full_diagnostics(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        fold_scores: list[float] | None = None,
        feature_importances: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Runs full diagnostic suite and returns serializable report dictionary."""
        y_pred_norm = normalize_probabilities(y_pred)
        overall_loss = compute_log_loss(y_true, y_pred_norm)
        brier = compute_brier_score(y_true, y_pred_norm)
        ece = compute_expected_calibration_error(y_true, y_pred_norm)

        cm_data = self.compute_confusion_matrix(y_true, y_pred_norm)
        verbosity_data = self.compute_verbosity_bias_curve(df, y_true, y_pred_norm)
        calibration_data = self.compute_calibration_curve(y_true, y_pred_norm)

        top_feats = []
        if feature_importances:
            top_feats = [
                {"name": k, "importance": round(float(v), 2)}
                for k, v in list(feature_importances.items())[:15]
            ]

        return {
            "overall_log_loss": round(overall_loss, 5),
            "brier_score": round(brier, 5),
            "ece": round(ece, 5),
            "confusion_matrix": cm_data,
            "verbosity_curve": verbosity_data,
            "calibration_curve": calibration_data,
            "fold_scores": [round(s, 5) for s in (fold_scores or [])],
            "top_features": top_feats,
        }
