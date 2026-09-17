"""
Validation Gates for the Self-Improving Kaggle Competition Solver.
Enforces CV gate, anti-leakage gate, position symmetry gate, ensemble diversity gate,
and the 'Iron Rule' submission gate.
"""

import numpy as np
import pandas as pd

from src.core.guardrails import AnomalyGuardrail, GateResult
from src.core.metrics import compute_symmetry_divergence


class ValidationGatekeeper:
    """
    Orchestrates validation gates inspired by Tom's BirdCLEF-2026 solution.
    No model can be promoted or ensembled without passing these checks.
    """

    def __init__(
        self,
        min_loss_improvement: float = 0.0001,
        max_symmetry_divergence: float = 0.12,
        max_ensemble_correlation: float = 0.96,
    ):
        self.min_loss_improvement = min_loss_improvement
        self.max_symmetry_divergence = max_symmetry_divergence
        self.max_ensemble_correlation = max_ensemble_correlation
        self.anomaly_guardrail = AnomalyGuardrail()

    def check_leakage(
        self,
        train_ids: np.ndarray,
        val_ids: np.ndarray,
        preds: np.ndarray,
    ) -> GateResult:
        """Gate 1: Checks for fold leakage, NaN, Inf, and shape mismatch."""
        train_set = set(train_ids)
        val_set = set(val_ids)
        overlap = train_set.intersection(val_set)

        if len(overlap) > 0:
            return GateResult(
                passed=False,
                gate_name="Leakage Gate",
                message=f"CRITICAL: Found {len(overlap)} overlapping IDs between train and val splits!",
                details={"overlap_count": len(overlap)},
            )

        if np.isnan(preds).any():
            return GateResult(
                passed=False,
                gate_name="Leakage Gate",
                message="CRITICAL: NaN values detected in predictions!",
                details={"nan_count": int(np.isnan(preds).sum())},
            )

        if np.isinf(preds).any():
            return GateResult(
                passed=False,
                gate_name="Leakage Gate",
                message="CRITICAL: Infinite values detected in predictions!",
                details={"inf_count": int(np.isinf(preds).sum())},
            )

        if preds.shape[1] != 3:
            return GateResult(
                passed=False,
                gate_name="Leakage Gate",
                message=f"Shape error: Expected 3 probability columns, got {preds.shape[1]}",
                details={"shape": list(preds.shape)},
            )

        return GateResult(
            passed=True,
            gate_name="Leakage Gate",
            message="No data leakage or numerical anomalies detected.",
            details={"samples_validated": len(preds)},
        )

    def check_cv_improvement(
        self,
        candidate_log_loss: float,
        best_log_loss: float | None,
        allow_slight_regression_for_diversity: bool = False,
    ) -> GateResult:
        """Gate 2: Verifies OOF Log Loss against baseline."""
        if best_log_loss is None:
            return GateResult(
                passed=True,
                gate_name="CV Gate",
                message=f"First model accepted as initial baseline. Log Loss: {candidate_log_loss:.5f}",
                details={"candidate_loss": candidate_log_loss, "delta": 0.0},
            )

        delta = candidate_log_loss - best_log_loss  # Negative is better

        if delta <= -self.min_loss_improvement:
            return GateResult(
                passed=True,
                gate_name="CV Gate",
                message=f"CV IMPROVEMENT: Log loss improved by {-delta:.5f} ({best_log_loss:.5f} -> {candidate_log_loss:.5f})",
                details={
                    "candidate_loss": candidate_log_loss,
                    "best_loss": best_log_loss,
                    "delta": round(delta, 5),
                },
            )
        elif allow_slight_regression_for_diversity and delta <= 0.005:
            return GateResult(
                passed=True,
                gate_name="CV Gate",
                message=f"Accepted for ensemble pool (diversity candidate within +{delta:.5f} of best).",
                details={
                    "candidate_loss": candidate_log_loss,
                    "best_loss": best_log_loss,
                    "delta": round(delta, 5),
                },
            )
        else:
            return GateResult(
                passed=False,
                gate_name="CV Gate",
                message=f"CV REJECTED: Candidate log loss {candidate_log_loss:.5f} did not beat best {best_log_loss:.5f} (delta: +{delta:.5f})",
                details={
                    "candidate_loss": candidate_log_loss,
                    "best_loss": best_log_loss,
                    "delta": round(delta, 5),
                },
            )

    def check_symmetry(
        self,
        preds_normal: np.ndarray,
        preds_swapped: np.ndarray | None,
    ) -> GateResult:
        """Gate 3: Checks position symmetry invariance."""
        if preds_swapped is None:
            return GateResult(
                passed=True,
                gate_name="Symmetry Gate",
                message="Symmetry check skipped (swapped predictions not provided).",
            )

        div = compute_symmetry_divergence(preds_normal, preds_swapped)
        passed = div <= self.max_symmetry_divergence

        return GateResult(
            passed=passed,
            gate_name="Symmetry Gate",
            message=f"Symmetry divergence: {div:.4f} (Threshold <= {self.max_symmetry_divergence:.4f})",
            details={"symmetry_divergence": round(div, 5), "passed": passed},
        )

    def check_diversity(
        self,
        candidate_oof: np.ndarray,
        existing_oofs: list[np.ndarray],
    ) -> GateResult:
        """Gate 4: Verifies correlation with existing models to prevent redundant ensembling."""
        if not existing_oofs:
            return GateResult(
                passed=True,
                gate_name="Diversity Gate",
                message="First model in ensemble pool - diversity check satisfied.",
                details={"max_correlation": 0.0},
            )

        cand_flat = candidate_oof.ravel()
        max_corr = 0.0

        for _idx, prev_oof in enumerate(existing_oofs):
            corr = np.corrcoef(cand_flat, prev_oof.ravel())[0, 1]
            if not np.isnan(corr) and corr > max_corr:
                max_corr = corr

        passed = max_corr <= self.max_ensemble_correlation
        return GateResult(
            passed=passed,
            gate_name="Diversity Gate",
            message=f"Max ensemble correlation: {max_corr:.4f} (Threshold <= {self.max_ensemble_correlation:.4f})",
            details={"max_correlation": round(float(max_corr), 4), "passed": passed},
        )

    def verify_submission_format(
        self,
        submission_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> GateResult:
        """Gate 5: Strict validation of submission CSV prior to any upload."""
        expected_cols = ["id", "winner_model_a", "winner_model_b", "winner_tie"]

        if list(submission_df.columns) != expected_cols:
            return GateResult(
                passed=False,
                gate_name="Submission Gate",
                message=f"Column mismatch: Expected {expected_cols}, got {list(submission_df.columns)}",
            )

        if len(submission_df) != len(test_df):
            return GateResult(
                passed=False,
                gate_name="Submission Gate",
                message=f"Row count mismatch: Submission has {len(submission_df)} rows, test has {len(test_df)}",
            )

        if not (submission_df["id"].values == test_df["id"].values).all():
            return GateResult(
                passed=False,
                gate_name="Submission Gate",
                message="ID sequence mismatch between test set and submission!",
            )

        prob_values = submission_df[["winner_model_a", "winner_model_b", "winner_tie"]].values

        if np.isnan(prob_values).any() or np.isinf(prob_values).any():
            return GateResult(
                passed=False,
                gate_name="Submission Gate",
                message="Submission contains NaN or Inf probability values!",
            )

        if (prob_values < 0.0).any() or (prob_values > 1.0).any():
            return GateResult(
                passed=False,
                gate_name="Submission Gate",
                message="Probabilities out of valid range [0.0, 1.0]!",
            )

        row_sums = prob_values.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-4):
            return GateResult(
                passed=False,
                gate_name="Submission Gate",
                message="Probabilities in submission do not sum to 1.0!",
            )

        return GateResult(
            passed=True,
            gate_name="Submission Gate",
            message=f"Submission format perfectly valid ({len(submission_df)} rows, valid probabilities).",
            details={"rows": len(submission_df)},
        )

    def check_anomalies(self, oof_preds: np.ndarray, fold_losses: list[float]) -> GateResult:
        """Gate 5: Checks for validation anomalies, entropy collapse, and fold instability."""
        return self.anomaly_guardrail.check_all(oof_preds, fold_losses)
