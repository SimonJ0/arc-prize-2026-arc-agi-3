"""
Validation Anomaly Guardrail.
Monitors candidate models for probability entropy collapse, uncalibrated extreme confidences,
and excessive cross-fold variance to protect against catastrophic log loss degradation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np


@dataclass
class GateResult:
    passed: bool
    gate_name: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


class AnomalyGuardrail:
    """
    Guards against validation anomalies before a model can pass the validation gatekeeper.
    """

    def __init__(
        self,
        max_fold_std: float = 0.05,
        min_probability_bound: float = 1e-6,
        max_probability_bound: float = 0.9999,
        min_mean_entropy: float = 0.05,
    ):
        self.max_fold_std = max_fold_std
        self.min_probability_bound = min_probability_bound
        self.max_probability_bound = max_probability_bound
        self.min_mean_entropy = min_mean_entropy

    def check_fold_stability(self, fold_losses: List[float]) -> GateResult:
        """Flags high variance between CV fold losses."""
        if not fold_losses or len(fold_losses) < 2:
            return GateResult(
                passed=True,
                gate_name="Fold Stability Guardrail",
                message="Fold stability check skipped (fewer than 2 folds).",
                details={"fold_count": len(fold_losses)},
            )

        fold_std = float(np.std(fold_losses))
        fold_range = float(np.max(fold_losses) - np.min(fold_losses))

        if fold_std > self.max_fold_std:
            return GateResult(
                passed=False,
                gate_name="Fold Stability Guardrail",
                message=f"ANOMALY: High fold variance detected! std={fold_std:.4f} > {self.max_fold_std:.4f}",
                details={"fold_std": fold_std, "fold_range": fold_range, "fold_losses": fold_losses},
            )

        return GateResult(
            passed=True,
            gate_name="Fold Stability Guardrail",
            message=f"Fold stability verified (std={fold_std:.4f} <= {self.max_fold_std:.4f}).",
            details={"fold_std": fold_std, "fold_range": fold_range},
        )

    def check_entropy_and_extremes(self, oof_preds: np.ndarray) -> GateResult:
        """Flags uncalibrated probability collapse or extreme values."""
        clipped = np.clip(oof_preds, 1e-15, 1.0 - 1e-15)
        # Compute Shannon entropy per sample: H = -sum(p * log(p))
        entropy = -np.sum(clipped * np.log(clipped), axis=1)
        mean_entropy = float(np.mean(entropy))

        min_p = float(np.min(oof_preds))
        max_p = float(np.max(oof_preds))

        if mean_p_collapse := (mean_entropy < self.min_mean_entropy):
            return GateResult(
                passed=False,
                gate_name="Entropy Guardrail",
                message=f"ANOMALY: Probability entropy collapse detected (mean H={mean_entropy:.4f} < {self.min_mean_entropy:.4f}).",
                details={"mean_entropy": mean_entropy, "min_p": min_p, "max_p": max_p},
            )

        if min_p < self.min_probability_bound or max_p > self.max_probability_bound:
            return GateResult(
                passed=False,
                gate_name="Extreme Probability Guardrail",
                message=f"ANOMALY: Dangerous extreme probabilities detected (min={min_p:.6f}, max={max_p:.6f}). Requires calibration.",
                details={"min_p": min_p, "max_p": max_p, "mean_entropy": mean_entropy},
            )

        return GateResult(
            passed=True,
            gate_name="Entropy & Extremes Guardrail",
            message=f"Probability distribution well-behaved (H={mean_entropy:.3f}, range=[{min_p:.4f}, {max_p:.4f}]).",
            details={"mean_entropy": mean_entropy, "min_p": min_p, "max_p": max_p},
        )

    def check_all(self, oof_preds: np.ndarray, fold_losses: List[float]) -> GateResult:
        """Runs all anomaly guardrail checks."""
        stab_res = self.check_fold_stability(fold_losses)
        if not stab_res.passed:
            return stab_res

        ent_res = self.check_entropy_and_extremes(oof_preds)
        if not ent_res.passed:
            return ent_res

        return GateResult(
            passed=True,
            gate_name="Anomaly Guardrail",
            message="All validation anomaly guardrails verified clean.",
            details={"fold_std": stab_res.details.get("fold_std", 0.0), "entropy": ent_res.details.get("mean_entropy", 0.0)},
        )
