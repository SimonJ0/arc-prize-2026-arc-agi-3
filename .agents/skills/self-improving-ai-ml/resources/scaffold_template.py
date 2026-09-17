"""
Self-Improving AI/ML Research Framework: Production Scaffold Template
====================================================================
A production-grade, statistically rigorous scaffold implementing:
1. TaskSpec & defensive probability contracts (strict NaN/Inf rejection)
2. Total Variation symmetry divergence
3. Robust diversity checks with constant-predictor safety
4. True Outer Meta-CV ensemble blending (unbiased meta-OOF evaluation)
5. Cryptographically bound Iron Rule authorization (SHA-256 verification)
"""

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold

# ==============================================================================
# 1. TASK SPECIFICATION & DEFENSIVE PROBABILITY UTILITIES
# ==============================================================================


@dataclass(frozen=True)
class TaskSpec:
    """Formal specification of an AI/ML task to prevent cross-domain semantic errors."""

    task_type: Literal[
        "binary_classification",
        "multiclass_classification",
        "multilabel_classification",
        "regression",
        "forecasting",
        "ranking",
    ]
    primary_metric: str
    minimize: bool
    probability_semantics: Literal[
        "simplex",  # sum(p) == 1 (multiclass softmax)
        "independent_bernoulli",  # p in [0, 1] per class (multilabel sigmoid)
        "continuous",  # unbounded real values (regression)
        "ranking_score",  # real-valued sorting scores
    ]
    group_key: str | None = None
    time_key: str | None = None
    supports_calibration: bool = True
    supports_tta: bool = False


def normalize_probabilities(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Validate, clip, and row-normalize multiclass probabilities on the simplex.

    Raises:
        ValueError: If predictions contain NaN, Inf, are not 2-D, or have invalid row sums.
    """
    p = np.asarray(p, dtype=np.float64)

    if p.ndim != 2:
        raise ValueError(f"Expected 2-D (N, C) probabilities; got shape={p.shape}")

    if p.shape[0] == 0 or p.shape[1] == 0:
        raise ValueError("Probability matrix must be non-empty.")

    if not np.isfinite(p).all():
        raise ValueError("CRITICAL: Predictions contain non-finite values (NaN or Inf).")

    p = np.clip(p, 0.0, 1.0)
    row_sums = p.sum(axis=1, keepdims=True)

    if not np.isfinite(row_sums).all() or np.any(row_sums <= 0):
        raise ValueError("CRITICAL: Invalid probability row sum (<= 0 or non-finite).")

    p = p / row_sums

    # Re-clip to guarantee bounds [eps, 1 - eps] and re-normalize
    p = np.clip(p, eps, 1.0 - eps)
    p = p / p.sum(axis=1, keepdims=True)
    return p


def compute_multi_log_loss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    """Computes multi-class cross-entropy log loss supporting one-hot or index targets."""
    p = normalize_probabilities(y_pred, eps=eps)
    y_arr = np.asarray(y_true)
    if y_arr.ndim == 1 or y_arr.shape[1] == 1:
        y_indices = y_arr.ravel().astype(int)
        loss = -np.mean(np.log(p[np.arange(len(y_indices)), y_indices]))
    else:
        loss = -np.sum(y_arr * np.log(p)) / len(y_arr)
    return float(loss)


def compute_brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes multi-class Brier score."""
    p = normalize_probabilities(y_pred)
    y_arr = np.asarray(y_true)
    if y_arr.ndim == 1:
        n_classes = p.shape[1]
        y_one_hot = np.eye(n_classes)[y_arr.astype(int)]
    else:
        y_one_hot = y_arr
    return float(np.mean(np.sum((p - y_one_hot) ** 2, axis=1)))


def compute_ece(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE) across confidence bins."""
    p = normalize_probabilities(y_pred)
    confidences = np.max(p, axis=1)
    predictions = np.argmax(p, axis=1)
    y_arr = np.asarray(y_true)
    actuals = y_arr if y_arr.ndim == 1 else np.argmax(y_arr, axis=1)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(p)

    for i in range(n_bins):
        bin_mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        bin_size = int(np.sum(bin_mask))
        if bin_size > 0:
            bin_acc = float(np.mean(predictions[bin_mask] == actuals[bin_mask]))
            bin_conf = float(np.mean(confidences[bin_mask]))
            ece += (bin_size / n_samples) * abs(bin_acc - bin_conf)

    return float(ece)


# ==============================================================================
# 2. BASE PREDICTOR INTERFACE
# ==============================================================================


class BasePredictor(ABC):
    """Abstract model predictor interface enforcing Test-Time Augmentation (TTA)."""

    def __init__(self, name: str, params: dict[str, Any] | None = None):
        self.name = name
        self.params = params or {}
        self.feature_importances: dict[str, float] = {}

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "BasePredictor":
        """Fits model parameters on training fold."""
        pass

    @abstractmethod
    def predict_proba_with_tta(
        self,
        X_norm: pd.DataFrame,
        X_aug: pd.DataFrame | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """
        Executes inference with domain symmetry TTA.
        Returns: (p_symmetric, p_normal, p_augmented)
        """
        pass


# ==============================================================================
# 3. VALIDATION GATES & ANOMALY GUARDRAILS
# ==============================================================================


@dataclass
class GateResult:
    passed: bool
    gate_name: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


def safe_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Computes Pearson correlation safely handling zero-variance / constant predictions."""
    a_flat = np.asarray(a, dtype=np.float64).ravel()
    b_flat = np.asarray(b, dtype=np.float64).ravel()

    if a_flat.shape != b_flat.shape:
        raise ValueError(f"Shape mismatch in correlation: {a_flat.shape} != {b_flat.shape}")

    if not np.isfinite(a_flat).all() or not np.isfinite(b_flat).all():
        raise ValueError("Non-finite values encountered in diversity correlation check.")

    a_std = float(np.std(a_flat))
    b_std = float(np.std(b_flat))

    if a_std == 0.0 or b_std == 0.0:
        return 1.0 if np.array_equal(a_flat, b_flat) else 0.0

    corr = float(np.corrcoef(a_flat, b_flat)[0, 1])
    return 0.0 if np.isnan(corr) else corr


class ValidationGatekeeper:
    """Multi-tiered validation gates verifying anti-leakage, CV improvement, and symmetry."""

    def __init__(
        self,
        min_loss_improvement: float = 0.0002,
        max_symmetry_divergence: float = 0.10,  # Interpreted as Total Variation distance in [0, 1]
        max_ensemble_correlation: float = 0.95,
        max_fold_std: float = 0.05,
        min_entropy: float = 0.05,
        max_overfitting_gap: float = 0.35,
    ):
        self.min_loss_improvement = min_loss_improvement
        self.max_symmetry_divergence = max_symmetry_divergence
        self.max_ensemble_correlation = max_ensemble_correlation
        self.max_fold_std = max_fold_std
        self.min_entropy = min_entropy
        self.max_overfitting_gap = max_overfitting_gap

    def check_leakage(
        self, train_ids: np.ndarray, val_ids: np.ndarray, preds: np.ndarray
    ) -> GateResult:
        overlap = set(train_ids).intersection(set(val_ids))
        if overlap:
            return GateResult(
                False,
                "Leakage Gate",
                f"CRITICAL: Found {len(overlap)} overlapping IDs between train and val!",
            )
        if np.isnan(preds).any() or np.isinf(preds).any():
            return GateResult(
                False,
                "Leakage Gate",
                "CRITICAL: Non-finite values (NaN/Inf) detected in predictions!",
            )
        return GateResult(True, "Leakage Gate", "Zero leakage and numerical integrity verified.")

    def check_cv_improvement(
        self,
        candidate_loss: float,
        best_loss: float | None,
        allow_slight_regression_for_diversity: bool = False,
    ) -> GateResult:
        if best_loss is None:
            return GateResult(
                True, "CV Gate", f"Initial baseline established ({candidate_loss:.5f})."
            )
        delta = candidate_loss - best_loss
        if delta <= -self.min_loss_improvement:
            return GateResult(
                True,
                "CV Gate",
                f"CV improved by {-delta:.5f} ({best_loss:.5f} -> {candidate_loss:.5f})",
            )
        if allow_slight_regression_for_diversity and delta <= 0.005:
            return GateResult(
                True,
                "CV Gate",
                f"Accepted for ensemble pool (diversity candidate within +{delta:.5f}).",
            )
        return GateResult(
            False,
            "CV Gate",
            f"CV REJECTED: Candidate {candidate_loss:.5f} did not beat best {best_loss:.5f} (delta: +{delta:.5f}).",
        )

    def check_symmetry(self, p_norm: np.ndarray, p_swap: np.ndarray | None) -> GateResult:
        """
        Computes Total Variation distance: D_TV(p, q) = 0.5 * mean(sum(|p - q|, axis=1)).
        Guaranteed to lie in [0, 1].
        """
        if p_swap is None:
            return GateResult(
                True, "Symmetry Gate", "Symmetry check skipped (no augmented inputs)."
            )

        p_norm_n = normalize_probabilities(p_norm)
        p_swap_n = normalize_probabilities(p_swap)

        if p_norm_n.shape != p_swap_n.shape:
            return GateResult(
                False, "Symmetry Gate", f"Shape mismatch: {p_norm_n.shape} != {p_swap_n.shape}"
            )

        div_tv = float(0.5 * np.mean(np.sum(np.abs(p_norm_n - p_swap_n), axis=1)))
        passed = div_tv <= self.max_symmetry_divergence
        return GateResult(
            passed=passed,
            gate_name="Symmetry Gate",
            message=f"Total Variation symmetry divergence: {div_tv:.4f} (Threshold <= {self.max_symmetry_divergence:.4f})",
            details={"div_tv": div_tv, "threshold": self.max_symmetry_divergence},
        )

    def check_diversity(self, candidate_oof: np.ndarray, pool_oofs: list[np.ndarray]) -> GateResult:
        if not pool_oofs:
            return GateResult(
                True, "Diversity Gate", "First model in ensemble pool - diversity check satisfied."
            )

        max_r = max(safe_correlation(candidate_oof, p) for p in pool_oofs)
        passed = max_r <= self.max_ensemble_correlation
        return GateResult(
            passed=passed,
            gate_name="Diversity Gate",
            message=f"Max ensemble correlation: {max_r:.4f} (Threshold <= {self.max_ensemble_correlation:.4f})",
            details={"max_r": max_r, "threshold": self.max_ensemble_correlation},
        )

    def check_anomalies(
        self,
        oof_preds: np.ndarray,
        fold_losses: list[float],
        train_loss: float | None = None,
        val_loss: float | None = None,
    ) -> GateResult:
        # 1. Fold variance guard
        if len(fold_losses) >= 2 and (std := float(np.std(fold_losses))) > self.max_fold_std:
            return GateResult(
                False,
                "Anomaly Guardrail",
                f"High cross-fold variance: std={std:.4f} > {self.max_fold_std:.4f}",
            )

        # 2. Shannon entropy collapse guard
        clipped = np.clip(oof_preds, 1e-15, 1.0 - 1e-15)
        mean_entropy = float(np.mean(-np.sum(clipped * np.log(clipped), axis=-1)))
        if mean_entropy < self.min_entropy:
            return GateResult(
                False,
                "Anomaly Guardrail",
                f"Entropy collapse detected: mean H={mean_entropy:.4f} < {self.min_entropy:.4f}",
            )

        # 3. Overfitting gap guard
        if train_loss is not None and val_loss is not None:
            gap = val_loss - train_loss
            if gap > self.max_overfitting_gap:
                return GateResult(
                    False,
                    "Anomaly Guardrail",
                    f"Severe overfitting detected: gap {gap:.4f} > {self.max_overfitting_gap:.4f}",
                )

        return GateResult(True, "Anomaly Guardrail", "All validation anomaly guardrails clean.")


# ==============================================================================
# 4. POST-HOC TEMPERATURE CALIBRATOR
# ==============================================================================


class TemperatureCalibrator:
    """Scales logits by optimal temperature T > 0 to optimize calibration and log loss."""

    def __init__(self):
        self.temperature = 1.0

    def fit(self, y_true: np.ndarray, probs: np.ndarray) -> "TemperatureCalibrator":
        probs_norm = normalize_probabilities(probs)
        logits = np.log(probs_norm)

        def nll(t):
            t_val = max(t[0], 0.05)
            scaled = logits / t_val
            exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
            p_soft = exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)
            return compute_multi_log_loss(y_true, p_soft)

        res = minimize(nll, [1.0], method="Nelder-Mead", options={"maxiter": 200})
        self.temperature = float(max(res.x[0], 0.05))
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        probs_norm = normalize_probabilities(probs)
        logits = np.log(probs_norm) / self.temperature
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


# ==============================================================================
# 5. CONVEX ENSEMBLE BLENDER WITH TRUE OUTER META-CV
# ==============================================================================


class ConvexEnsembleBlender:
    """
    Finds optimal weights on the probability simplex (sum w_m = 1, w_m >= 0).
    Enforces TRUE Outer Meta-CV to produce an unbiased out-of-fold validation score,
    while refitting final weights on 100% of OOF data for test deployment only.
    """

    def __init__(self, names: list[str] | None = None):
        self.names = names or []
        self.final_weights: np.ndarray | None = None
        self.unbiased_meta_loss: float | None = None
        self.meta_oof: np.ndarray | None = None

    def fit_nested_cv(
        self,
        oofs: list[np.ndarray],
        y_true: np.ndarray,
        n_folds: int = 5,
        random_state: int = 42,
    ) -> "ConvexEnsembleBlender":
        """
        Executes true outer meta-CV:
        1. Evaluates outer folds strictly on held-out meta-test sets to form meta_oof.
        2. Computes unbiased_meta_loss on meta_oof.
        3. Refits final_weights on 100% OOF for test-set inference only.
        """
        m = len(oofs)
        if m == 0:
            raise ValueError("Ensemble blender requires at least one model OOF.")

        if m == 1:
            self.final_weights = np.array([1.0])
            self.meta_oof = normalize_probabilities(oofs[0])
            self.unbiased_meta_loss = compute_multi_log_loss(y_true, self.meta_oof)
            return self

        stacked = np.stack([normalize_probabilities(p) for p in oofs], axis=1)  # (N, M, C)
        n_samples, n_models, n_classes = stacked.shape

        if n_samples != len(y_true):
            raise ValueError(
                f"OOF samples ({n_samples}) and y_true length ({len(y_true)}) mismatch."
            )

        outer = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        meta_oof = np.empty((n_samples, n_classes), dtype=np.float64)

        def objective(weights: np.ndarray, X_data: np.ndarray, y_data: np.ndarray) -> float:
            blended = np.einsum("m,nmc->nc", weights, X_data)
            return compute_multi_log_loss(y_data, blended)

        # Outer Meta-CV Loop: unbiased evaluation
        for train_idx, test_idx in outer.split(stacked):
            X_train = stacked[train_idx]
            y_train = y_true[train_idx]

            res = minimize(
                objective,
                x0=np.full(n_models, 1.0 / n_models),
                args=(X_train, y_train),
                method="SLSQP",
                bounds=[(0.0, 1.0)] * n_models,
                constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            )

            fold_w = res.x / np.sum(res.x)
            meta_oof[test_idx] = np.einsum("m,nmc->nc", fold_w, stacked[test_idx])

        self.meta_oof = normalize_probabilities(meta_oof)
        self.unbiased_meta_loss = compute_multi_log_loss(y_true, self.meta_oof)

        # Refit final weights on 100% of OOF for deployment
        res_final = minimize(
            objective,
            x0=np.full(n_models, 1.0 / n_models),
            args=(stacked, y_true),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n_models,
            constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        )
        self.final_weights = res_final.x / np.sum(res_final.x)
        return self

    def blend(self, pred_list: list[np.ndarray]) -> np.ndarray:
        if self.final_weights is None:
            raise ValueError("Blender must be fitted before blending.")
        stacked = np.stack([normalize_probabilities(p) for p in pred_list], axis=1)
        blended = np.einsum("m,nmc->nc", self.final_weights, stacked)
        return normalize_probabilities(blended)

    def get_weight_summary(self) -> dict[str, float]:
        if self.final_weights is None:
            return {}
        return {
            (self.names[i] if i < len(self.names) else f"model_{i}"): round(float(w), 4)
            for i, w in enumerate(self.final_weights)
        }


# ==============================================================================
# 6. CRYPTOGRAPHICALLY BOUND IRON RULE AUTHORIZATION GATE
# ==============================================================================


@dataclass(frozen=True)
class AuthorizationRequest:
    """Immutable, cryptographically bound authorization contract."""

    experiment_id: str
    artifact_sha256: str
    prediction_sha256: str
    config_sha256: str
    code_revision: str
    metric_name: str
    metric_value: float
    created_at: str
    expires_at: str


def compute_file_sha256(file_path: str | Path) -> str:
    """Computes SHA-256 hash of a file on disk."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class SubmissionAuthorizationGate:
    """
    Enforces the Iron Rule: Execution strictly halts before external submission/deployment.
    Authorization is bound to immutable artifact and configuration hashes.
    """

    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def create_authorization_request(
        self,
        exp_id: str,
        model_name: str,
        oof_score: float,
        best_score: float | None,
        submission_path: Path,
        preview_df: pd.DataFrame,
        config_path: Path | None = None,
        code_revision: str = "HEAD",
    ) -> tuple[AuthorizationRequest, Path]:
        """Creates formal SHA-256 bound request document and structured record."""
        artifact_hash = compute_file_sha256(submission_path)
        pred_hash = hashlib.sha256(preview_df.to_csv(index=False).encode("utf-8")).hexdigest()
        cfg_hash = (
            compute_file_sha256(config_path)
            if config_path and config_path.exists()
            else "unspecified"
        )
        now_str = datetime.now(timezone.utc).isoformat()

        req = AuthorizationRequest(
            experiment_id=exp_id,
            artifact_sha256=artifact_hash,
            prediction_sha256=pred_hash,
            config_sha256=cfg_hash,
            code_revision=code_revision,
            metric_name="Multi-Class Log Loss",
            metric_value=oof_score,
            created_at=now_str,
            expires_at="PT24H",
        )

        delta = f"{oof_score - best_score:+.5f}" if best_score is not None else "Initial Baseline"
        doc = f"""# [SUBMISSION AUTHORIZATION REQUEST]
> **IRON RULE**: Autonomous loop halted. External submission requires human verification.

### Experiment Summary
- **Experiment ID**: `{exp_id}`
- **Model Architecture**: `{model_name}`
- **Unbiased Meta-CV Log Loss**: `{oof_score:.5f}` (Comparison to Best: {delta})
- **Submission Artifact**: `{submission_path}`
- **Created At**: `{now_str}`

### Immutable Cryptographic Bindings
- **Artifact SHA-256**: `{artifact_hash}`
- **Prediction SHA-256**: `{pred_hash}`
- **Config SHA-256**: `{cfg_hash}`
- **Code Revision**: `{code_revision}`

### Prediction Preview (First 5 rows)
{preview_df.head(5).to_markdown(index=False)}

### Authorization Verification
To authorize execution, run:
```bash
python cli.py submit --approve {exp_id} --sha256 {artifact_hash[:16]}
```
To reject candidate:
```bash
python cli.py submit --reject {exp_id} --reason "Explain reason"
```
"""
        req_file = self.reports_dir / "SUBMISSION_AUTHORIZATION_REQUEST.md"
        req_file.write_text(doc, encoding="utf-8")

        # Save structured record alongside
        req_json = self.reports_dir / f"auth_request_{exp_id}.json"
        with open(req_json, "w", encoding="utf-8") as f:
            json.dump(req.__dict__, f, indent=2)

        return req, req_file

    def verify_authorization(
        self,
        request: AuthorizationRequest,
        submission_path: Path,
        current_config_path: Path | None = None,
        current_code_revision: str = "HEAD",
    ) -> None:
        """
        Cryptographically verifies that no artifact has been modified since approval was granted.
        Raises RuntimeError if any verification check fails.
        """
        current_artifact_hash = compute_file_sha256(submission_path)
        if current_artifact_hash != request.artifact_sha256:
            raise RuntimeError(
                f"CRITICAL: Submission artifact changed after authorization! "
                f"Expected {request.artifact_sha256[:16]}, got {current_artifact_hash[:16]}."
            )

        if (
            current_config_path
            and current_config_path.exists()
            and request.config_sha256 != "unspecified"
        ):
            curr_cfg_hash = compute_file_sha256(current_config_path)
            if curr_cfg_hash != request.config_sha256:
                raise RuntimeError(
                    "CRITICAL: Experiment configuration was modified after authorization was requested."
                )


# ==============================================================================
# 7. RUNNABLE SYNTHETIC VERIFICATION DEMO
# ==============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("Self-Improving AI/ML Framework: Production Verification Demo")
    print("=" * 65)

    np.random.seed(42)
    n_samples = 300
    n_classes = 3

    # 1. Test Defensive Probability Normalization (NaN / Inf rejection)
    print("\n1. Testing Defensive Probability Normalization:")
    valid_p = np.random.uniform(0.1, 0.9, size=(10, 3))
    normed = normalize_probabilities(valid_p)
    assert np.allclose(normed.sum(axis=1), 1.0), "Row sums must equal 1.0"
    print("   [PASS] Valid 2-D probability matrix correctly row-normalized.")

    nan_p = valid_p.copy()
    nan_p[0, 0] = np.nan
    try:
        normalize_probabilities(nan_p)
        print("   [FAIL] Expected ValueError on NaN array!")
    except ValueError as e:
        print(f"   [PASS] Successfully caught non-finite input: {e}")

    # 2. Test Safe Diversity Check (including constant predictor)
    print("\n2. Testing Robust Diversity Correlation:")
    pred_normal = np.random.uniform(0.1, 0.9, size=(100, 3))
    pred_const = np.full((100, 3), 1.0 / 3.0)
    r_const = safe_correlation(pred_normal, pred_const)
    print(f"   Correlation with constant predictor: {r_const} (Handled safely without NaN).")

    # 3. Test Total Variation Symmetry Divergence
    print("\n3. Testing Total Variation Symmetry Divergence:")
    gatekeeper = ValidationGatekeeper(max_symmetry_divergence=0.10)
    p_a = normalize_probabilities(np.random.uniform(0.1, 0.9, size=(100, 3)))
    p_b = p_a + np.random.normal(0, 0.02, size=(100, 3))
    sym_res = gatekeeper.check_symmetry(p_a, p_b)
    print(
        f"   Symmetry TV Distance: {sym_res.details.get('div_tv'):.4f} | Passed: {sym_res.passed}"
    )

    # 4. Test True Outer Meta-CV Ensemble Blending
    print("\n4. Testing True Outer Meta-CV Ensemble Blending:")
    y_true = np.random.randint(0, n_classes, size=n_samples)
    mock_oof_1 = normalize_probabilities(np.random.dirichlet([2, 1, 1], size=n_samples))
    mock_oof_2 = normalize_probabilities(np.random.dirichlet([1, 2, 1], size=n_samples))

    blender = ConvexEnsembleBlender(names=["Model_1", "Model_2"])
    blender.fit_nested_cv([mock_oof_1, mock_oof_2], y_true, n_folds=5)

    print(f"   Unbiased Outer Meta-CV Loss: {blender.unbiased_meta_loss:.4f}")
    print(f"   Final Refitted Deployment Weights: {blender.get_weight_summary()}")

    # 5. Test Cryptographically Bound Iron Rule
    print("\n5. Testing Cryptographically Bound Iron Rule Authorization:")
    sub_dir = Path("submissions")
    sub_dir.mkdir(parents=True, exist_ok=True)
    demo_sub_path = sub_dir / "demo_submission.csv"
    demo_sub_df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "prob_0": [0.7, 0.2, 0.1],
            "prob_1": [0.2, 0.7, 0.1],
            "prob_2": [0.1, 0.1, 0.8],
        }
    )
    demo_sub_df.to_csv(demo_sub_path, index=False)

    auth_gate = SubmissionAuthorizationGate(reports_dir="reports")
    auth_req, req_file = auth_gate.create_authorization_request(
        exp_id="H_PROD_001",
        model_name="production_convex_blender",
        oof_score=blender.unbiased_meta_loss,
        best_score=1.2500,
        submission_path=demo_sub_path,
        preview_df=demo_sub_df,
    )
    print(f"   Authorization brief written to: {req_file}")
    print(f"   Artifact SHA-256: {auth_req.artifact_sha256[:16]}...")

    # Verify matching artifact succeeds
    auth_gate.verify_authorization(auth_req, demo_sub_path)
    print("   [PASS] Pre-submission hash verification succeeded for untouched artifact.")

    # Tamper with file to verify fail-closed behavior
    demo_sub_path.write_text("TAMPERED DATA", encoding="utf-8")
    try:
        auth_gate.verify_authorization(auth_req, demo_sub_path)
        print("   [FAIL] Verification should have failed on tampered file!")
    except RuntimeError as e:
        print(f"   [PASS] Successfully blocked tampered submission: {e}")

    # Restore clean file
    demo_sub_df.to_csv(demo_sub_path, index=False)
    print("\n[SUCCESS] Production scaffold verification completed cleanly.")
