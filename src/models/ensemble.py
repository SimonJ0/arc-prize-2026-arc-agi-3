"""
Optimal Convex Ensemble Blending Engine.
Optimizes multi-class Log Loss directly over model probabilities using SLSQP / Dirichlet weighting.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize

from src.core.metrics import compute_log_loss, normalize_probabilities


class EnsembleBlender:
    """
    Finds optimal weights for combining multiple out-of-fold predictions
    to minimize Kaggle multi-class log loss.
    """

    def __init__(self, names: Optional[List[str]] = None):
        self.names = names or []
        self.weights: Optional[np.ndarray] = None
        self.optimal_log_loss: Optional[float] = None

    def fit(self, oofs: List[np.ndarray], y_true: np.ndarray) -> "EnsembleBlender":
        """
        Fits optimal convex blending weights across M model OOF predictions.
        
        Args:
            oofs: List of M arrays, each of shape (N, 3).
            y_true: True one-hot or target array of shape (N, 3).
        """
        m = len(oofs)
        if m == 0:
            raise ValueError("At least one model OOF is required.")
        if m == 1:
            self.weights = np.array([1.0])
            self.optimal_log_loss = compute_log_loss(y_true, oofs[0])
            return self

        # Stack into (M, N, 3)
        stacked_oofs = np.array([normalize_probabilities(oof) for oof in oofs])

        def loss_func(weights):
            weights = np.asarray(weights)
            # Weighted average across models
            blended = np.tensordot(weights, stacked_oofs, axes=(0, 0))
            blended = normalize_probabilities(blended)
            return compute_log_loss(y_true, blended)

        # Equal weight initialization
        init_weights = np.ones(m) / m
        bounds = [(0.0, 1.0) for _ in range(m)]
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        res = minimize(
            loss_func,
            init_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 300, "ftol": 1e-7},
        )

        self.weights = res.x / np.sum(res.x)
        self.optimal_log_loss = float(res.fun)
        return self

    def fit_with_nested_cv(
        self,
        oofs: List[np.ndarray],
        y_true: np.ndarray,
        n_blend_folds: int = 3,
    ) -> "EnsembleBlender":
        """
        Fits optimal convex blending weights using nested cross-validation
        to prevent in-sample overfitting of weights on OOF predictions.
        
        Args:
            oofs: List of M arrays of shape (N, 3).
            y_true: True targets array (N, 3).
            n_blend_folds: Number of inner folds for weight stability (default: 3).
        """
        m = len(oofs)
        if m <= 1:
            return self.fit(oofs, y_true)

        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_blend_folds, shuffle=True, random_state=42)
        fold_weights = []

        for train_idx, _ in kf.split(y_true):
            inner_oofs = [oof[train_idx] for oof in oofs]
            inner_blender = EnsembleBlender(names=self.names)
            inner_blender.fit(inner_oofs, y_true[train_idx])
            fold_weights.append(inner_blender.weights)

        # Average weights across inner blend folds to regularize
        avg_weights = np.mean(fold_weights, axis=0)
        self.weights = avg_weights / np.sum(avg_weights)

        # Compute out-of-sample evaluated log loss
        blended_full = self.blend(oofs)
        self.optimal_log_loss = compute_log_loss(y_true, blended_full)
        return self

    def blend(self, pred_list: List[np.ndarray]) -> np.ndarray:
        """Blends test predictions using fitted weights."""
        if self.weights is None:
            raise ValueError("Blender must be fitted before blending.")

        stacked = np.array([normalize_probabilities(p) for p in pred_list])
        blended = np.tensordot(self.weights, stacked, axes=(0, 0))
        return normalize_probabilities(blended)

    def get_weight_summary(self) -> Dict[str, float]:
        """Returns readable dictionary of model names and their assigned ensemble weights."""
        if self.weights is None:
            return {}
        summary = {}
        for idx, w in enumerate(self.weights):
            name = self.names[idx] if idx < len(self.names) else f"model_{idx}"
            summary[name] = round(float(w), 4)
        return summary
