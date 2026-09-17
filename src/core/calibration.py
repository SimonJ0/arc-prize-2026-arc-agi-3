"""
Post-Hoc Probability Calibration Engine for Multi-Class Log Loss.
Implements Softmax Temperature Scaling over probability / pseudo-logit representations.
"""

import numpy as np
from scipy.optimize import minimize_scalar

from src.core.metrics import compute_log_loss, normalize_probabilities


class TemperatureCalibrator:
    """
    Post-hoc temperature scaling calibrator.
    Optimizes a single temperature parameter T > 0 on out-of-fold predictions
    to minimize Kaggle multi-class log loss by fixing overconfidence or underconfidence.
    """

    def __init__(self, bounds: tuple[float, float] = (0.1, 5.0)):
        self.bounds = bounds
        self.temperature: float = 1.0
        self.is_fitted: bool = False

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        # Numerically stable softmax
        shift = logits - np.max(logits, axis=1, keepdims=True)
        exp_shift = np.exp(shift)
        return exp_shift / exp_shift.sum(axis=1, keepdims=True)

    @staticmethod
    def _probs_to_logits(probs: np.ndarray, eps: float = 1e-15) -> np.ndarray:
        clipped = np.clip(probs, eps, 1.0 - eps)
        return np.log(clipped)

    def fit(self, y_true: np.ndarray, probs: np.ndarray) -> "TemperatureCalibrator":
        """
        Finds the optimal temperature parameter T that minimizes log loss on validation predictions.

        Args:
            y_true: Ground truth target array (N, 3).
            probs: Uncalibrated predicted probabilities (N, 3).
        """
        logits = self._probs_to_logits(probs)

        def objective(t: float) -> float:
            scaled_logits = logits / t
            calibrated_probs = self._softmax(scaled_logits)
            return compute_log_loss(y_true, calibrated_probs)

        res = minimize_scalar(
            objective,
            bounds=self.bounds,
            method="bounded",
            options={"xatol": 1e-4, "maxiter": 100},
        )

        self.temperature = float(res.x)
        self.is_fitted = True
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """
        Applies learned temperature scaling to predicted probabilities.
        """
        if not self.is_fitted:
            return normalize_probabilities(probs)

        logits = self._probs_to_logits(probs)
        scaled_logits = logits / self.temperature
        calibrated = self._softmax(scaled_logits)
        return normalize_probabilities(calibrated)


def optimize_temperature(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Convenience function returning optimal temperature scalar."""
    calibrator = TemperatureCalibrator()
    calibrator.fit(y_true, probs)
    return calibrator.temperature
