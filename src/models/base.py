"""
Abstract Base Predictor for Chatbot Arena Preference Modeling.
Provides standardized interface and built-in Test-Time Augmentation (TTA) symmetry blending.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.core.metrics import normalize_probabilities


class BasePreferencePredictor(ABC):
    """Abstract interface for all competition model architectures."""

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: Any, y: np.ndarray, **kwargs) -> "BasePreferencePredictor":
        """Fits model on training features and integer targets (0: A, 1: B, 2: Tie)."""
        pass

    @abstractmethod
    def predict_proba(self, X: Any) -> np.ndarray:
        """Returns array of shape (N, 3) with predicted probabilities."""
        pass

    def predict_proba_with_tta(
        self,
        X_normal: Any,
        X_swapped: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Executes Test-Time Augmentation (TTA) by evaluating both original
        and response-swapped pairs, then enforcing exact mathematical position symmetry:

            P_sym(win_a)   = 0.5 * (P_norm(win_a) + P_swap(win_b))
            P_sym(win_b)   = 0.5 * (P_norm(win_b) + P_swap(win_a))
            P_sym(win_tie) = 0.5 * (P_norm(win_tie) + P_swap(win_tie))

        Returns:
            Tuple of (p_symmetrized, p_normal, p_swapped)
        """
        p_norm = normalize_probabilities(self.predict_proba(X_normal))
        p_swap = normalize_probabilities(self.predict_proba(X_swapped))

        p_sym = np.zeros_like(p_norm)
        p_sym[:, 0] = 0.5 * (p_norm[:, 0] + p_swap[:, 1])
        p_sym[:, 1] = 0.5 * (p_norm[:, 1] + p_swap[:, 0])
        p_sym[:, 2] = 0.5 * (p_norm[:, 2] + p_swap[:, 2])

        p_sym = normalize_probabilities(p_sym)
        return p_sym, p_norm, p_swap
