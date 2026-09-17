"""
Linear Model on TF-IDF Differential Text Representations.
Complements tree-based structural features by capturing high-dimensional lexical cues.
"""

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.core.metrics import normalize_probabilities
from src.models.base import BasePreferencePredictor


class TfidfLogisticPredictor(BasePreferencePredictor):
    """
    Multinomial Logistic Regression trained on TF-IDF difference features:
    TFIDF(Response A) - TFIDF(Response B).
    """

    def __init__(self, C: float = 1.0, max_iter: int = 200, name: str = "tfidf_logistic"):
        super().__init__(name=name)
        self.C = C
        self.max_iter = max_iter
        self.clf = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver="lbfgs",
            random_state=42,
        )

    def fit(self, X: Any, y: np.ndarray, **kwargs) -> "TfidfLogisticPredictor":
        """Fits multinomial logistic regression."""
        self.clf.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """Outputs normalized probabilities."""
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        raw_probs = self.clf.predict_proba(X)
        return normalize_probabilities(raw_probs)
