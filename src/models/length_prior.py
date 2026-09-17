"""
Length-Bias and Class Frequency Prior Baseline.
Models Chatbot Arena preference strictly using empirical class frequencies
and logistic length-differential scaling.
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

from src.core.metrics import normalize_probabilities
from src.models.base import BasePreferencePredictor


class LengthPriorPredictor(BasePreferencePredictor):
    """
    Baseline model demonstrating length/verbosity bias:
    - Base tie probability is learned from class frequencies.
    - Relative advantage between A and B is modeled as a sigmoid function
      of the normalized length ratio (len_a - len_b) / (len_a + len_b).
    """

    def __init__(self, beta: float = 1.5, name: str = "length_prior_baseline"):
        super().__init__(name=name)
        self.beta = beta
        self.p_tie_prior = 0.31
        self.p_a_prior = 0.345
        self.p_b_prior = 0.345

    def fit(self, X: Any, y: np.ndarray, **kwargs) -> "LengthPriorPredictor":
        """Computes empirical priors from target distribution."""
        len(y)
        self.p_a_prior = float(np.mean(y == 0))
        self.p_b_prior = float(np.mean(y == 1))
        self.p_tie_prior = float(np.mean(y == 2))
        self.is_fitted = True
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """
        Predicts probabilities from length_ratio feature:
        If length_ratio > 0, A has higher probability than B, and vice versa.
        """
        if isinstance(X, pd.DataFrame) and "char_ratio" in X.columns:
            char_ratios = X["char_ratio"].values
        elif isinstance(X, np.ndarray):
            # Assume first column or diff column
            char_ratios = np.clip(X[:, 0], -1.0, 1.0)
        else:
            char_ratios = np.zeros(len(X))

        # Sigmoid shift: 0 ratio gives 0.5; positive gives > 0.5
        sig = expit(self.beta * char_ratios)

        # Non-tie probability budget
        rem_budget = 1.0 - self.p_tie_prior

        p_a = rem_budget * sig
        p_b = rem_budget * (1.0 - sig)
        p_tie = np.full_like(p_a, self.p_tie_prior)

        preds = np.column_stack([p_a, p_b, p_tie])
        return normalize_probabilities(preds)
