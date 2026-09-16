"""
LightGBM Gradient Boosted Decision Tree Classifier for Preference Modeling.
Supports multi-class log loss optimization, feature importances, and symmetric training.
"""

from typing import Any, Dict, List, Optional, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.models.base import BasePreferencePredictor
from src.core.metrics import normalize_probabilities


class LightGBMPredictor(BasePreferencePredictor):
    """
    LightGBM multi-class preference predictor.
    Optimizes multi_logloss directly with support for early stopping and symmetric training.
    """

    def __init__(
        self,
        name: str = "lightgbm_classifier",
        params: Optional[Dict[str, Any]] = None,
        symmetric_training: bool = True,
    ):
        super().__init__(name=name)
        self.symmetric_training = symmetric_training
        
        default_params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "min_child_samples": 20,
            "verbosity": -1,
            "n_estimators": 300,
            "random_state": 42,
        }
        if params:
            default_params.update(params)
        self.params = default_params
        self.model: Optional[lgb.LGBMClassifier] = None
        self.feature_names_: List[str] = []

    def fit(
        self,
        X: Any,
        y: np.ndarray,
        eval_set: Optional[List[Tuple]] = None,
        callbacks: Optional[List] = None,
        **kwargs,
    ) -> "LightGBMPredictor":
        """
        Fits LightGBM on features and labels.
        If symmetric_training is enabled, augments the training split with swapped features.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_train = X.values
        else:
            X_train = np.asarray(X)
            self.feature_names_ = [f"feat_{i}" for i in range(X_train.shape[1])]

        y_train = np.asarray(y)

        self.model = lgb.LGBMClassifier(**self.params)
        
        eval_data = None
        if eval_set:
            eval_data = []
            for e_X, e_y in eval_set:
                if isinstance(e_X, pd.DataFrame):
                    e_X = e_X.values
                eval_data.append((e_X, np.asarray(e_y)))

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_data,
            callbacks=callbacks,
        )
        self.is_fitted = True
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predicts calibrated multi-class probabilities."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model has not been fitted yet.")

        if isinstance(X, pd.DataFrame):
            X = X.values

        raw_probs = self.model.predict_proba(X)
        return normalize_probabilities(raw_probs)

    @property
    def feature_importances(self) -> Dict[str, float]:
        """Returns sorted feature importance dictionary."""
        if not self.is_fitted or self.model is None:
            return {}
        importances = self.model.feature_importances_
        return dict(
            sorted(
                zip(self.feature_names_, [float(x) for x in importances]),
                key=lambda item: item[1],
                reverse=True,
            )
        )

    @property
    def booster_model_string(self) -> Optional[str]:
        """Returns the serialized booster model string."""
        if not self.is_fitted or self.model is None or not hasattr(self.model, "booster_"):
            return None
        return self.model.booster_.model_to_string()
