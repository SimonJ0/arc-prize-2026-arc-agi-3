"""
Unit tests for SelfDrivingResearchLoop: state persistence, fold-bagging, and resumability.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from src.core.loop import SelfDrivingResearchLoop
from src.models.length_prior import LengthPriorPredictor
from src.features.extractor import FeatureExtractor


@pytest.fixture
def mock_loop(tmp_path):
    loop = SelfDrivingResearchLoop(config_path="configs/default_config.yaml")
    loop.experiments_dir = tmp_path / "experiments"
    loop.experiments_dir.mkdir(parents=True, exist_ok=True)
    loop.state_path = loop.experiments_dir / "ensemble_state.npz"
    loop.registry_path = loop.experiments_dir / "registry.json"
    loop.ensemble_oofs = []
    loop.ensemble_names = []
    loop.registry = {"experiments": [], "current_best": None}
    return loop


def test_ensemble_state_save_and_load(mock_loop):
    oof1 = np.array([[0.5, 0.3, 0.2], [0.3, 0.4, 0.3]])
    oof2 = np.array([[0.4, 0.4, 0.2], [0.2, 0.5, 0.3]])
    mock_loop.ensemble_oofs = [oof1, oof2]
    mock_loop.ensemble_names = ["Model_A", "Model_B"]

    mock_loop._save_ensemble_state()
    assert mock_loop.state_path.exists()

    # Clear and restore
    mock_loop.ensemble_oofs = []
    mock_loop.ensemble_names = []
    mock_loop._load_ensemble_state()

    assert len(mock_loop.ensemble_oofs) == 2
    assert mock_loop.ensemble_names == ["Model_A", "Model_B"]
    np.testing.assert_allclose(mock_loop.ensemble_oofs[0], oof1)
    np.testing.assert_allclose(mock_loop.ensemble_oofs[1], oof2)


def test_predict_test_bagged_length_prior(mock_loop):
    df_test = pd.DataFrame({
        "id": [1, 2, 3],
        "prompt": ["hello", "how are you", "write code"],
        "response_a": ["short", "very very very very long text", "a"],
        "response_b": ["longer text here", "short", "b"],
    })

    # Create 3 fold models
    extractor = FeatureExtractor()
    X_test = extractor.extract_features(df_test)
    X_test_swap = extractor.extract_swapped_features(df_test)

    m1 = LengthPriorPredictor(beta=1.5)
    m2 = LengthPriorPredictor(beta=1.6)
    m3 = LengthPriorPredictor(beta=1.7)
    # Fit dummy
    y_dummy = np.array([0, 1, 2])
    m1.fit(X_test, y_dummy)
    m2.fit(X_test, y_dummy)
    m3.fit(X_test, y_dummy)

    mock_loop.trained_models["H001"] = [m1, m2, m3]

    preds = mock_loop._predict_test_bagged(
        exp_id="H001",
        model_type="length_prior",
        params={"beta": 1.6},
        df_train=df_test,
        df_test=df_test,
    )

    assert preds.shape == (3, 3)
    np.testing.assert_allclose(preds.sum(axis=1), np.ones(3), rtol=1e-5)
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 1.0)
