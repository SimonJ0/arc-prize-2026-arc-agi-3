"""
Unit tests for AdaptiveHypothesisGenerator.
"""

import pytest
import yaml
from pathlib import Path
from src.core.adaptive import AdaptiveHypothesisGenerator


def test_adaptive_propose_and_register(tmp_path):
    reg_path = tmp_path / "registry.json"
    hyp_path = tmp_path / "hypotheses.yaml"

    # Start with initial hypotheses H001-H004
    initial_hyps = {
        "hypotheses": [
            {"id": "H001", "name": "base", "model_type": "length_prior"},
            {"id": "H002", "name": "lgb", "model_type": "lightgbm"},
        ]
    }
    with open(hyp_path, "w") as f:
        yaml.dump(initial_hyps, f)

    generator = AdaptiveHypothesisGenerator(
        registry_path=str(reg_path),
        hypotheses_path=str(hyp_path),
    )

    hyp = generator.propose_next_hypothesis()
    assert hyp["id"] == "H005"
    assert "dense_semantic_lsa_lightgbm" in hyp["name"]

    # Register it
    success = generator.register_proposed_hypothesis(hyp)
    assert success is True

    # Check that file now has H005
    data = generator.load_hypotheses()
    ids = [h["id"] for h in data["hypotheses"]]
    assert "H005" in ids

    # Second registration should return False (already present)
    success_dup = generator.register_proposed_hypothesis(hyp)
    assert success_dup is False
