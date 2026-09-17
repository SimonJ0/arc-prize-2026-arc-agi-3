"""
Unit tests for ARC-AGI-3 domain contracts and legality adapter.
"""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from src.arc_agent.legality_adapter import LegalityAdapter
from src.arc_core.contracts import (
    Observation,
)


def test_observation_immutability():
    frame = np.zeros((10, 10), dtype=int)
    obs = Observation(
        frames=(frame,),
        state="NOT_FINISHED",
        available_actions=frozenset({"RESET", "ACTION1"}),
        game_key="test_game",
        level=1,
        action_count=0,
    )
    assert obs.state == "NOT_FINISHED"
    assert "ACTION1" in obs.available_actions
    with pytest.raises(FrozenInstanceError):
        obs.state = "WIN"  # type: ignore[misc]  # Frozen dataclass cannot be mutated


def test_legality_adapter_game_over_forces_reset():
    # Even if agent proposes ACTION1, GAME_OVER must strictly emit RESET
    action, payload = LegalityAdapter.validate_action(
        state="GAME_OVER",
        available_actions={"RESET", "ACTION1"},
        proposed_action="ACTION1",
        proposed_payload={"x": 5, "y": 5},
    )
    assert action == "RESET"
    assert payload == {}


def test_legality_adapter_available_actions_fallback():
    # If proposed action is not in available_actions, fallback to legal action
    action, payload = LegalityAdapter.validate_action(
        state="NOT_FINISHED",
        available_actions={"ACTION2", "ACTION3"},
        proposed_action="ACTION1",  # Not available
    )
    assert action in {"ACTION2", "ACTION3"}


def test_legality_adapter_action6_coordinate_clamping():
    # ACTION6 coordinates must be clamped to [0, 63]
    action, payload = LegalityAdapter.validate_action(
        state="NOT_FINISHED",
        available_actions={"ACTION6"},
        proposed_action="ACTION6",
        proposed_payload={"x": 100, "y": -10},
    )
    assert action == "ACTION6"
    assert payload["x"] == 63
    assert payload["y"] == 0


def test_legality_adapter_strips_coordinates_for_simple_actions():
    # Coordinates must be removed for simple actions
    action, payload = LegalityAdapter.validate_action(
        state="NOT_FINISHED",
        available_actions={"ACTION1"},
        proposed_action="ACTION1",
        proposed_payload={"x": 12, "y": 15},
    )
    assert action == "ACTION1"
    assert "x" not in payload
    assert "y" not in payload
