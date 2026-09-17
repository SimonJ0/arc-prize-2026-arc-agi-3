"""
Capability Matrix Test 24: Deadlock Classification for Invalid Interactions (Baseline 3.0 B3.12).
Verifies:
- Detection of DeadlockType.TYPE_3_INVALID_INTERACTION when repeatedly clicking inert coordinates
- Recovery and stall counter resetting when an active mutation is achieved
"""

import pytest
from src.arc_agent.planning.deadlock_taxonomy import DeadlockTaxonomyEngine, DeadlockType


def test_type_3_invalid_interaction_deadlock():
    """DeadlockTaxonomyEngine flags TYPE_3_INVALID_INTERACTION upon repeated inert ACTION6 clicks."""
    engine = DeadlockTaxonomyEngine()

    coord = (15, 20)

    # Click 1: Inert (diff=0)
    d1 = engine.record_step(
        current_pos=None,
        action="ACTION6",
        state="NOT_FINISHED",
        active_goal=None,
        diff_count=0,
        target_coord=coord,
    )
    assert d1 is None

    # Click 2: Inert (diff=1)
    d2 = engine.record_step(
        current_pos=None,
        action="ACTION6",
        state="NOT_FINISHED",
        active_goal=None,
        diff_count=1,
        target_coord=coord,
    )
    assert d2 is None

    # Click 3: Inert (diff=0) -> DEADLOCK TRIGGERED
    d3 = engine.record_step(
        current_pos=None,
        action="ACTION6",
        state="NOT_FINISHED",
        active_goal=None,
        diff_count=0,
        target_coord=coord,
    )
    assert d3 is not None
    assert d3.deadlock_type == DeadlockType.TYPE_3_INVALID_INTERACTION
    assert d3.affected_coord == coord

    # Click 4: Active mutation (diff=35) -> resets stall counter
    d4 = engine.record_step(
        current_pos=None,
        action="ACTION6",
        state="NOT_FINISHED",
        active_goal=None,
        diff_count=35,
        target_coord=(32, 5),
    )
    assert d4 is None
    assert engine.action_stall_counter.get("ACTION6_INERT", 0) == 0
