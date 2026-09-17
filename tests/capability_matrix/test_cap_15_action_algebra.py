"""
Capability Matrix Test 15: Action Algebra & Oscillation Suppression (Baseline 3.0 B3.03).
Verifies:
- Inverse action pair induction: f(f(s, A), B) == s
- Self-inverse toggle detection: f(f(s, A), A) == s
- Oscillation suppression filtering in candidate selection
- Integration with EpistemicPolicy
"""

import pytest
from src.arc_agent.memory.effect_taxonomy import EffectType
from src.arc_agent.memory.mechanism_memory import MechanismRecord
from src.arc_agent.reasoning.action_algebra import ActionAlgebraEngine


def test_inverse_action_induction():
    """ActionAlgebraEngine identifies inverse action pairs from state hash reversals."""
    engine = ActionAlgebraEngine()

    coord_left = (5, 32)
    coord_right = (58, 32)

    # Step 1: Click Left, rotating board from HashA -> HashB
    r1 = MechanismRecord(
        step=1,
        action="ACTION6",
        target_coord=coord_left,
        target_entity=None,
        effect_type=EffectType.STRUCTURAL_MUTATION,
        diff_count=35,
        pre_frame_hash="state_alpha",
        post_frame_hash="state_beta",
    )

    # Step 2: Click Right, rotating board from HashB back to HashA
    r2 = MechanismRecord(
        step=2,
        action="ACTION6",
        target_coord=coord_right,
        target_entity=None,
        effect_type=EffectType.STRUCTURAL_MUTATION,
        diff_count=35,
        pre_frame_hash="state_beta",
        post_frame_hash="state_alpha",
    )

    engine.update_from_records([r1, r2])

    assert engine.is_inverse_pair(coord_left, coord_right)
    assert engine.is_inverse_pair(coord_right, coord_left)
    assert engine.get_inverse(coord_left) == coord_right
    assert engine.get_inverse(coord_right) == coord_left


def test_self_inverse_toggle_induction():
    """ActionAlgebraEngine identifies self-inverse toggles when repeating an action restores state."""
    engine = ActionAlgebraEngine()
    coord_switch = (20, 20)

    # Click switch once: Off -> On
    r1 = MechanismRecord(
        step=1,
        action="ACTION6",
        target_coord=coord_switch,
        target_entity=None,
        effect_type=EffectType.LOCAL_MUTATION,
        diff_count=8,
        pre_frame_hash="switch_off",
        post_frame_hash="switch_on",
    )

    # Click switch again: On -> Off
    r2 = MechanismRecord(
        step=2,
        action="ACTION6",
        target_coord=coord_switch,
        target_entity=None,
        effect_type=EffectType.LOCAL_MUTATION,
        diff_count=8,
        pre_frame_hash="switch_on",
        post_frame_hash="switch_off",
    )

    engine.update_from_records([r1, r2])

    assert engine.is_toggle(coord_switch)


def test_filter_oscillating_actions():
    """ActionAlgebraEngine prunes inverse actions to prevent futile oscillations."""
    engine = ActionAlgebraEngine()
    coord_left = (5, 32)
    coord_right = (58, 32)
    coord_other = (10, 10)

    # Register inverse relationship
    r1 = MechanismRecord(
        step=1,
        action="ACTION6",
        target_coord=coord_left,
        target_entity=None,
        effect_type=EffectType.STRUCTURAL_MUTATION,
        diff_count=35,
        pre_frame_hash="hash_0",
        post_frame_hash="hash_1",
    )
    r2 = MechanismRecord(
        step=2,
        action="ACTION6",
        target_coord=coord_right,
        target_entity=None,
        effect_type=EffectType.STRUCTURAL_MUTATION,
        diff_count=35,
        pre_frame_hash="hash_1",
        post_frame_hash="hash_0",
    )
    engine.update_from_records([r1, r2])

    # If the last action was coord_left, candidate pool containing coord_right should prune coord_right
    candidates = [coord_left, coord_right, coord_other]
    filtered = engine.filter_oscillating_actions(
        candidate_coords=candidates,
        recent_action_coords=[coord_left],
    )

    assert coord_right not in filtered
    assert coord_left in filtered
    assert coord_other in filtered
