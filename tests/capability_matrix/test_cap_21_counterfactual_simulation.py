"""
Capability Matrix Test 21: Counterfactual Simulation & Plan Simplification (Baseline 3.0 B3.09).
Verifies:
- Elimination of cancelling inverse action pairs before execution
- Elimination of self-cancelling toggle sequences
- Invariance of genuine unidirectional progression plans
"""

import pytest
from src.arc_agent.memory.effect_taxonomy import EffectType
from src.arc_agent.memory.mechanism_memory import MechanismRecord
from src.arc_agent.reasoning.action_algebra import ActionAlgebraEngine


def test_counterfactual_plan_simplification():
    """ActionAlgebraEngine simplifies candidate plans using learned group dynamics."""
    engine = ActionAlgebraEngine()

    coord_left = (5, 32)
    coord_right = (58, 32)
    coord_toggle = (20, 20)
    coord_other = (10, 10)

    # Teach inverse pair (Left, Right)
    r1 = MechanismRecord(
        step=1,
        action="ACTION6",
        target_coord=coord_left,
        target_entity=None,
        effect_type=EffectType.STRUCTURAL_MUTATION,
        diff_count=35,
        pre_frame_hash="h0",
        post_frame_hash="h1",
    )
    r2 = MechanismRecord(
        step=2,
        action="ACTION6",
        target_coord=coord_right,
        target_entity=None,
        effect_type=EffectType.STRUCTURAL_MUTATION,
        diff_count=35,
        pre_frame_hash="h1",
        post_frame_hash="h0",
    )

    # Teach toggle (Switch)
    r3 = MechanismRecord(
        step=3,
        action="ACTION6",
        target_coord=coord_toggle,
        target_entity=None,
        effect_type=EffectType.LOCAL_MUTATION,
        diff_count=5,
        pre_frame_hash="s0",
        post_frame_hash="s1",
    )
    r4 = MechanismRecord(
        step=4,
        action="ACTION6",
        target_coord=coord_toggle,
        target_entity=None,
        effect_type=EffectType.LOCAL_MUTATION,
        diff_count=5,
        pre_frame_hash="s1",
        post_frame_hash="s0",
    )

    engine.update_from_records([r1, r2, r3, r4])

    # 1. Simple inverse cancellation: [Left, Right] -> []
    plan1 = [coord_left, coord_right]
    assert engine.simplify_plan(plan1) == []

    # 2. Toggle cancellation: [Toggle, Toggle] -> []
    plan2 = [coord_toggle, coord_toggle]
    assert engine.simplify_plan(plan2) == []

    # 3. Complex cancellation: [Left, Left, Right, Left] -> [Left, Left]
    # (Left followed by Right cancels out, leaving Left + Left)
    plan3 = [coord_left, coord_left, coord_right, coord_left]
    assert engine.simplify_plan(plan3) == [coord_left, coord_left]

    # 4. Pure additive progress preserved: [Left, Left, Left] -> [Left, Left, Left]
    plan4 = [coord_left, coord_left, coord_left]
    assert engine.simplify_plan(plan4) == [coord_left, coord_left, coord_left]
