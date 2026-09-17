"""
Capability Matrix Test 18: Mechanism vs Goal Confidence Separation (Baseline 3.0 B3.06).
Verifies:
- Separation between causal certainty ("I know what X does") and goal relevance ("X helps win")
- Prevention of exploitation on active but non-goal-relevant controllers
- Authorization of exploitation on verified goal-advancing mechanisms
- Immediate disqualification of lethal mechanisms
"""

import pytest
from src.arc_agent.memory.effect_taxonomy import EffectType
from src.arc_agent.memory.mechanism_memory import EntitySignature, MechanismHypothesis, MechanismMemory


def test_confidence_separation_active_vs_useful():
    """Active mechanism with zero goal relevance is prevented from being exploited repeatedly."""
    sig = EntitySignature(color=3, size_bucket="small", aspect_ratio_bucket="square", solidity_bucket="solid")
    hyp = MechanismHypothesis(
        mechanism_id="mech_test",
        target_coord=(10, 10),
        entity_signature=sig,
        dominant_effect=EffectType.LOCAL_MUTATION,
    )

    # 4 consecutive clicks produce local color flicker but NO structural board change or level advance
    for _ in range(4):
        hyp.update(effect=EffectType.LOCAL_MUTATION, level_advanced=False)

    # Causal confidence is high: 5 / 6 = 0.833
    assert hyp.causal_confidence > 0.8
    # Goal relevance is low: 1 / 6 = 0.167
    assert hyp.goal_relevance < 0.25
    # Must NOT be authorized for blind exploitation
    assert not hyp.can_reliably_exploit()


def test_confidence_separation_goal_relevant():
    """Mechanism producing structural board mutations or level advance is authorized for exploitation."""
    sig = EntitySignature(color=1, size_bucket="tiny", aspect_ratio_bucket="square", solidity_bucket="solid")
    hyp = MechanismHypothesis(
        mechanism_id="mech_rotator",
        target_coord=(5, 32),
        entity_signature=sig,
        dominant_effect=EffectType.STRUCTURAL_MUTATION,
    )

    # 3 consecutive clicks rotate the board (structural mutation)
    for _ in range(3):
        hyp.update(effect=EffectType.STRUCTURAL_MUTATION, level_advanced=False)

    assert hyp.causal_confidence > 0.75
    assert hyp.goal_relevance > 0.75
    assert hyp.can_reliably_exploit()


def test_confidence_separation_lethal_disqualification():
    """Lethal interactions are permanently barred from exploitation."""
    hyp = MechanismHypothesis(
        mechanism_id="mech_trap",
        target_coord=(15, 15),
        entity_signature=None,
        dominant_effect=EffectType.GLOBAL_MUTATION,
    )

    hyp.update(effect=EffectType.GAME_OVER, level_advanced=False, is_lethal=True)
    assert hyp.is_lethal
    assert not hyp.can_reliably_exploit()
