"""
Capability Matrix Test 13: Structured Effect Taxonomy (Baseline 3.0 B3.01).
Verifies multi-tier causal impact classification:
- NONE (Δ == 0)
- UI_NOISE (0 < Δ <= 2)
- LOCAL_MUTATION (2 < Δ <= 20)
- STRUCTURAL_MUTATION (20 < Δ <= 100)
- GLOBAL_MUTATION (Δ > 100)
- LEVEL_ADVANCE (win / next level trigger)
- GAME_OVER (terminal fatal state)
And verifies integration with PersistentReasoningState affordance registration.
"""

import pytest
from src.arc_agent.memory.effect_taxonomy import (
    EffectType,
    UI_NOISE_CEIL,
    LOCAL_CEIL,
    STRUCTURAL_CEIL,
    classify_effect,
)
from src.arc_agent.memory.reasoning_state import PersistentReasoningState


def test_classify_effect_terminal():
    """Terminal outcomes take priority over pixel counts."""
    assert classify_effect(diff_count=0, is_lethal=True) == EffectType.GAME_OVER
    assert classify_effect(diff_count=50, state="GAME_OVER") == EffectType.GAME_OVER
    assert classify_effect(diff_count=0, level_advanced=True) == EffectType.LEVEL_ADVANCE
    assert classify_effect(diff_count=10, state="WIN") == EffectType.LEVEL_ADVANCE


def test_classify_effect_pixel_thresholds():
    """Pixel count boundaries partition into discrete mutation tiers."""
    # NONE
    assert classify_effect(diff_count=0) == EffectType.NONE

    # UI_NOISE
    assert classify_effect(diff_count=1) == EffectType.UI_NOISE
    assert classify_effect(diff_count=UI_NOISE_CEIL) == EffectType.UI_NOISE

    # LOCAL_MUTATION
    assert classify_effect(diff_count=UI_NOISE_CEIL + 1) == EffectType.LOCAL_MUTATION
    assert classify_effect(diff_count=LOCAL_CEIL) == EffectType.LOCAL_MUTATION

    # STRUCTURAL_MUTATION
    assert classify_effect(diff_count=LOCAL_CEIL + 1) == EffectType.STRUCTURAL_MUTATION
    assert classify_effect(diff_count=STRUCTURAL_CEIL) == EffectType.STRUCTURAL_MUTATION

    # GLOBAL_MUTATION
    assert classify_effect(diff_count=STRUCTURAL_CEIL + 1) == EffectType.GLOBAL_MUTATION
    assert classify_effect(diff_count=500) == EffectType.GLOBAL_MUTATION


def test_reasoning_state_affordance_integration():
    """PersistentReasoningState uses EffectType to route affordances."""
    state = PersistentReasoningState(game_id="test_game")

    # 1. UI Noise -> inert
    state.register_affordance_result(coord=(10, 10), diff_count=1)
    assert (10, 10) in state.inert_affordances
    assert (10, 10) not in state.active_affordances
    assert state.last_effect_type == EffectType.UI_NOISE

    # 2. Local mutation -> active
    state.register_affordance_result(coord=(5, 32), diff_count=5)
    assert (5, 32) in state.active_affordances
    assert (5, 32) not in state.inert_affordances
    assert state.last_effect_type == EffectType.LOCAL_MUTATION
    assert state.affordance_repeat_count == 1

    # 3. Structural mutation (puzzle rotation) -> active with streak
    state.register_affordance_result(coord=(5, 32), diff_count=45)
    assert (5, 32) in state.active_affordances
    assert state.last_effect_type == EffectType.STRUCTURAL_MUTATION
    assert state.affordance_repeat_count == 2

    # 4. Lethal trap -> lethal
    state.register_affordance_result(coord=(5, 32), diff_count=0, is_lethal=True)
    assert (5, 32) in state.lethal_affordances
    assert (5, 32) not in state.active_affordances
    assert state.last_effect_type == EffectType.GAME_OVER
    assert state.affordance_repeat_count == 0

    # 5. Level reset clears level-scoped affordances
    state.reset_level(new_level=2)
    assert len(state.active_affordances) == 0
    assert len(state.inert_affordances) == 0
    assert len(state.lethal_affordances) == 0
    assert state.last_effect_type is None
