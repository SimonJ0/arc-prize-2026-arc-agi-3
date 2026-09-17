"""
Structured Effect Taxonomy for ARC-AGI-3 (Baseline 3.0).
Replaces coarse binary delta threshold (diff_count > 2) with a rich 8-class taxonomy:
  NONE, UI_NOISE, LOCAL_MUTATION, STRUCTURAL_MUTATION, GLOBAL_MUTATION,
  DELAYED_EFFECT, LEVEL_ADVANCE, GAME_OVER.
"""

from __future__ import annotations

from enum import Enum


class EffectType(str, Enum):
    """Classification of state transition consequences."""

    NONE = "NONE"                          # Δ == 0: no observable effect
    UI_NOISE = "UI_NOISE"                  # 0 < Δ <= 2: counter tick, animation flicker
    LOCAL_MUTATION = "LOCAL_MUTATION"      # 2 < Δ <= 20: single object/handle changed
    STRUCTURAL_MUTATION = "STRUCTURAL"     # 20 < Δ <= 100: board geometry / puzzle rotation
    GLOBAL_MUTATION = "GLOBAL_MUTATION"    # Δ > 100: massive multi-object transformation / clear
    DELAYED_EFFECT = "DELAYED_EFFECT"      # Δ == 0 immediately, consequence manifests later
    LEVEL_ADVANCE = "LEVEL_ADVANCE"        # Goal condition met, advancing level
    GAME_OVER = "GAME_OVER"               # Terminal failure, lethal trap, or collision quota exceeded


# Threshold boundaries
UI_NOISE_CEIL: int = 2
LOCAL_CEIL: int = 20
STRUCTURAL_CEIL: int = 100


def classify_effect(
    diff_count: int,
    state: str = "NOT_FINISHED",
    level_advanced: bool = False,
    is_lethal: bool = False,
    prev_diff: int = 0,
) -> EffectType:
    """
    Categorizes the observable impact of an action into EffectType.

    Args:
        diff_count: Number of pixels altered between pre- and post-action observations.
        state: Environment state string ("NOT_FINISHED", "WIN", "GAME_OVER", etc.).
        level_advanced: True if this action triggered level advancement.
        is_lethal: True if the action triggered a lethal obstacle penalty or fatal reset.
        prev_diff: Diff count from previous step for delayed effect correlation.

    Returns:
        EffectType enumeration member.
    """
    # 1. Terminal / Fatal Outcomes
    if is_lethal or state in ("GAME_OVER", "GameState.GAME_OVER"):
        return EffectType.GAME_OVER

    # 2. Level Advancement
    if level_advanced or state in ("WIN", "GameState.WIN"):
        return EffectType.LEVEL_ADVANCE

    # 3. Observable pixel delta classification
    if diff_count == 0:
        return EffectType.NONE

    if diff_count <= UI_NOISE_CEIL:
        return EffectType.UI_NOISE

    if diff_count <= LOCAL_CEIL:
        return EffectType.LOCAL_MUTATION

    if diff_count <= STRUCTURAL_CEIL:
        return EffectType.STRUCTURAL_MUTATION

    return EffectType.GLOBAL_MUTATION
