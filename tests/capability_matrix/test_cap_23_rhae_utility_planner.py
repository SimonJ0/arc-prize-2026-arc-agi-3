"""
Capability Matrix Test 23: RHAE-Aware Action Utility Planner (Baseline 3.0 B3.11).
Verifies:
- Utility ranking formula U(a) = αG + βI + γC - λK - μR
- Prioritization of high-information and goal-relevant entities
- Severe penalization of lethal mechanisms and death coordinates
"""

from dataclasses import dataclass
import pytest

from src.arc_agent.memory.effect_taxonomy import EffectType
from src.arc_agent.memory.mechanism_memory import EntitySignature, MechanismMemory
from src.arc_agent.memory.reasoning_state import PersistentReasoningState
from src.arc_agent.memory.structured_belief import StructuredBeliefState
from src.arc_agent.planning.epistemic_policy import EpistemicPolicy


@dataclass
class MockEntity:
    centroid: tuple[float, float]
    size: int = 4
    color: int = 1


def test_rhae_utility_scoring():
    """_compute_action_utility balances goal relevance, epistemic value, and lethal risk."""
    policy = EpistemicPolicy()
    belief = StructuredBeliefState(game_id="test_game")
    reasoning_state = PersistentReasoningState(game_id="test_game")

    coord_goal = (32, 5)
    coord_inert = (10, 10)
    coord_lethal = (20, 20)

    # 1. Goal-advancing mechanism: structural mutations observed
    sig_btn = EntitySignature(color=1, size_bucket="tiny", aspect_ratio_bucket="square", solidity_bucket="solid")
    belief.mechanisms.record_transition(
        step=1,
        action="ACTION6",
        target_coord=coord_goal,
        target_entity=sig_btn,
        effect_type=EffectType.STRUCTURAL_MUTATION,
        diff_count=35,
    )

    # 2. Lethal hazard: recorded death coordinate
    reasoning_state.record_death(coord_lethal, fatal_color=None)

    # Compute utilities
    u_goal = policy._compute_action_utility(
        coord=coord_goal,
        entity=MockEntity(centroid=(32.0, 5.0)),
        structured_belief=belief,
        reasoning_state=reasoning_state,
    )

    u_unvisited = policy._compute_action_utility(
        coord=(40, 40),
        entity=MockEntity(centroid=(40.0, 40.0)),
        structured_belief=belief,
        reasoning_state=reasoning_state,
    )

    u_lethal = policy._compute_action_utility(
        coord=coord_lethal,
        entity=MockEntity(centroid=(20.0, 20.0)),
        structured_belief=belief,
        reasoning_state=reasoning_state,
    )

    # Goal mechanism utility must exceed unvisited exploration utility
    assert u_goal > u_unvisited
    # Lethal hazard utility must be deeply negative
    assert u_lethal < 0.0
    assert u_goal > u_lethal
