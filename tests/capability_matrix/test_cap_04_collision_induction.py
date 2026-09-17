"""
Capability Diagnostic 4: Collision Barrier Induction.
Verifies that stationary moves against expected motion deduce impassable barriers.
"""

from src.arc_agent.world_model.falsification_engine import (
    FalsificationEngine,
    ForwardPrediction,
)
from src.arc_agent.memory.structured_belief import HypothesisMemory


def test_collision_barrier_induction():
    engine = FalsificationEngine()
    hypotheses = HypothesisMemory()

    # Step: Action1 attempted to move UP to (9, 10), but avatar remained at (10, 10)
    pred = ForwardPrediction(
        predicted_player_pos=(9, 10),
        predicted_delta=(-1, 0),
        expected_collision=False,
        expected_interactive_trigger=False,
        expected_goal_reach=False,
    )
    err = engine.evaluate_prediction_error(pred, actual_pos=(10, 10), prev_pos=(10, 10))
    engine.update_and_falsify("ACTION1", prev_pos=(10, 10), actual_pos=(10, 10), prediction=pred, error=err, hypotheses=hypotheses)

    assert (9, 10) in engine.learned_barriers
