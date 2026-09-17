"""
Capability Diagnostic 3: Movement Dynamics Induction.
Verifies that the agent correctly deduces action-direction kinematics from observed motion.
"""

from src.arc_agent.world_model.falsification_engine import (
    FalsificationEngine,
    ForwardPrediction,
)
from src.arc_agent.memory.structured_belief import HypothesisMemory


def test_kinematic_movement_induction():
    engine = FalsificationEngine()
    hypotheses = HypothesisMemory()

    # Step: Action1 moved player from (10, 10) to (9, 10) (UP)
    pred = ForwardPrediction(
        predicted_player_pos=(9, 10),
        predicted_delta=(-1, 0),
        expected_collision=False,
        expected_interactive_trigger=False,
        expected_goal_reach=False,
    )
    err = engine.evaluate_prediction_error(pred, actual_pos=(9, 10), prev_pos=(10, 10))
    engine.update_and_falsify("ACTION1", prev_pos=(10, 10), actual_pos=(9, 10), prediction=pred, error=err, hypotheses=hypotheses)

    assert "kinematics_ACTION1" in hypotheses.hypotheses
    hyp = hypotheses.hypotheses["kinematics_ACTION1"]
    assert hyp.details["delta"] == (-1, 0)
    assert hyp.posterior > 0.70
