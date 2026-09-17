"""
Capability Diagnostic 12: Prediction-Error Adaptation & Bayesian Revision.
Verifies that prediction errors update Bayesian posteriors and trigger model revision.
"""

from src.arc_agent.world_model.falsification_engine import (
    FalsificationEngine,
    ForwardPrediction,
)
from src.arc_agent.memory.structured_belief import HypothesisMemory


def test_prediction_error_and_model_revision():
    engine = FalsificationEngine()
    hypotheses = HypothesisMemory()

    hypotheses.register(
        hypothesis_id="kinematics_ACTION1",
        category="kinematics",
        description="Action ACTION1 moves UP",
        prior=0.8,
        details={"delta": (-1, 0)},
    )

    # Contradicting observation: Action1 actually moved DOWN (1, 0)
    pred = ForwardPrediction(
        predicted_player_pos=(9, 10),
        predicted_delta=(-1, 0),
        expected_collision=False,
        expected_interactive_trigger=False,
        expected_goal_reach=False,
    )
    err = engine.evaluate_prediction_error(pred, actual_pos=(11, 10), prev_pos=(10, 10))
    engine.update_and_falsify("ACTION1", prev_pos=(10, 10), actual_pos=(11, 10), prediction=pred, error=err, hypotheses=hypotheses)

    # Hypothesis must be falsified due to invariant contradiction
    assert hypotheses.hypotheses["kinematics_ACTION1"].falsified is True
    assert hypotheses.hypotheses["kinematics_ACTION1"].posterior == 0.0
