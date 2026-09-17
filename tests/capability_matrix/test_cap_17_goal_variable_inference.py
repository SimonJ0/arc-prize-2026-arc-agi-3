"""
Capability Matrix Test 17: Goal-Variable Inference & Causal Relevance (Baseline 3.0 B3.05).
Verifies:
- Tracking candidate latent goal variables
- Bayesian updating of P(goal | variable) via Laplace-smoothed progress correlation
- Arbitration of top goal-relevant variable over confounding or inert variables
"""

import pytest
from src.arc_agent.memory.structured_belief import GoalMemory, GoalVariableHypothesis


def test_goal_variable_hypothesis_laplace_smoothing():
    """Posterior probability calculates Laplace-smoothed P(goal | variable)."""
    hyp = GoalVariableHypothesis(variable_name="rotation_index")
    assert hyp.posterior == 0.5  # Prior with 0 observations

    hyp.observations = 10
    hyp.progress_correlations = 8
    # (8 + 1) / (10 + 2) = 9/12 = 0.75
    assert pytest.approx(hyp.posterior, 0.001) == 0.75


def test_goal_memory_variable_arbitration():
    """GoalMemory discriminates truly goal-relevant variables from spurious/counter-productive variables."""
    memory = GoalMemory()

    # Variable 1: Board Rotation (correlated with progress)
    for _ in range(5):
        memory.record_variable_transition(var_name="board_rotation", progress_occurred=True)

    # Variable 2: Background Flicker (observed but uncorrelated with progress)
    for _ in range(5):
        memory.record_variable_transition(var_name="bg_flicker", progress_occurred=False)

    # Top goal variable must be board_rotation
    top_var = memory.get_top_goal_variable()
    assert top_var is not None
    assert top_var.variable_name == "board_rotation"
    assert top_var.confidence > 0.8
    assert memory.variable_hypotheses["bg_flicker"].confidence < 0.2
