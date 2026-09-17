"""
Capability Diagnostic 8: Multiple Candidate Goals & Negative Falsification.
Verifies that when reaching an incorrect candidate goal yields no level advance, it is falsified.
"""

from src.arc_agent.memory.structured_belief import GoalMemory


def test_multiple_goal_falsification():
    gm = GoalMemory()
    distractor = (5, 5)
    true_goal = (12, 12)

    gm.add_candidate(distractor, color=4)
    gm.add_candidate(true_goal, color=9)

    # Reaching distractor twice without advance should falsify it
    gm.record_visit(distractor, level_advanced=False)
    gm.record_visit(distractor, level_advanced=False)

    assert gm.candidate_goals[distractor].falsified is True
    # True goal remains viable
    assert gm.candidate_goals[true_goal].falsified is False
    assert gm.get_best_candidate((0, 0)) == true_goal
