"""
Capability Diagnostic 7: Hidden Precondition Tracking.
Verifies that GoalMemory accurately marks preconditions without prematurely discarding candidate targets.
"""

from src.arc_agent.memory.structured_belief import GoalMemory


def test_hidden_precondition_memory():
    gm = GoalMemory()
    target_coord = (10, 10)
    gm.add_candidate(target_coord, color=5)

    # Goal is not yet completed; mark that it requires a key
    gm.mark_precondition(target_coord, key_color=7)

    assert target_coord in gm.candidate_goals
    goal_rec = gm.candidate_goals[target_coord]
    assert goal_rec.requires_precondition is True
    assert goal_rec.precondition_key_color == 7
    # Should not be falsified
    assert goal_rec.falsified is False
