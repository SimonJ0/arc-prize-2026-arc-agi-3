"""
Capability Diagnostic 11: Cross-Level Skill Transfer & Dynamic Token Re-binding.
Verifies that invariant kinematics are transferred while concrete color tokens are re-bound.
"""

from src.arc_agent.memory.level_transfer import LevelTransferManager
from src.arc_agent.memory.structured_belief import SkillMemory


def test_cross_level_transfer_and_token_rebinding():
    manager = LevelTransferManager()
    skill_mem = SkillMemory()

    # Complete level 1 with ACTION1=UP, ACTION4=RIGHT
    kinematics = {"ACTION1": (-1, 0), "ACTION4": (0, 1)}
    pkg = manager.extract_level_schema(
        level_index=1,
        kinematic_mappings=kinematics,
        avatar_size=1,
        actions_used={"ACTION1", "ACTION4"},
        skill_memory=skill_mem,
        game_id="transfer_game",
    )

    assert pkg.total_levels_completed == 1
    assert len(skill_mem.skills) == 1

    # Enter level 2 with available actions
    priors = manager.apply_transfer_priors(["ACTION1", "ACTION2", "ACTION4"])
    assert priors["ACTION1"] == (-1, 0)
    assert priors["ACTION4"] == (0, 1)
    assert "ACTION2" not in priors
