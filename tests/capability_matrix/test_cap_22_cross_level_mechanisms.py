"""
Capability Matrix Test 22: Cross-Level Mechanism Transfer (Baseline 3.0 B3.10).
Verifies:
- Transfer of abstract entity mechanism priors (inert / active signatures) across levels
- Reset of concrete coordinates between levels to avoid negative spatial transfer
- Immediate pruning of known-inert equivalence classes in Level 2 without re-probing
"""

import pytest
from src.arc_agent.memory.effect_taxonomy import EffectType
from src.arc_agent.memory.level_transfer import LevelTransferManager
from src.arc_agent.memory.mechanism_memory import EntitySignature, MechanismMemory
from src.arc_agent.memory.structured_belief import SkillMemory


def test_cross_level_mechanism_prior_transfer():
    """LevelTransferManager preserves invariant entity signatures while clearing concrete coordinates."""
    transfer_mgr = LevelTransferManager()
    skill_mem = SkillMemory()

    # Level 1 Mechanism Memory
    lvl1_mem = MechanismMemory()
    sig_puzzle = EntitySignature(color=4, size_bucket="small", aspect_ratio_bucket="square", solidity_bucket="solid")
    sig_button = EntitySignature(color=1, size_bucket="tiny", aspect_ratio_bucket="square", solidity_bucket="solid")

    # Learn that puzzle piece is inert and button is active
    lvl1_mem.record_transition(
        step=1,
        action="ACTION6",
        target_coord=(10, 10),
        target_entity=sig_puzzle,
        effect_type=EffectType.NONE,
        diff_count=0,
    )
    lvl1_mem.record_transition(
        step=2,
        action="ACTION6",
        target_coord=(32, 5),
        target_entity=sig_button,
        effect_type=EffectType.STRUCTURAL_MUTATION,
        diff_count=35,
    )

    # 1. Extract schema at end of Level 1
    pkg = transfer_mgr.extract_level_schema(
        level_index=1,
        kinematic_mappings={},
        avatar_size=1,
        actions_used={"ACTION6"},
        skill_memory=skill_mem,
        game_id="test_game",
        mechanism_memory=lvl1_mem,
    )

    assert sig_puzzle in pkg.transferred_inert_signatures
    assert sig_button in pkg.transferred_active_signatures

    # 2. Start Level 2: Fresh mechanism memory
    lvl2_mem = MechanismMemory()
    assert len(lvl2_mem.get_inert_signatures()) == 0
    assert len(lvl2_mem.get_active_controllers()) == 0

    # 3. Apply transfer priors
    transfer_mgr.apply_mechanism_priors(lvl2_mem)

    # Invariant signatures transferred!
    assert sig_puzzle in lvl2_mem.get_inert_signatures()
    assert sig_button in lvl2_mem.active_signatures
    # Concrete coordinates are clean for new level
    assert len(lvl2_mem.get_active_controllers()) == 0
