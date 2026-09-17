"""
Capability Matrix Test 14: Causal Mechanism Ledger (Baseline 3.0 B3.02).
Verifies:
- EntitySignature morphological abstraction
- MechanismMemory transition recording and indexing
- Querying by signature, effect type, active/inert sets
- Level reset retention of invariant signatures
- Integration with StructuredBeliefState
"""

from dataclasses import dataclass
import pytest

from src.arc_agent.memory.effect_taxonomy import EffectType
from src.arc_agent.memory.mechanism_memory import EntitySignature, MechanismMemory, MechanismRecord
from src.arc_agent.memory.structured_belief import StructuredBeliefState


@dataclass
class MockEntity:
    color: int
    size: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float] = (0.0, 0.0)
    cells: tuple[tuple[int, int], ...] = ()


def test_entity_signature_abstraction():
    """EntitySignature maps continuous geometry into invariant discrete buckets."""
    # Tiny button (color 3, size 4, bbox 2x2 -> square solid)
    button = MockEntity(color=3, size=4, bbox=(0, 0, 2, 2))
    sig_btn = EntitySignature.from_entity(button)
    assert sig_btn.color == 3
    assert sig_btn.size_bucket == "tiny"
    assert sig_btn.aspect_ratio_bucket == "square"
    assert sig_btn.solidity_bucket == "solid"

    # Wide bar (color 5, size 30, bbox 2x15 -> wide solid)
    bar = MockEntity(color=5, size=30, bbox=(10, 0, 12, 15))
    sig_bar = EntitySignature.from_entity(bar)
    assert sig_bar.color == 5
    assert sig_bar.size_bucket == "small"
    assert sig_bar.aspect_ratio_bucket == "wide"
    assert sig_bar.solidity_bucket == "solid"

    # Large sparse obstacle (color 10, size 150, bbox 20x20 -> large sparse)
    obs = MockEntity(color=10, size=150, bbox=(0, 0, 20, 20))
    sig_obs = EntitySignature.from_entity(obs)
    assert sig_obs.color == 10
    assert sig_obs.size_bucket == "large"
    assert sig_obs.aspect_ratio_bucket == "square"
    assert sig_obs.solidity_bucket == "sparse"


def test_mechanism_memory_ledger_indexing():
    """MechanismMemory records, indexes, and queries transitions correctly."""
    mem = MechanismMemory()

    sig_btn_l = EntitySignature(color=1, size_bucket="tiny", aspect_ratio_bucket="square", solidity_bucket="solid")
    sig_piece = EntitySignature(color=4, size_bucket="small", aspect_ratio_bucket="square", solidity_bucket="solid")

    # 1. Inert click on puzzle piece
    mem.record_transition(
        step=1,
        action="ACTION6",
        target_coord=(15, 20),
        target_entity=sig_piece,
        effect_type=EffectType.NONE,
        diff_count=0,
    )
    assert sig_piece in mem.get_inert_signatures()
    assert sig_piece not in mem.active_signatures
    assert (15, 20) not in mem.get_active_controllers()

    # 2. Active click on button L
    mem.record_transition(
        step=2,
        action="ACTION6",
        target_coord=(5, 32),
        target_entity=sig_btn_l,
        effect_type=EffectType.STRUCTURAL_MUTATION,
        diff_count=45,
    )
    assert sig_btn_l in mem.active_signatures
    assert (5, 32) in mem.get_active_controllers()

    # 3. Querying
    btn_records = mem.get_by_entity_signature(sig_btn_l)
    assert len(btn_records) == 1
    assert btn_records[0].diff_count == 45

    structural_records = mem.get_by_effect_type(EffectType.STRUCTURAL_MUTATION)
    assert len(structural_records) == 1
    assert structural_records[0].target_coord == (5, 32)


def test_mechanism_memory_level_retention():
    """Level reset wipes coordinate caches but preserves abstract entity signatures."""
    mem = MechanismMemory()
    sig = EntitySignature(color=2, size_bucket="small", aspect_ratio_bucket="square", solidity_bucket="solid")

    mem.record_transition(
        step=1,
        action="ACTION6",
        target_coord=(12, 12),
        target_entity=sig,
        effect_type=EffectType.LOCAL_MUTATION,
        diff_count=10,
    )
    assert (12, 12) in mem.get_active_controllers()
    assert sig in mem.active_signatures

    # Reset level keeping abstract signatures
    mem.reset_level(keep_abstract_signatures=True)
    assert len(mem.get_active_controllers()) == 0  # Coordinates cleared
    assert sig in mem.active_signatures             # Invariant signature preserved
    assert len(mem.records) == 1                   # History intact


def test_structured_belief_state_integration():
    """StructuredBeliefState integrates MechanismMemory."""
    belief = StructuredBeliefState(game_id="test_game")
    assert hasattr(belief, "mechanisms")
    assert isinstance(belief.mechanisms, MechanismMemory)

    sig = EntitySignature(color=7, size_bucket="tiny", aspect_ratio_bucket="wide", solidity_bucket="solid")
    belief.mechanisms.record_transition(
        step=1,
        action="ACTION6",
        target_coord=(3, 3),
        target_entity=sig,
        effect_type=EffectType.UI_NOISE,
        diff_count=1,
    )
    assert sig in belief.mechanisms.get_inert_signatures()

    belief.reset_level(new_level=2)
    assert belief.current_level == 2
    assert sig in belief.mechanisms.get_inert_signatures()
