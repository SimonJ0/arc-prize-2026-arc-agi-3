"""
Capability Matrix Test 20: Symbolic Compression & Astra-Style Shorthand (Baseline 3.0 B3.08).
Verifies:
- High information density encoding of the causal ledger
- Extraction of active controllers, effect types, and signature counts into compact shorthand
"""

import pytest
from src.arc_agent.memory.effect_taxonomy import EffectType
from src.arc_agent.memory.mechanism_memory import EntitySignature, MechanismMemory


def test_symbolic_compression_conciseness():
    """compress_to_symbolic encodes causal ledger state into dense compact shorthand."""
    mem = MechanismMemory()

    # Empty state
    empty_str = mem.compress_to_symbolic()
    assert "CTRL=[none]" in empty_str
    assert "INERT_SIGS=0" in empty_str

    # Add active controller B1
    sig_btn = EntitySignature(color=1, size_bucket="tiny", aspect_ratio_bucket="square", solidity_bucket="solid")
    mem.record_transition(
        step=1,
        action="ACTION6",
        target_coord=(32, 5),
        target_entity=sig_btn,
        effect_type=EffectType.STRUCTURAL_MUTATION,
        diff_count=35,
    )

    # Add inert puzzle piece
    sig_piece = EntitySignature(color=4, size_bucket="small", aspect_ratio_bucket="square", solidity_bucket="solid")
    mem.record_transition(
        step=2,
        action="ACTION6",
        target_coord=(10, 10),
        target_entity=sig_piece,
        effect_type=EffectType.NONE,
        diff_count=0,
    )

    encoded = mem.compress_to_symbolic()
    assert "(32,5):STRU" in encoded
    assert "INERT_SIGS=1" in encoded
    assert "ACTIVE_SIGS=1" in encoded
    assert len(encoded) < 100  # High density representation
