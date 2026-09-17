"""
Capability Matrix Test 16: Causal Equivalence Classes & Ontology Induction (Baseline 3.0 B3.04).
Verifies:
- Morphological grouping of entities into equivalence classes
- Pruning of entire equivalence classes upon observing an inert representative (e.g., 38 puzzle pieces)
- Retention of candidate active controllers (e.g., buttons)
- Fallback preservation when all candidates belong to tested classes
"""

from dataclasses import dataclass
import pytest

from src.arc_agent.memory.effect_taxonomy import EffectType
from src.arc_agent.memory.mechanism_memory import EntitySignature, MechanismMemory


@dataclass
class MockPuzzleEntity:
    entity_id: int
    color: int
    size: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    cells: tuple[tuple[int, int], ...] = ()


def test_equivalence_class_clustering_and_pruning():
    """Testing 1 entity of an equivalence class prunes all 38 identical puzzle pieces."""
    mem = MechanismMemory()

    # Generate 38 identical interior puzzle pieces at various board coordinates
    puzzle_pieces = [
        MockPuzzleEntity(
            entity_id=100 + i,
            color=4,
            size=6,
            bbox=(10 + (i // 6) * 5, 10 + (i % 6) * 5, 12 + (i // 6) * 5, 13 + (i % 6) * 5),
            centroid=(float(11 + (i // 6) * 5), float(11 + (i % 6) * 5)),
        )
        for i in range(38)
    ]

    # Generate 2 interactive border buttons
    btn_left = MockPuzzleEntity(
        entity_id=1,
        color=1,
        size=4,
        bbox=(31, 4, 33, 6),
        centroid=(32.0, 5.0),
    )
    btn_right = MockPuzzleEntity(
        entity_id=2,
        color=1,
        size=4,
        bbox=(31, 57, 33, 59),
        centroid=(32.0, 58.0),
    )

    all_entities = puzzle_pieces + [btn_left, btn_right]
    assert len(all_entities) == 40

    # 1. Check equivalence class grouping
    classes = mem.compute_equivalence_classes(all_entities)
    assert len(classes) == 2  # Exactly two classes: PuzzlePiece class and Button class

    sig_piece = EntitySignature.from_entity(puzzle_pieces[0])
    sig_btn = EntitySignature.from_entity(btn_left)

    assert len(classes[sig_piece]) == 38
    assert len(classes[sig_btn]) == 2

    # 2. Before any interaction, filter_inert_classes keeps all 40 entities
    assert len(mem.filter_inert_classes(all_entities)) == 40

    # 3. Agent clicks just ONE puzzle piece (e.g., entity 100) -> yields EffectType.NONE
    mem.record_transition(
        step=1,
        action="ACTION6",
        target_coord=(11, 11),
        target_entity=sig_piece,
        effect_type=EffectType.NONE,
        diff_count=0,
    )

    # 4. Now filter_inert_classes should prune ALL 38 puzzle pieces simultaneously!
    remaining = mem.filter_inert_classes(all_entities)
    assert len(remaining) == 2
    assert btn_left in remaining
    assert btn_right in remaining
    for piece in puzzle_pieces:
        assert piece not in remaining


def test_equivalence_class_fallback():
    """If all entities belong to inert classes, filter_inert_classes safely returns all entities."""
    mem = MechanismMemory()

    pieces = [
        MockPuzzleEntity(
            entity_id=i,
            color=4,
            size=6,
            bbox=(10, 10, 12, 13),
            centroid=(11.0, 11.0),
        )
        for i in range(5)
    ]

    sig = EntitySignature.from_entity(pieces[0])
    mem.record_transition(
        step=1,
        action="ACTION6",
        target_coord=(11, 11),
        target_entity=sig,
        effect_type=EffectType.NONE,
        diff_count=0,
    )

    # All entities are inert -> fallback returns full list
    fallback = mem.filter_inert_classes(pieces)
    assert len(fallback) == 5
