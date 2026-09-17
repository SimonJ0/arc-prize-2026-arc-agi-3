"""
Capability Diagnostic 10: Deadlock Classification & Recovery.
Verifies detection of Type 1 navigation oscillations and taboo diversion.
"""

from src.arc_agent.planning.deadlock_taxonomy import (
    DeadlockTaxonomyEngine,
    DeadlockType,
)


def test_deadlock_classification_and_taboo():
    engine = DeadlockTaxonomyEngine()

    # Simulate oscillation: (5, 5) -> (5, 6) -> (5, 5) -> (5, 6)
    event1 = engine.record_step((5, 5), "ACTION4", "NOT_FINISHED", (10, 10))
    event2 = engine.record_step((5, 6), "ACTION3", "NOT_FINISHED", (10, 10))
    event3 = engine.record_step((5, 5), "ACTION4", "NOT_FINISHED", (10, 10))
    event4 = engine.record_step((5, 6), "ACTION3", "NOT_FINISHED", (10, 10))

    assert event4 is not None
    assert event4.deadlock_type == DeadlockType.TYPE_1_NAVIGATION_LOOP
    assert (5, 5) in engine.taboo_nodes
    assert (5, 6) in engine.taboo_nodes
