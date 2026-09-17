"""
Action Algebra Engine for ARC-AGI-3 (Baseline 3.0 B3.03).
Infers algebraic relationships between actions from observed transitions:
  - Inverse operations: f(f(s, A), B) == s  ==>  B = A^(-1)
  - Self-inverse toggles: f(f(s, A), A) == s
  - Additive compositions: f(s, A) yields monotonic state variable progress
Prevents detrimental oscillation (e.g., alternating +1 / -1 buttons) and enables
shortest-sequence planning under known group/monoid dynamics.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from src.arc_agent.memory.mechanism_memory import MechanismRecord


@dataclass(frozen=True)
class ActionRelation:
    """Algebraic relationship between two actions or self-action."""

    action_a: tuple[int, int]
    action_b: tuple[int, int]
    relation: str  # "inverse", "toggle", "additive", "identity"
    evidence_count: int = 1


class ActionAlgebraEngine:
    """Discovers and maintains relational action algebra from transition histories."""

    def __init__(self):
        self.known_relations: dict[tuple[tuple[int, int], tuple[int, int]], ActionRelation] = {}
        self.inverse_map: dict[tuple[int, int], tuple[int, int]] = {}
        self.toggles: set[tuple[int, int]] = set()

    def update_from_records(self, records: Sequence[MechanismRecord]):
        """Analyzes transition records sequentially to identify algebraic structures."""
        if len(records) < 2:
            return

        for i in range(len(records) - 1):
            r1 = records[i]
            r2 = records[i + 1]

            # Only analyze coordinate interventions with valid frame hashes
            if not (r1.pre_frame_hash and r1.post_frame_hash and r2.pre_frame_hash and r2.post_frame_hash):
                continue

            # Check if r2 was executed directly on the state left by r1
            if r1.post_frame_hash != r2.pre_frame_hash:
                continue

            c1 = r1.target_coord
            c2 = r2.target_coord

            # 1. State reversibility: does r2 restore the state to r1's initial state?
            if r2.post_frame_hash == r1.pre_frame_hash:
                if c1 == c2:
                    # Self-inverse / toggle: A followed by A returns to initial state
                    rel = ActionRelation(action_a=c1, action_b=c2, relation="toggle")
                    self.known_relations[(c1, c2)] = rel
                    self.toggles.add(c1)
                else:
                    # Inverse pair: A followed by B returns to initial state
                    rel = ActionRelation(action_a=c1, action_b=c2, relation="inverse")
                    self.known_relations[(c1, c2)] = rel
                    self.known_relations[(c2, c1)] = ActionRelation(action_a=c2, action_b=c1, relation="inverse")
                    self.inverse_map[c1] = c2
                    self.inverse_map[c2] = c1

    def is_inverse_pair(self, coord_a: tuple[int, int], coord_b: tuple[int, int]) -> bool:
        """Returns True if clicking coord_b undoes clicking coord_a."""
        return self.inverse_map.get(coord_a) == coord_b or (coord_a, coord_b) in self.known_relations

    def is_toggle(self, coord: tuple[int, int]) -> bool:
        """Returns True if repeating this coordinate toggles state back and forth."""
        return coord in self.toggles

    def get_inverse(self, coord: tuple[int, int]) -> tuple[int, int] | None:
        """Returns the inverse coordinate controller if known."""
        return self.inverse_map.get(coord)

    def filter_oscillating_actions(
        self,
        candidate_coords: Sequence[tuple[int, int]],
        recent_action_coords: Sequence[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """
        Prunes candidate coordinates that would immediately undo the most recent action.
        """
        if not recent_action_coords:
            return list(candidate_coords)

        last_coord = recent_action_coords[-1]
        inv_coord = self.get_inverse(last_coord)

        if inv_coord is None:
            return list(candidate_coords)

        # Filter out the inverse action to avoid cancelling forward progress
        filtered = [c for c in candidate_coords if c != inv_coord]
        return filtered if filtered else list(candidate_coords)

    def simplify_plan(self, plan: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
        """
        Counterfactual simulation: algebraically simplifies a multi-step sequence of
        actions before execution by cancelling adjacent inverse pairs and toggle pairs.
        Example: [L, R, L, L] -> [L, L]
        """
        if len(plan) <= 1:
            return list(plan)

        simplified: list[tuple[int, int]] = []
        for action in plan:
            if not simplified:
                simplified.append(action)
                continue

            prev = simplified[-1]

            # 1. Toggle cancellation: T followed by T cancels out
            if action == prev and self.is_toggle(action):
                simplified.pop()
                continue

            # 2. Inverse cancellation: A followed by B cancels out
            if self.is_inverse_pair(prev, action):
                simplified.pop()
                continue

            simplified.append(action)

        return simplified
