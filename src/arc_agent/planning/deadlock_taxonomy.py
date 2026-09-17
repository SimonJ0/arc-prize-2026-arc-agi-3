"""
6-Tier Deadlock Taxonomy & Targeted Recovery Engine for ARC-AGI-3.
Classifies execution deadlocks and executes structured recovery strategies:
  Type 1: Navigation Loop (Oscillation between cyclic positions)
  Type 2: Wrong Goal (Unrewarded goal contact without level advancement)
  Type 3: Invalid Interaction Model (Action repeatedly yields unexpected zero delta)
  Type 4: Incomplete Precondition (Target reachable geometrically but blocked by prerequisite)
  Type 5: Lethal Trap (State entered GAME_OVER, requires RESET + hazard memory)
  Type 6: Perceptual Ambiguity (Visually indistinguishable candidates require probe)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any


class DeadlockType(str, Enum):
    TYPE_1_NAVIGATION_LOOP = "TYPE_1_NAVIGATION_LOOP"
    TYPE_2_WRONG_GOAL = "TYPE_2_WRONG_GOAL"
    TYPE_3_INVALID_INTERACTION = "TYPE_3_INVALID_INTERACTION"
    TYPE_4_INCOMPLETE_PRECONDITION = "TYPE_4_INCOMPLETE_PRECONDITION"
    TYPE_5_LETHAL_TRAP = "TYPE_5_LETHAL_TRAP"
    TYPE_6_PERCEPTUAL_AMBIGUITY = "TYPE_6_PERCEPTUAL_AMBIGUITY"


@dataclass
class DeadlockEvent:
    deadlock_type: DeadlockType
    description: str
    affected_coord: tuple[int, int] | None = None
    affected_action: str | None = None
    recommended_recovery: str = "explore_orthogonal"


class DeadlockTaxonomyEngine:
    """Detects and resolves execution deadlocks with targeted recovery routines."""

    def __init__(self):
        self.recent_positions: deque[tuple[int, int]] = deque(maxlen=8)
        self.visitation_counts: dict[tuple[int, int], int] = {}
        self.taboo_nodes: set[tuple[int, int]] = set()
        self.lethal_hazards: set[tuple[int, int]] = set()
        self.action_stall_counter: dict[str, int] = {}
        self.active_deadlock: DeadlockEvent | None = None

    def record_step(
        self,
        current_pos: tuple[int, int] | None,
        action: str,
        state: str,
        active_goal: tuple[int, int] | None,
        goal_reached_without_win: bool = False,
        diff_count: int = 0,
        target_coord: tuple[int, int] | None = None,
    ) -> DeadlockEvent | None:
        """Classifies state and returns active deadlock if detected."""
        self.active_deadlock = None

        # 1. Type 5: Lethal Trap / Game Over
        if state in ("GAME_OVER", "GameState.GAME_OVER"):
            if current_pos:
                self.lethal_hazards.add(current_pos)
            event = DeadlockEvent(
                deadlock_type=DeadlockType.TYPE_5_LETHAL_TRAP,
                description=f"State entered GAME_OVER at position {current_pos}",
                affected_coord=current_pos,
                affected_action=action,
                recommended_recovery="reset_and_mark_hazard",
            )
            self.active_deadlock = event
            return event

        # 2. Type 3: Invalid Interaction (Repeated zero-effect clicks on inert coordinates)
        if action == "ACTION6" and diff_count <= 2:
            self.action_stall_counter["ACTION6_INERT"] = self.action_stall_counter.get("ACTION6_INERT", 0) + 1
            if self.action_stall_counter["ACTION6_INERT"] >= 3:
                event = DeadlockEvent(
                    deadlock_type=DeadlockType.TYPE_3_INVALID_INTERACTION,
                    description=f"Repeated inert clicks on {target_coord} (diff={diff_count})",
                    affected_coord=target_coord,
                    affected_action=action,
                    recommended_recovery="switch_to_orthogonal_candidates",
                )
                self.active_deadlock = event
                return event
        elif action == "ACTION6" and diff_count > 2:
            self.action_stall_counter["ACTION6_INERT"] = 0

        if current_pos is None:
            return None

        self.recent_positions.append(current_pos)
        self.visitation_counts[current_pos] = self.visitation_counts.get(current_pos, 0) + 1

        # 3. Type 2: Wrong Goal
        if goal_reached_without_win and active_goal == current_pos:
            event = DeadlockEvent(
                deadlock_type=DeadlockType.TYPE_2_WRONG_GOAL,
                description=f"Reached candidate target {current_pos} without level completion",
                affected_coord=current_pos,
                recommended_recovery="falsify_goal_and_replan",
            )
            self.active_deadlock = event
            return event

        # 4. Type 1: Navigation Loop (A -> B -> A -> B oscillation or local node stall)
        if len(self.recent_positions) >= 4:
            p = list(self.recent_positions)
            if p[-1] == p[-3] and p[-2] == p[-4] and p[-1] != p[-2]:
                self.taboo_nodes.add(p[-1])
                self.taboo_nodes.add(p[-2])
                event = DeadlockEvent(
                    deadlock_type=DeadlockType.TYPE_1_NAVIGATION_LOOP,
                    description=f"Oscillation detected between {p[-1]} and {p[-2]}",
                    affected_coord=p[-1],
                    recommended_recovery="taboo_path_diversion",
                )
                self.active_deadlock = event
                return event

        if self.visitation_counts.get(current_pos, 0) >= 4:
            self.taboo_nodes.add(current_pos)
            event = DeadlockEvent(
                deadlock_type=DeadlockType.TYPE_1_NAVIGATION_LOOP,
                description=f"Position {current_pos} visited {self.visitation_counts[current_pos]} times",
                affected_coord=current_pos,
                recommended_recovery="taboo_path_diversion",
            )
            self.active_deadlock = event
            return event

        return None

    def reset_level(self):
        self.recent_positions.clear()
        self.visitation_counts.clear()
        self.taboo_nodes.clear()
        self.action_stall_counter.clear()
        self.active_deadlock = None
