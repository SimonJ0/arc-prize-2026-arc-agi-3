"""
OpenAI Reasoning Persistence & Context Compaction Engine for ARC-AGI-3.
Implements the two breakthrough architectural settings that tripled ARC-AGI-3 benchmark scores:
1. Hidden Reasoning Persistence: Retains multi-step macro-plans, entity roles,
   and falsified hypotheses across turns instead of stateless single-step resets.
2. Automatic Context Compaction: Compresses past trajectory steps into dense semantic summaries,
   eliminating raw image array bloat and preventing context rot.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any

from src.arc_agent.memory.effect_taxonomy import EffectType, classify_effect


@dataclass(frozen=True)
class StepSummary:
    """Compact semantic representation of an environment step."""

    step: int
    level: int
    action: str
    player_pos: tuple[int, int] | None
    delta_pos: tuple[int, int]  # (dy, dx)
    state: str
    levels_completed: int


@dataclass
class PersistentReasoningState:
    """
    In-memory working scratchpad preserving reasoning state across environment turns.
    Prevents restarting hypothesis generation and planning from scratch on every step.
    """

    game_id: str
    # 1. Macro-Planning: Active queue of planned atomic actions in flight
    active_macro_plan: deque[str] = field(default_factory=deque)
    active_goal_coord: tuple[int, int] | None = None

    # 2. Semantic Entity Roles: e.g. {player_color: 'PLAYER', goal_color: 'GOAL'}
    entity_roles: dict[int, str] = field(default_factory=dict)
    player_color: int | None = None
    goal_color: int | None = None

    # 3. Lethal Hazard Avoidance (Learned from GAME_OVER)
    death_coords: set[tuple[int, int]] = field(default_factory=set)
    hazard_colors: set[int] = field(default_factory=set)

    # 4. Verified Causal Action Mappings: action_name -> (dy, dx)
    action_effects: dict[str, tuple[int, int]] = field(default_factory=dict)
    ineffective_actions: set[tuple[str, int]] = field(default_factory=set)  # (action, level)

    # 5. Spatial Visitation & Deadlock Tracking
    visited_positions: set[tuple[int, int]] = field(default_factory=set)
    visitation_counts: dict[tuple[int, int], int] = field(default_factory=dict)

    # 6. Surprise Detection & Falsification
    last_predicted_pos: tuple[int, int] | None = None
    falsified_goals: set[tuple[int, int]] = field(default_factory=set)
    current_level: int = 1

    # 7. Coordinate Affordance Memory (ACTION6)
    active_affordances: list[tuple[int, int]] = field(default_factory=list)
    inert_affordances: set[tuple[int, int]] = field(default_factory=set)
    lethal_affordances: set[tuple[int, int]] = field(default_factory=set)
    last_affordance_coord: tuple[int, int] | None = None
    last_affordance_diff: int = 0
    affordance_repeat_count: int = 0
    last_effect_type: EffectType | None = None

    def has_active_plan(self) -> bool:
        """Returns True if there is a pending macro-action sequence."""
        return len(self.active_macro_plan) > 0

    def next_planned_action(self, available_actions: Collection[str]) -> str | None:
        """Pops the next action if it is currently legal, otherwise invalidates plan."""
        if not self.active_macro_plan:
            return None
        candidate = self.active_macro_plan[0]
        if candidate in available_actions:
            return self.active_macro_plan.popleft()
        # Plan blocked or illegal, invalidate
        self.clear_plan()
        return None

    def set_macro_plan(self, plan: list[str], goal: tuple[int, int] | None = None):
        """Sets a new multi-step macro-plan in flight."""
        self.active_macro_plan = deque(plan)
        self.active_goal_coord = goal

    def clear_plan(self):
        """Discards active plan upon surprise or obstruction."""
        self.active_macro_plan.clear()
        self.active_goal_coord = None

    def record_visitation(self, pos: tuple[int, int]) -> int:
        """Increments and returns visitation frequency for position in current level."""
        self.visited_positions.add(pos)
        count = self.visitation_counts.get(pos, 0) + 1
        self.visitation_counts[pos] = count
        return count

    def is_loop_detected(self, pos: tuple[int, int], threshold: int = 3) -> bool:
        """Returns True if agent has visited this position repeatedly, signaling oscillation."""
        return self.visitation_counts.get(pos, 0) >= threshold

    def falsify_goal(self, goal: tuple[int, int]):
        """Marks goal coordinate as falsified/ineffective for this level."""
        self.falsified_goals.add(goal)
        if self.active_goal_coord == goal:
            self.clear_plan()

    def check_and_handle_surprise(self, actual_pos: tuple[int, int]) -> bool:
        """
        Compares actual position with last_predicted_pos.
        If surprise occurs while executing a macro plan, immediately invalidates plan.
        """
        surprise = False
        if self.last_predicted_pos is not None:
            if self.last_predicted_pos != actual_pos:
                surprise = True
                if self.has_active_plan():
                    self.clear_plan()
        self.last_predicted_pos = None
        return surprise

    def record_death(self, fatal_pos: tuple[int, int] | None, fatal_color: int | None):
        """Commits lethal position and entity color to permanent negative memory."""
        if fatal_pos is not None:
            self.death_coords.add(fatal_pos)
        if fatal_color is not None:
            self.hazard_colors.add(fatal_color)
        self.clear_plan()
        self.last_predicted_pos = None

    def update_action_effect(self, action: str, dy: int, dx: int):
        """Caches verified movement vector for an action."""
        if (dy, dx) != (0, 0):
            self.action_effects[action] = (dy, dx)

    def register_affordance_result(
        self,
        coord: tuple[int, int],
        diff_count: int,
        is_lethal: bool = False,
        state_str: str = "NOT_FINISHED",
        level_advanced: bool = False,
        effect_type: EffectType | None = None,
    ):
        """Registers the causal outcome of clicking coordinate (y, x)."""
        self.last_affordance_coord = coord
        self.last_affordance_diff = diff_count

        if effect_type is None:
            effect_type = classify_effect(
                diff_count=diff_count,
                state=state_str,
                level_advanced=level_advanced,
                is_lethal=is_lethal,
            )
        self.last_effect_type = effect_type

        if effect_type == EffectType.GAME_OVER:
            self.lethal_affordances.add(coord)
            if coord in self.active_affordances:
                self.active_affordances.remove(coord)
            self.affordance_repeat_count = 0
            return

        if effect_type in (EffectType.NONE, EffectType.UI_NOISE):
            self.inert_affordances.add(coord)
            if coord in self.active_affordances:
                self.active_affordances.remove(coord)
            self.affordance_repeat_count = 0
        else:
            # Genuine multi-pixel board transformation confirmed (LOCAL_MUTATION, STRUCTURAL_MUTATION, GLOBAL_MUTATION, LEVEL_ADVANCE)
            if coord not in self.active_affordances:
                self.active_affordances.append(coord)
            if coord in self.inert_affordances:
                self.inert_affordances.remove(coord)
            self.affordance_repeat_count += 1

    def reset_level(self, new_level: int):
        """
        Resets level-scoped working memory upon level transition.
        Retains persistent dynamics, death coordinates, and hazard colors.
        """
        self.current_level = new_level
        self.active_macro_plan.clear()
        self.active_goal_coord = None
        self.falsified_goals.clear()
        self.visitation_counts.clear()
        self.visited_positions.clear()
        self.last_predicted_pos = None
        self.active_affordances.clear()
        self.inert_affordances.clear()
        self.lethal_affordances.clear()
        self.last_affordance_coord = None
        self.last_affordance_diff = 0
        self.affordance_repeat_count = 0
        self.last_effect_type = None


class ContextCompactor:
    """
    Compresses raw 2D grid observations into dense semantic trajectory summaries.
    Avoids carrying large 2D frame arrays in memory.
    """

    def __init__(self, max_rolling_steps: int = 15):
        self.max_rolling_steps = max_rolling_steps
        self.rolling_history: deque[StepSummary] = deque(maxlen=max_rolling_steps)
        self.total_steps_recorded = 0
        self.level_step_counts: dict[int, int] = {}

    def compact_step(
        self,
        step: int,
        level: int,
        action: str,
        curr_pos: tuple[int, int] | None,
        prev_pos: tuple[int, int] | None,
        state: str,
        levels_completed: int,
    ) -> StepSummary:
        """Produces a StepSummary and stores it in compact rolling memory."""
        if curr_pos is not None and prev_pos is not None:
            delta = (curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1])
        else:
            delta = (0, 0)

        summary = StepSummary(
            step=step,
            level=level,
            action=action,
            player_pos=curr_pos,
            delta_pos=delta,
            state=state,
            levels_completed=levels_completed,
        )

        self.rolling_history.append(summary)
        self.total_steps_recorded += 1
        self.level_step_counts[level] = self.level_step_counts.get(level, 0) + 1
        return summary

    def get_trajectory_summary(self) -> dict[str, Any]:
        """Returns compact state dictionary for logging or planning."""
        recent = [
            {
                "step": s.step,
                "action": s.action,
                "pos": s.player_pos,
                "delta": s.delta_pos,
                "state": s.state,
            }
            for s in self.rolling_history
        ]
        return {
            "total_steps": self.total_steps_recorded,
            "steps_per_level": dict(self.level_step_counts),
            "recent_trajectory": recent,
        }
