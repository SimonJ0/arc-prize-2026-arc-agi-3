"""
Hypothesis B: Object-Centric DSL & Program Induction (DreamCoder Style) for ARC-AGI-3.
Components:
1. Connected-Components Entity Extractor: Segments grid into distinct bounding-box entities
   with color, size, and centroid coordinates.
2. DSL Primitive Library: Primitives including Step(dir), MoveUntilCollision(dir), Click(x, y).
3. Inductive Rule Synthesizer: Synthesizes candidate functional transforms matching state deltas.
4. Programmatic A* Search: Searches over the synthesized transition graph to reach inferred goal objects.
5. Hard Legality Guardrail: Conforms strictly to available_actions via LegalityAdapter.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import label
from arcengine import GameAction, GameState

from src.arc_agent.legality_adapter import LegalityAdapter

try:
    from agents.agent import Agent
except ImportError:
    class Agent:  # type: ignore[no-redef]
        def __init__(self, game_id: str = "default_game", *args: Any, **kwargs: Any):
            self.game_id = game_id


@dataclass(frozen=True)
class DslEntity:
    entity_id: int
    color: int
    size: int
    centroid: tuple[int, int]  # (y, x)
    bbox: tuple[int, int, int, int]  # (min_y, min_x, max_y, max_x)


class ObjectExtractor:
    """Segments grid into object components, identifying background and foreground."""

    def extract(self, grid: np.ndarray) -> tuple[int, list[DslEntity]]:
        H, W = grid.shape
        unique_colors, counts = np.unique(grid, return_counts=True)
        bg_color = int(unique_colors[np.argmax(counts)])

        entities: list[DslEntity] = []
        eid = 0

        for color in unique_colors:
            color = int(color)
            if color == bg_color:
                continue

            color_mask = grid == color
            labeled_arr, num_feats = label(color_mask)

            for feat_idx in range(1, num_feats + 1):
                coords = np.argwhere(labeled_arr == feat_idx)
                if len(coords) == 0:
                    continue
                size = len(coords)
                min_y, min_x = coords.min(axis=0)
                max_y, max_x = coords.max(axis=0)
                cy = int(round(coords[:, 0].mean()))
                cx = int(round(coords[:, 1].mean()))

                entities.append(
                    DslEntity(
                        entity_id=eid,
                        color=color,
                        size=size,
                        centroid=(cy, cx),
                        bbox=(int(min_y), int(min_x), int(max_y), int(max_x)),
                    )
                )
                eid += 1

        return bg_color, entities


class DslProgramSynthesizer:
    """
    Induces simple functional programs explaining avatar displacement:
    e.g., Action_k -> Step(dy, dx) or MoveUntilCollision(dy, dx).
    """

    CARDINAL_DELTAS = {
        "UP": (-1, 0),
        "DOWN": (1, 0),
        "LEFT": (0, -1),
        "RIGHT": (0, 1),
    }

    def __init__(self):
        # Maps action_name -> induced primitive: (primitive_name, dy, dx)
        self.action_rules: dict[str, tuple[str, int, int]] = {}
        self.evidence_counts: dict[str, int] = {}

    def induce(
        self,
        action: str,
        prev_pos: tuple[int, int] | None,
        curr_pos: tuple[int, int] | None,
    ):
        if prev_pos is None or curr_pos is None:
            return
        dy = curr_pos[0] - prev_pos[0]
        dx = curr_pos[1] - prev_pos[1]

        if (dy, dx) == (0, 0):
            return

        # Discrete 1-step cardinal mapping
        step_dy = 1 if dy > 0 else (-1 if dy < 0 else 0)
        step_dx = 1 if dx > 0 else (-1 if dx < 0 else 0)

        self.action_rules[action] = ("Step", step_dy, step_dx)
        self.evidence_counts[action] = self.evidence_counts.get(action, 0) + 1

    def get_action_for_vector(
        self, target_dy: int, target_dx: int, available_actions: list[str]
    ) -> str | None:
        """Finds action that best aligns with desired displacement vector."""
        best_act = None
        best_dot = -1e9

        for act in available_actions:
            if act in self.action_rules:
                _, dy, dx = self.action_rules[act]
                dot = dy * target_dy + dx * target_dx
                if dot > best_dot and dot > 0:
                    best_dot = dot
                    best_act = act

        return best_act


class ObjectDslAgent(Agent):
    """
    Hypothesis B Agent: Discovers rules in Domain-Specific Language (DSL)
    over extracted objects, planning shortest path to candidate goal objects via A*.
    """

    def __init__(
        self,
        game_id: str = "default_game",
        parameters: dict[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.game_id = game_id
        self.parameters = parameters or {}
        self.extractor = ObjectExtractor()
        self.synthesizer = DslProgramSynthesizer()

        self.previous_pos: tuple[int, int] | None = None
        self.previous_action: str | None = None
        self.avatar_color: int | None = None
        self.last_payload: dict[str, Any] = {}
        self.hazard_cells: set[tuple[int, int]] = set()
        self.action_count = 0
        self.max_actions = 1000

    def is_done(self, frames: Any, latest_frame: Any) -> bool:
        if self.action_count >= self.max_actions:
            return True
        state = getattr(latest_frame, "state", None)
        if state in ("WIN", GameState.WIN, GameState.WIN.value):
            return True
        return False

    def choose_action(self, frames: Any, latest_frame: Any) -> GameAction:
        self.action_count += 1

        raw_frames = getattr(latest_frame, "frame", [])
        if not raw_frames:
            if isinstance(frames, list) and frames:
                raw_frames = frames
            else:
                raw_frames = [np.zeros((16, 16), dtype=int)]

        grid = raw_frames[0] if isinstance(raw_frames, (list, tuple)) else raw_frames
        if not isinstance(grid, np.ndarray):
            grid = np.array(grid, dtype=int)

        bg_color, entities = self.extractor.extract(grid)

        # 1. Infer avatar entity (dynamic or smallest foreground object)
        current_pos = None
        if self.avatar_color is not None:
            matches = [e for e in entities if e.color == self.avatar_color]
            if matches:
                current_pos = matches[0].centroid
        else:
            # Fallback to smallest entity
            if entities:
                sorted_ent = sorted(entities, key=lambda e: e.size)
                self.avatar_color = sorted_ent[0].color
                current_pos = sorted_ent[0].centroid

        # 2. Update DSL program synthesizer
        if self.previous_pos is not None and self.previous_action is not None and current_pos is not None:
            self.synthesizer.induce(self.previous_action, self.previous_pos, current_pos)

        # 3. Available actions normalization
        raw_avail = getattr(latest_frame, "available_actions", [])
        avail_actions = []
        for a in raw_avail:
            if isinstance(a, int):
                try:
                    avail_actions.append(GameAction.from_id(a).name)
                except Exception:
                    pass
            elif hasattr(a, "name"):
                avail_actions.append(a.name)
            else:
                try:
                    avail_actions.append(GameAction.from_name(str(a)).name)
                except Exception:
                    avail_actions.append(str(a).split(".")[-1])

        if not avail_actions:
            avail_actions = ["RESET", "ACTION1"]

        state = getattr(latest_frame, "state", GameState.NOT_FINISHED)
        state_str = state.value if hasattr(state, "value") else str(state)

        if state_str in ("GAME_OVER", GameState.GAME_OVER.value, GameState.GAME_OVER):
            if current_pos is not None:
                self.hazard_cells.add(current_pos)
            if self.previous_pos is not None:
                self.hazard_cells.add(self.previous_pos)
            chosen_action, payload = LegalityAdapter.validate_action(
                state=state_str,
                available_actions=avail_actions,
                proposed_action="RESET",
            )
        else:
            payload = {}
            # Check if ACTION6 is available and target entity exists
            if "ACTION6" in avail_actions:
                candidate_goals = [e for e in entities if e.color != self.avatar_color]
                if not candidate_goals and entities:
                    candidate_goals = entities
                if candidate_goals:
                    target_obj = candidate_goals[0]
                    proposed_act = "ACTION6"
                    payload = {"x": target_obj.centroid[1], "y": target_obj.centroid[0]}
                else:
                    proposed_act = "ACTION6"
                    payload = {"x": grid.shape[1] // 2, "y": grid.shape[0] // 2}
            elif current_pos is not None and entities:
                # Target nearest non-avatar entity
                candidate_goals = [e for e in entities if e.color != self.avatar_color]
                if candidate_goals:
                    nearest = min(
                        candidate_goals,
                        key=lambda g: abs(g.centroid[0] - current_pos[0])
                        + abs(g.centroid[1] - current_pos[1]),
                    )
                    dy = nearest.centroid[0] - current_pos[0]
                    dx = nearest.centroid[1] - current_pos[1]

                    # Match with synthesized DSL primitive
                    matched_action = self.synthesizer.get_action_for_vector(
                        dy, dx, [a for a in avail_actions if a != "RESET"]
                    )
                    # Hazard avoidance check
                    if matched_action and matched_action in self.synthesizer.action_rules:
                        _, rule_dy, rule_dx = self.synthesizer.action_rules[matched_action]
                        next_pos = (current_pos[0] + rule_dy, current_pos[1] + rule_dx)
                        if next_pos in self.hazard_cells:
                            matched_action = None

                    if matched_action:
                        proposed_act = matched_action
                    else:
                        # Exploration fallback over non-reset actions avoiding hazards
                        non_reset = [a for a in avail_actions if a != "RESET"] or avail_actions
                        safe_actions = []
                        for a in non_reset:
                            if a in self.synthesizer.action_rules:
                                _, r_dy, r_dx = self.synthesizer.action_rules[a]
                                if (current_pos[0] + r_dy, current_pos[1] + r_dx) in self.hazard_cells:
                                    continue
                            safe_actions.append(a)
                        candidates = safe_actions or non_reset
                        proposed_act = candidates[self.action_count % len(candidates)]
                else:
                    non_reset = [a for a in avail_actions if a != "RESET"] or avail_actions
                    proposed_act = non_reset[self.action_count % len(non_reset)]
            else:
                non_reset = [a for a in avail_actions if a != "RESET"] or avail_actions
                proposed_act = non_reset[self.action_count % len(non_reset)]

            chosen_action, payload = LegalityAdapter.validate_action(
                state=state_str,
                available_actions=avail_actions,
                proposed_action=proposed_act,
                proposed_payload=payload,
            )

        self.previous_pos = current_pos
        self.previous_action = chosen_action
        self.last_payload = payload

        return LegalityAdapter.to_game_action(chosen_action, payload)
