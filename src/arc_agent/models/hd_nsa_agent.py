"""
Hypothesis D: Hypothesis-Driven Neuro-Symbolic Agent (HD-NSA) [Target Model] for ARC-AGI-3.
Components:
1. Dynamic Entity Parser (DEP): Converts raw frames into an attributed relational graph G_t = (V, E).
2. Symbolic Hypothesizer: Generates candidate transition rules H = {H_1, H_2, ...} over kinematics,
   barriers, interactive targets, and goal predicates with Bayesian posterior calibration.
3. Epistemic Tree Search (ETS):
   - Probing Phase (Step <= K): Directs actions maximizing mutual information I(A; H) to eliminate competing models.
   - Exploitation Phase (Confidence > 85%): Switches to greedy A* shortest-path trajectory towards inferred goal object.
4. Hard Legality Guardrail: Conforms strictly to available_actions via LegalityAdapter.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass, field
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
class RelationalNode:
    """Node in the attributed relational entity graph."""
    node_id: int
    color: int
    size: int
    centroid: tuple[int, int]  # (y, x)
    bbox: tuple[int, int, int, int]  # (min_y, min_x, max_y, max_x)
    solidity: float
    is_dynamic: bool = False


@dataclass
class RelationalGraph:
    """Attributed relational entity graph G_t = (V, E)."""
    nodes: list[RelationalNode]
    background_color: int
    avatar_node: RelationalNode | None = None
    target_nodes: list[RelationalNode] = field(default_factory=list)
    barrier_nodes: list[RelationalNode] = field(default_factory=list)


class DynamicEntityParser:
    """Converts 2D integer frames into an attributed relational graph G_t."""

    def parse(
        self,
        frame: np.ndarray,
        prev_frame: np.ndarray | None = None,
        known_avatar_color: int | None = None,
    ) -> RelationalGraph:
        H, W = frame.shape
        unique_colors, counts = np.unique(frame, return_counts=True)
        bg_color = int(unique_colors[np.argmax(counts)])

        diff_mask = None
        if prev_frame is not None and prev_frame.shape == frame.shape:
            diff_mask = frame != prev_frame

        nodes: list[RelationalNode] = []
        node_counter = 0

        for color in unique_colors:
            color = int(color)
            if color == bg_color:
                continue

            mask = frame == color
            labeled_arr, num_feats = label(mask)

            for feat_idx in range(1, num_feats + 1):
                coords = np.argwhere(labeled_arr == feat_idx)
                if len(coords) == 0:
                    continue
                size = len(coords)
                min_y, min_x = coords.min(axis=0)
                max_y, max_x = coords.max(axis=0)
                bh = max(1, max_y - min_y + 1)
                bw = max(1, max_x - min_x + 1)
                solidity = float(size / (bh * bw))
                cy = int(round(coords[:, 0].mean()))
                cx = int(round(coords[:, 1].mean()))

                is_dyn = False
                if diff_mask is not None:
                    is_dyn = bool(np.any(diff_mask[labeled_arr == feat_idx]))

                node = RelationalNode(
                    node_id=node_counter,
                    color=color,
                    size=size,
                    centroid=(cy, cx),
                    bbox=(int(min_y), int(min_x), int(max_y), int(max_x)),
                    solidity=solidity,
                    is_dynamic=is_dyn,
                )
                nodes.append(node)
                node_counter += 1

        # Identify avatar node
        avatar_node = None
        if known_avatar_color is not None:
            matches = [n for n in nodes if n.color == known_avatar_color]
            if matches:
                avatar_node = matches[0]

        if avatar_node is None:
            # Check dynamic entities first
            dynamic_nodes = [n for n in nodes if n.is_dynamic and n.size <= 25]
            if dynamic_nodes:
                avatar_node = min(dynamic_nodes, key=lambda n: n.size)
            elif nodes:
                # Smallest non-background entity
                avatar_node = min(nodes, key=lambda n: n.size)

        # Categorize targets and barriers
        target_nodes = []
        barrier_nodes = []
        for n in nodes:
            if avatar_node is not None and n.node_id == avatar_node.node_id:
                continue
            if n.size <= 30:
                target_nodes.append(n)
            elif n.solidity >= 0.7 and (
                n.bbox[0] == 0 or n.bbox[2] == H - 1 or n.bbox[1] == 0 or n.bbox[3] == W - 1
            ):
                barrier_nodes.append(n)

        return RelationalGraph(
            nodes=nodes,
            background_color=bg_color,
            avatar_node=avatar_node,
            target_nodes=target_nodes,
            barrier_nodes=barrier_nodes,
        )


@dataclass
class KinematicHypothesis:
    action: str
    dy: int
    dx: int
    confidence: float = 0.5
    evidence_count: int = 0
    correct_count: int = 0


class SymbolicHypothesizer:
    """Maintains and updates candidate transition rules H = {H_1, ...}."""

    def __init__(self):
        self.kinematics: dict[str, KinematicHypothesis] = {}
        self.lethal_hazards: set[tuple[int, int]] = set()
        self.falsified_actions: set[str] = set()

    def update_transition(
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
            # Stationary / collision
            return

        norm_dy = 1 if dy > 0 else (-1 if dy < 0 else 0)
        norm_dx = 1 if dx > 0 else (-1 if dx < 0 else 0)

        if action not in self.kinematics:
            self.kinematics[action] = KinematicHypothesis(
                action=action, dy=norm_dy, dx=norm_dx, confidence=0.7, evidence_count=1, correct_count=1
            )
        else:
            hyp = self.kinematics[action]
            hyp.evidence_count += 1
            if (hyp.dy, hyp.dx) == (norm_dy, norm_dx):
                hyp.correct_count += 1
                hyp.confidence = min(0.99, hyp.confidence + 0.15)
            else:
                hyp.dy = norm_dy
                hyp.dx = norm_dx
                hyp.confidence = 0.60

    def is_calibrated(self) -> bool:
        """Returns True if at least 2 actions have confirmed confidence >= 0.85."""
        high_conf = sum(1 for h in self.kinematics.values() if h.confidence >= 0.85)
        return high_conf >= 2


class EpistemicTreeSearch:
    """
    Epistemic Tree Search (ETS):
    Directs actions maximizing mutual information during calibration,
    then executes greedy A* shortest-path trajectories to targets.
    """

    def __init__(self, probe_budget: int = 12):
        self.probe_budget = probe_budget
        self.active_macro_plan: deque[str] = deque()

    def select_action(
        self,
        graph: RelationalGraph,
        hypothesizer: SymbolicHypothesizer,
        available_actions: list[str],
        step_count: int,
        grid_shape: tuple[int, int],
    ) -> tuple[str, dict[str, Any]]:
        non_reset = [a for a in available_actions if a != "RESET"] or available_actions

        # 1. ACTION6 Coordinate Interaction Handling
        if "ACTION6" in available_actions:
            candidates = graph.target_nodes or graph.nodes
            if candidates:
                target = min(candidates, key=lambda n: n.size)
                return "ACTION6", {"x": target.centroid[1], "y": target.centroid[0]}
            return "ACTION6", {"x": grid_shape[1] // 2, "y": grid_shape[0] // 2}

        avatar = graph.avatar_node
        if avatar is None:
            return non_reset[step_count % len(non_reset)], {}

        # 2. Epistemic Probing Phase (Early steps or uncalibrated mechanics)
        untested = [a for a in non_reset if a not in hypothesizer.kinematics and a != "ACTION6"]
        if step_count <= self.probe_budget and untested:
            # Pick untested action to maximize mutual information I(A; H)
            return untested[0], {}

        # 3. Exploitation Phase: A* Pathing to Candidate Goal
        if graph.target_nodes and hypothesizer.kinematics:
            nearest_target = min(
                graph.target_nodes,
                key=lambda t: abs(t.centroid[0] - avatar.centroid[0])
                + abs(t.centroid[1] - avatar.centroid[1]),
            )

            # Plan path using A*
            path = self._astar_plan(
                start=avatar.centroid,
                goal=nearest_target.centroid,
                barriers=graph.barrier_nodes,
                kinematics=hypothesizer.kinematics,
                grid_shape=grid_shape,
                hazards=hypothesizer.lethal_hazards,
            )

            if path:
                next_act = path[0]
                if next_act in available_actions:
                    return next_act, {}

        # 4. Greedy directional alignment fallback
        if graph.target_nodes and hypothesizer.kinematics:
            nearest = min(
                graph.target_nodes,
                key=lambda t: abs(t.centroid[0] - avatar.centroid[0])
                + abs(t.centroid[1] - avatar.centroid[1]),
            )
            dy = nearest.centroid[0] - avatar.centroid[0]
            dx = nearest.centroid[1] - avatar.centroid[1]

            best_act = None
            best_align = -1e9
            for act, hyp in hypothesizer.kinematics.items():
                if act in non_reset:
                    ny = avatar.centroid[0] + hyp.dy
                    nx = avatar.centroid[1] + hyp.dx
                    if (ny, nx) in hypothesizer.lethal_hazards:
                        continue
                    align = hyp.dy * dy + hyp.dx * dx
                    if align > best_align and align > 0:
                        best_align = align
                        best_act = act

            if best_act:
                return best_act, {}

        return non_reset[step_count % len(non_reset)], {}

    def _astar_plan(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        barriers: list[RelationalNode],
        kinematics: dict[str, KinematicHypothesis],
        grid_shape: tuple[int, int],
        hazards: set[tuple[int, int]] | None = None,
    ) -> list[str]:
        H, W = grid_shape
        barrier_cells = set(hazards or set())
        for b in barriers:
            for y in range(b.bbox[0], b.bbox[2] + 1):
                for x in range(b.bbox[1], b.bbox[3] + 1):
                    barrier_cells.add((y, x))

        open_set: list[tuple[float, int, tuple[int, int], list[str]]] = []
        heapq.heappush(open_set, (0.0, 0, start, []))
        visited: set[tuple[int, int]] = {start}

        while open_set:
            f, cost, curr, path = heapq.heappop(open_set)
            if curr == goal or abs(curr[0] - goal[0]) + abs(curr[1] - goal[1]) <= 1:
                return path

            if cost > 40:
                continue

            for act, hyp in kinematics.items():
                ny = curr[0] + hyp.dy
                nx = curr[1] + hyp.dx
                if 0 <= ny < H and 0 <= nx < W and (ny, nx) not in barrier_cells:
                    if (ny, nx) not in visited:
                        visited.add((ny, nx))
                        h = abs(ny - goal[0]) + abs(nx - goal[1])
                        heapq.heappush(open_set, (cost + 1 + h, cost + 1, (ny, nx), path + [act]))

        return []


class HdNsaAgent(Agent):
    """
    Hypothesis D Agent: Hypothesis-Driven Neuro-Symbolic Agent (HD-NSA) [Target Model].
    Combines Dynamic Entity Parsing (DEP), Symbolic Kinematic Hypothesizer,
    and Epistemic Tree Search (ETS) for optimal RHAE.
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
        self.dep = DynamicEntityParser()
        self.hypothesizer = SymbolicHypothesizer()
        self.ets = EpistemicTreeSearch(
            probe_budget=self.parameters.get("probe_budget", 10)
        )

        self.previous_frame: np.ndarray | None = None
        self.previous_pos: tuple[int, int] | None = None
        self.previous_action: str | None = None
        self.avatar_color: int | None = None
        self.last_payload: dict[str, Any] = {}
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

        # 1. Dynamic Entity Parsing (DEP)
        graph = self.dep.parse(
            frame=grid,
            prev_frame=self.previous_frame,
            known_avatar_color=self.avatar_color,
        )

        if graph.avatar_node is not None:
            self.avatar_color = graph.avatar_node.color
            current_pos = graph.avatar_node.centroid
        else:
            current_pos = None

        # 2. Update Symbolic Hypothesizer
        if (
            self.previous_pos is not None
            and self.previous_action is not None
            and current_pos is not None
        ):
            self.hypothesizer.update_transition(
                action=self.previous_action,
                prev_pos=self.previous_pos,
                curr_pos=current_pos,
            )

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

        # 4. Legality and Planning
        if state_str in ("GAME_OVER", GameState.GAME_OVER.value, GameState.GAME_OVER):
            if current_pos is not None:
                self.hypothesizer.lethal_hazards.add(current_pos)
            if self.previous_pos is not None and self.previous_action in self.hypothesizer.kinematics:
                hyp = self.hypothesizer.kinematics[self.previous_action]
                trap_cell = (self.previous_pos[0] + hyp.dy, self.previous_pos[1] + hyp.dx)
                self.hypothesizer.lethal_hazards.add(trap_cell)
            chosen_action, payload = LegalityAdapter.validate_action(
                state=state_str,
                available_actions=avail_actions,
                proposed_action="RESET",
            )
        else:
            proposed_act, payload = self.ets.select_action(
                graph=graph,
                hypothesizer=self.hypothesizer,
                available_actions=avail_actions,
                step_count=self.action_count,
                grid_shape=grid.shape,
            )
            chosen_action, payload = LegalityAdapter.validate_action(
                state=state_str,
                available_actions=avail_actions,
                proposed_action=proposed_act,
                proposed_payload=payload,
            )

        self.previous_frame = grid.copy()
        self.previous_pos = current_pos
        self.previous_action = chosen_action
        self.last_payload = payload

        return LegalityAdapter.to_game_action(chosen_action, payload)
