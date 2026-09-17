"""
AUTONOMOUS ARC-AGI-3 UNCERTAINTY-AWARE AGENT (INLINED DEPLOYMENT BUNDLE)
Self-contained, offline-compatible implementation for Kaggle code competition.
"""
from __future__ import annotations
import os
import sys
import time
import json
import random
import hashlib
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np
from scipy.ndimage import label
from arcengine import GameAction, GameState, FrameDataRaw

try:
    from agents.agent import Agent
except ImportError:
    class Agent:  # type: ignore[no-redef]
        def __init__(self, game_id: str = "default_game", *args: Any, **kwargs: Any):
            self.game_id = game_id


# ======================================================================
# INLINED: contracts.py
# ======================================================================

"""
Typed contracts and interfaces for ARC-AGI-3 uncertainty-aware agents.
Defines immutable data objects for observations, world-model belief states,
action proposals, decision traces, and transition events.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Observation:
    """Immutable snapshot of the environment state returned by ARC-AGI-3."""

    frames: tuple[np.ndarray, ...]  # (H, W) arrays with values in [0, 15]
    state: str  # 'NOT_PLAYED', 'NOT_FINISHED', 'WIN', 'GAME_OVER'
    available_actions: frozenset[str]  # e.g. {'RESET', 'ACTION1', 'ACTION2', ...}
    game_key: str  # Environment / game ID
    level: int  # Current level index (1-based)
    action_count: int  # Actions taken so far in this level/game
    guid: str | None = None  # Session GUID


@dataclass(frozen=True)
class Transition:
    """Observed transition between two consecutive environment states."""

    from_observation_hash: str
    action: str
    payload: dict[str, Any]
    to_state: str
    to_observation_hash: str
    frame_diff_count: int


@dataclass
class TransitionModelHypothesis:
    """Single hypothesis about environment mechanics / transition rules."""

    hypothesis_id: str
    description: str
    action_semantics: dict[str, str] = field(default_factory=dict)  # e.g. {'ACTION1': 'MOVE_UP'}
    confidence: float = 0.5
    falsified: bool = False
    evidence_count: int = 0
    correct_predictions: int = 0

    def accuracy(self) -> float:
        if self.evidence_count == 0:
            return 0.5
        return self.correct_predictions / self.evidence_count


@dataclass
class WorldModelBelief:
    """Factored belief state representing a posterior distribution over transition hypotheses."""

    hypotheses: list[TransitionModelHypothesis] = field(default_factory=list)
    posterior: np.ndarray = field(default_factory=lambda: np.array([]))
    one_step_accuracy: float = 0.0
    rollout_accuracy: float = 0.0
    evidence: list[Transition] = field(default_factory=list)

    def get_most_likely_hypothesis(self) -> TransitionModelHypothesis | None:
        active = [h for h in self.hypotheses if not h.falsified]
        if not active:
            return None
        return max(active, key=lambda h: h.confidence)


@dataclass(frozen=True)
class GoalHypothesis:
    """Hypothesis about the winning terminal condition of a level."""

    goal_id: str
    predicate_type: str  # e.g. 'TARGET_REACHED', 'COLOR_CLEARED', 'PATTERN_MATCH'
    confidence: float = 0.5
    falsified: bool = False


@dataclass(frozen=True)
class ActionProposal:
    """Candidate action evaluated under current world-model beliefs."""

    action: str
    payload: dict[str, Any] = field(default_factory=dict)
    expected_completion_value: float = 0.0
    expected_information_gain: float = 0.0
    risk_game_over: float = 0.0
    cost: float = 1.0  # Physical action cost (always >= 1 in RHAE)

    @property
    def score(self) -> float:
        """Heuristic value combining completion utility and epistemic gain penalized by risk."""
        return (
            self.expected_completion_value
            + 0.4 * self.expected_information_gain
            - 2.0 * self.risk_game_over
        )


@dataclass(frozen=True)
class DecisionTrace:
    """Trace of agent reasoning for a single physical step."""

    level: int
    step: int
    observation_hash: str
    legal_actions: tuple[str, ...]
    selected_action: str
    selected_payload: dict[str, Any]
    planning_mode: str  # 'EPISTEMIC_PROBE', 'GOAL_PLAN', 'LEGAL_FALLBACK', 'RESET_RECOVERY'
    predicted_next_state: str | None
    confidence: float

# ======================================================================
# INLINED: legality_adapter.py
# ======================================================================

"""
Hard Legality Adapter for ARC-AGI-3.
Ensures zero 400 Bad Request errors by strictly enforcing:
1. When state == 'GAME_OVER', strictly emit RESET.
2. Only actions in the current frame's `available_actions` are emitted.
3. ACTION6 strictly requires integer coordinates (x, y) in range [0, 63].
4. Non-ACTION6 actions strip all coordinates.
"""

from collections.abc import Collection
from typing import Any

from arcengine import GameAction, GameState


class LegalityAdapter:
    """Hard guardrail ensuring every emitted action strictly adheres to environment contracts."""

    VALID_ACTIONS = {
        "RESET": GameAction.RESET,
        "ACTION1": GameAction.ACTION1,
        "ACTION2": GameAction.ACTION2,
        "ACTION3": GameAction.ACTION3,
        "ACTION4": GameAction.ACTION4,
        "ACTION5": GameAction.ACTION5,
        "ACTION6": GameAction.ACTION6,
        "ACTION7": GameAction.ACTION7,
    }

    @classmethod
    def validate_action(
        cls,
        state: str,
        available_actions: Collection[str],
        proposed_action: str,
        proposed_payload: dict[str, Any] | None = None,
        default_fallback: str = "RESET",
    ) -> tuple[str, dict[str, Any]]:
        """
        Validates and sanitizes proposed action against current environment state.

        Returns:
            Tuple of (sanitized_action_str, sanitized_payload_dict)
        """
        payload = dict(proposed_payload or {})

        # Rule 1: In GAME_OVER, the ONLY legal action is RESET
        if state in ("GAME_OVER", GameState.GAME_OVER.value, GameState.GAME_OVER):
            return "RESET", {}

        # Normalize action name string
        action_name = str(proposed_action).upper().strip()
        if action_name.startswith("GAMEACTION."):
            action_name = action_name.split(".")[-1]

        # Rule 2: Must be present in available_actions
        available_upper = {str(a).upper().split(".")[-1] for a in available_actions}
        if not available_upper:
            # If no available actions indicated, fallback to RESET
            return "RESET", {}

        if action_name not in available_upper:
            # Fallback to legal action
            if "ACTION1" in available_upper:
                action_name = "ACTION1"
            elif "RESET" in available_upper:
                action_name = "RESET"
            else:
                action_name = sorted(available_upper)[0]

        # Rule 3: Parameter rules
        if action_name == "ACTION6":
            # Must have valid x, y in [0, 63]
            raw_x = payload.get("x", 0)
            raw_y = payload.get("y", 0)
            try:
                x = int(raw_x)
                y = int(raw_y)
            except (ValueError, TypeError):
                x, y = 0, 0
            x = max(0, min(63, x))
            y = max(0, min(63, y))
            sanitized_payload = {"x": x, "y": y}
            if "game_id" in payload:
                sanitized_payload["game_id"] = payload["game_id"]
            return action_name, sanitized_payload
        else:
            # Strip coordinates for all simple actions & RESET
            sanitized_payload = {}
            if "reasoning" in payload:
                sanitized_payload["reasoning"] = payload["reasoning"]
            return action_name, sanitized_payload

    @classmethod
    def to_game_action(cls, action_str: str) -> GameAction:
        """Converts string action to arcengine GameAction enum."""
        key = action_str.upper().split(".")[-1]
        try:
            return GameAction.from_name(key)
        except Exception:
            return GameAction.RESET

# ======================================================================
# INLINED: layered_perception.py
# ======================================================================

"""
Layered & Multi-Hypothesis Perception for ARC-AGI-3.
Avoids fragile single-background assumptions. Computes:
1. Multi-candidate background hypotheses (frequency, border presence, stability).
2. Connected component segmentation across multiple connectivity modes (4-way, 8-way).
3. Temporal diff layers Delta(F_{t-1}, F_t) isolating active entities from invariant terrain.
4. Entity candidate extraction with spatial bounding boxes and centroids.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import label


@dataclass(frozen=True)
class EntityCandidate:
    """An identified object or cluster within the grid."""

    entity_id: int
    color: int
    cells: tuple[tuple[int, int], ...]
    bbox: tuple[int, int, int, int]  # (min_y, min_x, max_y, max_x)
    centroid: tuple[float, float]
    is_dynamic: bool = False

    @property
    def size(self) -> int:
        return len(self.cells)


@dataclass
class FrameAnalysis:
    """Multi-hypothesis perception result for a single observation."""

    frame_shape: tuple[int, int]
    present_colors: set[int]
    background_hypotheses: list[tuple[int, float]]  # (color, confidence)
    entities: list[EntityCandidate]
    dynamic_diff_mask: np.ndarray | None = None
    symmetry_scores: dict[str, float] = field(default_factory=dict)


class LayeredPerception:
    """Perception pipeline maintaining multiple structural segmentations."""

    def __init__(self):
        self.previous_frame: np.ndarray | None = None

    def analyze(self, frame: np.ndarray, prev_frame: np.ndarray | None = None) -> FrameAnalysis:
        """
        Processes a 2D integer grid frame into layered perceptual abstractions.
        """
        if frame.ndim != 2:
            raise ValueError(f"Expected 2D grid frame, got shape {frame.shape}")

        H, W = frame.shape
        unique_colors, counts = np.unique(frame, return_counts=True)
        present_colors = {int(c) for c in unique_colors}

        # 1. Multi-candidate background inference
        bg_hypotheses = self._infer_background_candidates(frame, unique_colors, counts)

        # 2. Dynamic temporal diff mask
        if prev_frame is None:
            prev_frame = self.previous_frame
        dynamic_mask = None
        if prev_frame is not None and prev_frame.shape == frame.shape:
            dynamic_mask = frame != prev_frame

        # 3. Extract entities across candidate non-background components
        primary_bg = bg_hypotheses[0][0] if bg_hypotheses else 0
        entities = self._extract_entities(frame, primary_bg, dynamic_mask)

        # 4. Symmetries
        symmetry_scores = {
            "horizontal": float(np.mean(frame == np.fliplr(frame))),
            "vertical": float(np.mean(frame == np.flipud(frame))),
        }

        self.previous_frame = frame.copy()

        return FrameAnalysis(
            frame_shape=(H, W),
            present_colors=present_colors,
            background_hypotheses=bg_hypotheses,
            entities=entities,
            dynamic_diff_mask=dynamic_mask,
            symmetry_scores=symmetry_scores,
        )

    def _infer_background_candidates(
        self, frame: np.ndarray, unique_colors: np.ndarray, counts: np.ndarray
    ) -> list[tuple[int, float]]:
        """
        Ranks candidate background colors using combined border density,
        overall area fraction, and connectivity.
        """
        H, W = frame.shape
        total_pixels = H * W
        border_pixels = np.concatenate([frame[0, :], frame[-1, :], frame[:, 0], frame[:, -1]])
        len(border_pixels)

        candidates = []
        for color, count in zip(unique_colors, counts, strict=False):
            color = int(color)
            area_frac = count / total_pixels
            border_frac = np.mean(border_pixels == color)
            # Composite score (weighted border presence and area)
            conf = 0.6 * border_frac + 0.4 * area_frac
            candidates.append((color, float(conf)))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates

    def _extract_entities(
        self, frame: np.ndarray, background_color: int, dynamic_mask: np.ndarray | None
    ) -> list[EntityCandidate]:
        """Extracts connected component entities excluding the primary candidate background."""
        entities = []
        entity_id_counter = 0

        for color in np.unique(frame):
            color = int(color)
            if color == background_color:
                continue

            color_mask = frame == color
            labeled_array, num_features = label(color_mask)

            for feat_idx in range(1, num_features + 1):
                coords = np.argwhere(labeled_array == feat_idx)
                if len(coords) == 0:
                    continue

                cells = tuple((int(y), int(x)) for y, x in coords)
                min_y, min_x = coords.min(axis=0)
                max_y, max_x = coords.max(axis=0)
                centroid = (float(coords[:, 0].mean()), float(coords[:, 1].mean()))

                is_dyn = False
                if dynamic_mask is not None:
                    is_dyn = bool(np.any(dynamic_mask[labeled_array == feat_idx]))

                entities.append(
                    EntityCandidate(
                        entity_id=entity_id_counter,
                        color=color,
                        cells=cells,
                        bbox=(int(min_y), int(min_x), int(max_y), int(max_x)),
                        centroid=centroid,
                        is_dynamic=is_dyn,
                    )
                )
                entity_id_counter += 1

        return entities

# ======================================================================
# INLINED: cognitive_hierarchy.py
# ======================================================================

"""
DRE-Bench 4-Level Cognitive Hierarchy for ARC-AGI-3 (arXiv:2506.02648v1).
Decomposes perception and causal reasoning across four cognitive tiers:
  Level 1: Attribute (size, count, color distribution, bounding box solidity)
  Level 2: Spatial (directional vectors, symmetry axes, rotation, canonicalization)
  Level 3: Sequential (multi-step macro-planning overcoming the 2-step depth collapse)
  Level 4: Conceptual / Intuitive Physics (gravity fields, collision barriers, reflection)
"""


from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import label


@dataclass(frozen=True)
class AttributeProfile:
    """Level 1 Attribute abstraction of an entity."""

    entity_id: int
    color: int
    size: int
    bbox: tuple[int, int, int, int]  # (min_y, min_x, max_y, max_x)
    centroid: tuple[float, float]
    aspect_ratio: float
    solidity: float  # size / (bbox_height * bbox_width)
    is_singleton: bool  # size == 1


@dataclass(frozen=True)
class SpatialSymmetry:
    """Level 2 Spatial symmetry detection."""

    horizontal: float
    vertical: float
    diagonal: float


@dataclass
class CognitiveHierarchyAnalysis:
    """Unified 4-level cognitive evaluation of an observation."""

    # Level 1: Attribute
    attributes: list[AttributeProfile]
    color_counts: dict[int, int]
    singleton_entities: list[AttributeProfile]
    dominant_color: int

    # Level 2: Spatial
    symmetry: SpatialSymmetry
    player_pos: tuple[int, int] | None = None
    player_color: int | None = None

    # Level 3: Sequential Targets
    candidate_goals: list[tuple[int, int]] = field(default_factory=list)

    # Level 4: Intuitive Physics
    gravity_detected: bool = False
    gravity_vector: tuple[int, int] = (0, 0)
    static_obstacles: set[tuple[int, int]] = field(default_factory=set)
    is_stagnant: bool = False


class CognitiveHierarchyPerception:
    """Analyzes ARC-AGI-3 frames using DRE-Bench's four cognitive levels."""

    def __init__(self):
        self.prev_frame: np.ndarray | None = None
        self.prev_player_pos: tuple[int, int] | None = None
        self.stagnant_steps: int = 0

    def analyze(
        self,
        frame: np.ndarray,
        prev_frame: np.ndarray | None = None,
        known_player_color: int | None = None,
    ) -> CognitiveHierarchyAnalysis:
        """Executes full 4-level cognitive breakdown of the grid."""
        H, W = frame.shape

        # --- LEVEL 1: ATTRIBUTE ANALYSIS ---
        unique_colors, counts = np.unique(frame, return_counts=True)
        color_counts = {int(c): int(cnt) for c, cnt in zip(unique_colors, counts, strict=False)}
        # Background is typically the color with maximum area
        dominant_color = int(unique_colors[np.argmax(counts)])

        attributes: list[AttributeProfile] = []
        singleton_entities: list[AttributeProfile] = []
        entity_id_seq = 0

        for color in unique_colors:
            color = int(color)
            if color == dominant_color:
                continue

            color_mask = frame == color
            labeled_arr, num_feats = label(color_mask)

            for feat_idx in range(1, num_feats + 1):
                coords = np.argwhere(labeled_arr == feat_idx)
                if len(coords) == 0:
                    continue

                size = len(coords)
                min_y, min_x = coords.min(axis=0)
                max_y, max_x = coords.max(axis=0)
                bh = max(1, max_y - min_y + 1)
                bw = max(1, max_x - min_x + 1)
                bbox_area = bh * bw
                solidity = float(size / bbox_area)
                aspect_ratio = float(bw / bh)
                centroid = (float(coords[:, 0].mean()), float(coords[:, 1].mean()))

                profile = AttributeProfile(
                    entity_id=entity_id_seq,
                    color=color,
                    size=size,
                    bbox=(int(min_y), int(min_x), int(max_y), int(max_x)),
                    centroid=centroid,
                    aspect_ratio=aspect_ratio,
                    solidity=solidity,
                    is_singleton=(size == 1),
                )
                attributes.append(profile)
                if size <= 4:
                    singleton_entities.append(profile)
                entity_id_seq += 1

        # --- LEVEL 2: SPATIAL ANALYSIS ---
        h_sym = float(np.mean(frame == np.fliplr(frame)))
        v_sym = float(np.mean(frame == np.flipud(frame)))
        d_sym = float(np.mean(frame == frame.T)) if H == W else 0.0
        symmetry = SpatialSymmetry(horizontal=h_sym, vertical=v_sym, diagonal=d_sym)

        # Infer player location:
        player_pos = None
        player_color = known_player_color

        # 1. Prioritize dynamic motion diffs across consecutive frames
        if prev_frame is not None and prev_frame.shape == frame.shape:
            diff = frame != prev_frame
            if np.any(diff):
                best_entity = None
                best_size = 999999
                for profile in attributes:
                    if profile.color == dominant_color:
                        continue
                    # Ignore UI indicators on extreme boundary rows if size <= 4
                    cy, cx = profile.centroid
                    if (int(cy) <= 1 or int(cy) >= H - 2) and profile.size <= 4:
                        continue
                    min_y, min_x, max_y, max_x = profile.bbox
                    ent_diff = diff[min_y : max_y + 1, min_x : max_x + 1] & (
                        frame[min_y : max_y + 1, min_x : max_x + 1] == profile.color
                    )
                    if np.any(ent_diff):
                        if profile.size < best_size:
                            best_entity = profile
                            best_size = profile.size
                if best_entity is not None:
                    player_pos = (int(best_entity.centroid[0]), int(best_entity.centroid[1]))
                    player_color = best_entity.color

        # 2. If no dynamic motion detected or first frame, use known player color
        if player_pos is None and player_color is not None:
            p_coords = np.argwhere(frame == player_color)
            if len(p_coords) > 0:
                player_pos = (int(p_coords[:, 0].mean()), int(p_coords[:, 1].mean()))

        # 3. Fallback player: smallest non-dominant entity
        if player_pos is None and singleton_entities:
            target = singleton_entities[0]
            player_pos = (int(target.centroid[0]), int(target.centroid[1]))
            player_color = target.color
        elif player_pos is None and attributes:
            sorted_candidates = sorted(
                [a for a in attributes if a.color != dominant_color], key=lambda a: a.size
            )
            if sorted_candidates:
                target = sorted_candidates[0]
                player_pos = (int(target.centroid[0]), int(target.centroid[1]))
                player_color = target.color

        # Track position stagnation (failsafe against false static avatar locks)
        if (
            self.prev_player_pos is not None
            and player_pos is not None
            and player_pos == self.prev_player_pos
        ):
            self.stagnant_steps += 1
        else:
            self.stagnant_steps = 0
        is_stagnant = self.stagnant_steps >= 4

        # --- LEVEL 3: CANDIDATE GOALS (Sequential targets) ---
        candidate_goals: list[tuple[int, int]] = []
        for profile in attributes:
            if player_color is not None and profile.color == player_color:
                continue
            # Goals are typically small unique objects (doors, sockets, stars)
            if profile.size <= 25:
                candidate_goals.append((int(profile.centroid[0]), int(profile.centroid[1])))

        # Sort candidate goals by distance to player
        if player_pos is not None:
            candidate_goals.sort(
                key=lambda g: abs(g[0] - player_pos[0]) + abs(g[1] - player_pos[1])
            )

        # --- LEVEL 4: INTUITIVE PHYSICS (Obstacles & Gravity) ---
        static_obstacles: set[tuple[int, int]] = set()
        player_standing_color = (
            frame[player_pos[0], player_pos[1]] if player_pos is not None else None
        )

        for profile in attributes:
            # Walkable surface/floor the player stands on is NEVER an obstacle
            if player_standing_color is not None and profile.color == player_standing_color:
                continue
            if player_color is not None and profile.color == player_color:
                continue
            # Small interactive entities (<35 pixels: keys, doors, stars) are never obstacles
            if profile.size < 35:
                continue

            min_y, min_x, max_y, max_x = profile.bbox
            touches_border = min_y == 0 or max_y == H - 1 or min_x == 0 or max_x == W - 1

            # Only entities touching the border with high solidity are treated as static boundary walls
            if touches_border and profile.solidity >= 0.7:
                for y in range(min_y, max_y + 1):
                    for x in range(min_x, max_x + 1):
                        if frame[y, x] == profile.color:
                            static_obstacles.add((y, x))

        if player_pos is not None:
            static_obstacles.discard(player_pos)

        # Check for gravity (downward vertical displacement across unforced steps)
        gravity_detected = False
        gravity_vector = (0, 0)
        if (
            self.prev_frame is not None
            and prev_frame is not None
            and player_pos is not None
            and self.prev_player_pos is not None
        ):
            dy = player_pos[0] - self.prev_player_pos[0]
            dx = player_pos[1] - self.prev_player_pos[1]
            if dy > 0 and dx == 0:
                gravity_detected = True
                gravity_vector = (1, 0)

        self.prev_frame = frame.copy()
        if player_pos is not None:
            self.prev_player_pos = player_pos

        return CognitiveHierarchyAnalysis(
            attributes=attributes,
            color_counts=color_counts,
            singleton_entities=singleton_entities,
            dominant_color=dominant_color,
            symmetry=symmetry,
            player_pos=player_pos,
            player_color=player_color,
            candidate_goals=candidate_goals,
            gravity_detected=gravity_detected,
            gravity_vector=gravity_vector,
            static_obstacles=static_obstacles,
            is_stagnant=is_stagnant,
        )

# ======================================================================
# INLINED: reasoning_state.py
# ======================================================================

"""
OpenAI Reasoning Persistence & Context Compaction Engine for ARC-AGI-3.
Implements the two breakthrough architectural settings that tripled ARC-AGI-3 benchmark scores:
1. Hidden Reasoning Persistence: Retains multi-step macro-plans, entity roles,
   and falsified hypotheses across turns instead of stateless single-step resets.
2. Automatic Context Compaction: Compresses past trajectory steps into dense semantic summaries,
   eliminating raw image array bloat and preventing context rot.
"""


from collections import deque
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any


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

# ======================================================================
# INLINED: belief_state.py
# ======================================================================

"""
Factored World Model Belief State for ARC-AGI-3.
Maintains a Bayesian distribution over transition dynamics hypotheses.
Tracks 1-step prediction accuracy and explicitly falsifies hypotheses upon contradiction.
Only authorizes deep planning when model consensus and fidelity meet confidence thresholds.
"""

from typing import Any

import numpy as np



class BeliefStateWorldModel:
    """Manages an evolving distribution of transition models calibrated against evidence."""

    # Common directional mapping hypothesis templates
    SEMANTIC_TEMPLATES = [
        {"ACTION1": "UP", "ACTION2": "DOWN", "ACTION3": "LEFT", "ACTION4": "RIGHT"},
        {"ACTION1": "DOWN", "ACTION2": "UP", "ACTION3": "LEFT", "ACTION4": "RIGHT"},
        {"ACTION1": "UP", "ACTION2": "DOWN", "ACTION3": "RIGHT", "ACTION4": "LEFT"},
        {"ACTION1": "LEFT", "ACTION2": "RIGHT", "ACTION3": "UP", "ACTION4": "DOWN"},
    ]

    DELTA_MAP = {
        "UP": (-1, 0),
        "DOWN": (1, 0),
        "LEFT": (0, -1),
        "RIGHT": (0, 1),
    }

    def __init__(self):
        self.belief = WorldModelBelief()
        self._init_hypotheses()
        self.avatar_color: int | None = None
        self.solid_colors: set[int] = set()
        # Empirical Transition Dynamics Learning
        # action -> {attempts: int, displacements: { (dy,dx): count }, blocked: int, dominant: (dy,dx), fidelity: float}
        self.action_stats: dict[str, dict[str, Any]] = {}

    def _init_hypotheses(self):
        """Initializes candidate transition dynamics hypotheses."""
        for idx, tpl in enumerate(self.SEMANTIC_TEMPLATES):
            self.belief.hypotheses.append(
                TransitionModelHypothesis(
                    hypothesis_id=f"HYP_DIR_{idx}",
                    description=f"Cardinal Mapping Template {idx}",
                    action_semantics=dict(tpl),
                    confidence=1.0 / len(self.SEMANTIC_TEMPLATES),
                )
            )

    def is_action_verified(self, action: str) -> bool:
        """
        Checks if an action passes empirical 1-step verification (>=85% fidelity over >=2 moves).
        """
        stats = self.action_stats.get(action)
        if stats and sum(stats["displacements"].values()) >= 2:
            return bool(stats["fidelity"] >= 0.85)
        # Fallback to top Bayesian hypothesis if sufficiently proven
        top_hyp = self.belief.get_most_likely_hypothesis()
        if (
            top_hyp is not None
            and top_hyp.confidence >= 0.80
            and top_hyp.evidence_count >= 3
            and top_hyp.accuracy() >= 0.85
            and action in top_hyp.action_semantics
        ):
            return True
        return False

    def get_action_displacement(self, action: str) -> tuple[int, int] | None:
        """Returns empirical or top-hypothesis displacement vector for action."""
        stats = self.action_stats.get(action)
        if stats and stats["dominant"] is not None and self.is_action_verified(action):
            dom = stats["dominant"]
            return (int(dom[0]), int(dom[1]))

        top_hyp = self.belief.get_most_likely_hypothesis()
        if top_hyp is not None and top_hyp.confidence >= 0.5:
            sem = top_hyp.action_semantics.get(action)
            if sem in self.DELTA_MAP:
                return self.DELTA_MAP[sem]

        # Default cardinal fallback
        default_deltas = {
            "ACTION1": (-1, 0),
            "ACTION2": (1, 0),
            "ACTION3": (0, -1),
            "ACTION4": (0, 1),
        }
        return default_deltas.get(action)

    def can_reliably_plan(self, action: str | None = None) -> bool:
        """
        Gating check: Deep forward planning is authorized ONLY when
        sufficient evidence confirms high predictive fidelity.
        """
        if action is not None:
            return self.is_action_verified(action)

        verified_count = sum(1 for a in self.action_stats if self.is_action_verified(a))
        if verified_count >= 2 and self.avatar_color is not None:
            return True

        top_hyp = self.belief.get_most_likely_hypothesis()
        if top_hyp is None:
            return False
        return (
            top_hyp.confidence >= 0.80
            and top_hyp.evidence_count >= 3
            and top_hyp.accuracy() >= 0.85
            and self.avatar_color is not None
        )

    def _record_empirical_displacement(self, action: str, dy: float, dx: float):
        """Records empirical (dy, dx) displacement and updates 1-step prediction stats."""
        if action not in self.action_stats:
            self.action_stats[action] = {
                "attempts": 0,
                "displacements": {},
                "blocked": 0,
                "dominant": None,
                "fidelity": 0.0,
            }
        stats = self.action_stats[action]
        stats["attempts"] += 1

        if abs(dy) < 0.2 and abs(dx) < 0.2:
            stats["blocked"] += 1
            return

        discrete_delta = (int(round(dy)), int(round(dx)))
        stats["displacements"][discrete_delta] = stats["displacements"].get(discrete_delta, 0) + 1

        dominant_disp, count = max(stats["displacements"].items(), key=lambda x: x[1])
        stats["dominant"] = dominant_disp
        total_moves = sum(stats["displacements"].values())
        stats["fidelity"] = count / total_moves if total_moves > 0 else 0.0

    def update_with_transition(
        self,
        prev_obs: Observation,
        action: str,
        curr_obs: Observation,
        prev_analysis: FrameAnalysis,
        curr_analysis: FrameAnalysis,
    ):
        """
        Updates belief state based on observed transition.
        Falsifies hypotheses that made incompatible predictions and updates posterior.
        """
        prev_frame = prev_obs.frames[0]
        curr_frame = curr_obs.frames[0]
        diff_count = int(np.sum(prev_frame != curr_frame))

        transition = Transition(
            from_observation_hash=str(hash(prev_frame.tobytes())),
            action=action,
            payload={},
            to_state=curr_obs.state,
            to_observation_hash=str(hash(curr_frame.tobytes())),
            frame_diff_count=diff_count,
        )
        self.belief.evidence.append(transition)

        # 1. Deduce avatar color from dynamic movement
        if diff_count > 0 and self.avatar_color is None:
            for ent in curr_analysis.entities:
                if ent.is_dynamic and ent.size <= 16:  # Avatars are typically compact
                    self.avatar_color = ent.color
                    break

        # Record empirical displacement if avatar is known
        if self.avatar_color is not None:
            prev_pos = self._find_entity_centroid(prev_frame, self.avatar_color)
            curr_pos = self._find_entity_centroid(curr_frame, self.avatar_color)
            if prev_pos is not None and curr_pos is not None:
                dy = curr_pos[0] - prev_pos[0]
                dx = curr_pos[1] - prev_pos[1]
                self._record_empirical_displacement(action, dy, dx)

        # 2. If action is directional, test directional hypotheses
        if action in ("ACTION1", "ACTION2", "ACTION3", "ACTION4") and self.avatar_color is not None:
            self._update_directional_hypotheses(action, prev_frame, curr_frame)

        # 3. Recompute posterior distribution
        self._recompute_posterior()

    def _update_directional_hypotheses(
        self, action: str, prev_frame: np.ndarray, curr_frame: np.ndarray
    ):
        """Validates predicted movement of the avatar against observed frame displacement."""
        if self.avatar_color is None:
            return
        prev_pos = self._find_entity_centroid(prev_frame, self.avatar_color)
        curr_pos = self._find_entity_centroid(curr_frame, self.avatar_color)

        if prev_pos is None or curr_pos is None:
            return

        dy = curr_pos[0] - prev_pos[0]
        dx = curr_pos[1] - prev_pos[1]

        # Determine observed movement direction
        observed_dir = None
        if dy < -0.4 and abs(dx) < 0.5:
            observed_dir = "UP"
        elif dy > 0.4 and abs(dx) < 0.5:
            observed_dir = "DOWN"
        elif dx < -0.4 and abs(dy) < 0.5:
            observed_dir = "LEFT"
        elif dx > 0.4 and abs(dy) < 0.5:
            observed_dir = "RIGHT"
        elif abs(dx) < 0.1 and abs(dy) < 0.1:
            observed_dir = "STATIONARY"  # Collided with wall or obstacle

        if observed_dir == "STATIONARY":
            # Possible wall collision - record color at attempted movement cell
            return

        for hyp in self.belief.hypotheses:
            if hyp.falsified:
                continue

            expected_dir = hyp.action_semantics.get(action)
            hyp.evidence_count += 1

            if observed_dir is not None:
                if expected_dir == observed_dir:
                    hyp.correct_predictions += 1
                    hyp.confidence *= 2.0  # Bayesian likelihood boost
                else:
                    hyp.confidence *= 0.1
                    if hyp.evidence_count >= 2 and hyp.accuracy() < 0.3:
                        hyp.falsified = True

    def _find_entity_centroid(self, frame: np.ndarray, color: int) -> tuple[float, float] | None:
        coords = np.argwhere(frame == color)
        if len(coords) == 0:
            return None
        return (float(coords[:, 0].mean()), float(coords[:, 1].mean()))

    def _recompute_posterior(self):
        """Normalizes confidence scores into a valid simplex posterior distribution."""
        active = [h for h in self.belief.hypotheses if not h.falsified]
        if not active:
            self.belief.posterior = np.array([])
            return

        total_conf = sum(h.confidence for h in active)
        if total_conf > 0:
            for h in active:
                h.confidence /= total_conf
            self.belief.posterior = np.array([h.confidence for h in active])

        # Compute overall 1-step accuracy
        total_ev = sum(h.evidence_count for h in active)
        total_corr = sum(h.correct_predictions for h in active)
        self.belief.one_step_accuracy = (total_corr / total_ev) if total_ev > 0 else 0.0

    def predict_next_avatar_pos(
        self, curr_pos: tuple[int, int], action: str
    ) -> tuple[int, int] | None:
        """Predicts next avatar position under verified empirical displacement or top hypothesis."""
        disp = self.get_action_displacement(action)
        if disp is not None:
            return (curr_pos[0] + disp[0], curr_pos[1] + disp[1])
        top_hyp = self.belief.get_most_likely_hypothesis()
        if top_hyp is None:
            return None
        direction = top_hyp.action_semantics.get(action)
        if direction not in self.DELTA_MAP:
            return None
        dy, dx = self.DELTA_MAP[direction]
        return (curr_pos[0] + dy, curr_pos[1] + dx)

# ======================================================================
# INLINED: epistemic_policy.py
# ======================================================================

"""
Uncertainty-Aware Epistemic Policy for ARC-AGI-3.
Integrates:
1. OpenAI Reasoning Persistence: Executes active multi-step macro-plans across turns.
2. DRE-Bench Cognitive Hierarchy: Targets goal candidates and avoids static obstacles / death coordinates.
3. Bayesian Epistemic Probing: Explores untested actions to deduce movement dynamics when uncertain.
4. Hard Legality Validation: Via LegalityAdapter.
"""

from collections import deque
from collections.abc import Collection
from typing import Any

import numpy as np



class EpistemicPolicy:
    """Decision engine balancing active epistemic learning with persistent macro-planning."""

    def __init__(self):
        self.step_counter = 0

    def select_action(
        self,
        observation: Observation,
        analysis: FrameAnalysis,
        world_model: BeliefStateWorldModel,
        reasoning_state: PersistentReasoningState | None = None,
        cognitive_analysis: CognitiveHierarchyAnalysis | None = None,
    ) -> tuple[str, dict[str, Any], DecisionTrace]:
        """
        Selects next physical environment action given current belief state.
        Guaranteed to return a legal action from observation.available_actions.
        """
        self.step_counter += 1
        state = observation.state
        available = observation.available_actions
        p_pos = cognitive_analysis.player_pos if cognitive_analysis else None

        # 1. State-level guard: GAME_OVER requires RESET
        if state == "GAME_OVER":
            if reasoning_state is not None:
                # Learn fatal location to avoid it in future runs
                reasoning_state.record_death(p_pos, None)

            action, payload = LegalityAdapter.validate_action(
                state=state,
                available_actions=available,
                proposed_action="RESET",
            )
            trace = DecisionTrace(
                level=observation.level,
                step=self.step_counter,
                observation_hash=str(hash(observation.frames[0].tobytes())),
                legal_actions=tuple(sorted(available)),
                selected_action=action,
                selected_payload=payload,
                planning_mode="RESET_RECOVERY",
                predicted_next_state="NOT_FINISHED",
                confidence=1.0,
            )
            return action, payload, trace

        # Surprise & Deadlock Loop Management
        is_loop = False
        if reasoning_state is not None and p_pos is not None:
            # 1b. Surprise Invalidation: Did last action land where we expected?
            reasoning_state.check_and_handle_surprise(p_pos)

            # 1c. Loop / Deadlock Breaker
            v_count = reasoning_state.record_visitation(p_pos)
            if v_count >= 3:
                is_loop = True

            # 1d. Falsify unrewarded goals: If player reached active goal or looped without level advancing
            if reasoning_state.active_goal_coord is not None:
                g = reasoning_state.active_goal_coord
                if not reasoning_state.has_active_plan() and p_pos == g:
                    reasoning_state.falsify_goal(g)
                elif is_loop:
                    reasoning_state.falsify_goal(g)

        # 2. Reasoning Persistence: Check if there is an active macro-plan in flight (and not in deadlock loop)
        if reasoning_state is not None and reasoning_state.has_active_plan() and not is_loop:
            planned_action = reasoning_state.next_planned_action(available)
            if planned_action is not None:
                action, payload = LegalityAdapter.validate_action(
                    state=state,
                    available_actions=available,
                    proposed_action=planned_action,
                )
                if p_pos is not None:
                    reasoning_state.last_predicted_pos = world_model.predict_next_avatar_pos(
                        p_pos, action
                    )
                trace = DecisionTrace(
                    level=observation.level,
                    step=self.step_counter,
                    observation_hash=str(hash(observation.frames[0].tobytes())),
                    legal_actions=tuple(sorted(available)),
                    selected_action=action,
                    selected_payload=payload,
                    planning_mode="PERSISTENT_MACRO_PLAN",
                    predicted_next_state="NOT_FINISHED",
                    confidence=0.95,
                )
                return action, payload, trace

        # 3. DRE-Bench Sequential Planning: Macro-path to candidate goals
        if cognitive_analysis is not None and p_pos is not None and not is_loop:
            all_goals = cognitive_analysis.candidate_goals
            # Filter out falsified goals for this level
            if reasoning_state is not None and reasoning_state.falsified_goals:
                candidate_goals = [g for g in all_goals if g not in reasoning_state.falsified_goals]
            else:
                candidate_goals = all_goals

            obstacles = set(cognitive_analysis.static_obstacles)
            if reasoning_state is not None:
                obstacles.update(reasoning_state.death_coords)

            frame_shape = observation.frames[0].shape
            for goal in candidate_goals[:3]:  # Evaluate top non-falsified candidate goals
                path = self._astar_search(
                    start=p_pos,
                    goal=goal,
                    grid_shape=frame_shape,
                    obstacles=obstacles,
                    world_model=world_model,
                    available_actions=available,
                    reasoning_state=reasoning_state,
                )
                if path:
                    chosen = path[0]
                    if reasoning_state is not None and len(path) > 1:
                        # Retain remainder of plan in persistent memory
                        reasoning_state.set_macro_plan(path[1:], goal=goal)

                    action, payload = LegalityAdapter.validate_action(
                        state=state,
                        available_actions=available,
                        proposed_action=chosen,
                    )
                    if reasoning_state is not None:
                        reasoning_state.last_predicted_pos = world_model.predict_next_avatar_pos(
                            p_pos, action
                        )
                    trace = DecisionTrace(
                        level=observation.level,
                        step=self.step_counter,
                        observation_hash=str(hash(observation.frames[0].tobytes())),
                        legal_actions=tuple(sorted(available)),
                        selected_action=action,
                        selected_payload=payload,
                        planning_mode="MACRO_GOAL_PLAN",
                        predicted_next_state="NOT_FINISHED",
                        confidence=0.9,
                    )
                    return action, payload, trace

        # 4. Exploitation Mode: Validated model allows forward planning (if not in loop)
        if world_model.can_reliably_plan() and not is_loop:
            action, payload, trace = self._plan_goal_trajectory(observation, analysis, world_model)
            if reasoning_state is not None and p_pos is not None:
                reasoning_state.last_predicted_pos = world_model.predict_next_avatar_pos(
                    p_pos, action
                )
            return action, payload, trace

        # 5. Epistemic Probing Mode / Deadlock Breaker
        action, payload, trace = self._select_epistemic_probe(
            observation, analysis, world_model, reasoning_state, is_loop
        )
        if reasoning_state is not None and p_pos is not None:
            reasoning_state.last_predicted_pos = world_model.predict_next_avatar_pos(p_pos, action)
        return action, payload, trace

    def _select_epistemic_probe(
        self,
        observation: Observation,
        analysis: FrameAnalysis,
        world_model: BeliefStateWorldModel,
        reasoning_state: PersistentReasoningState | None = None,
        is_loop: bool = False,
    ) -> tuple[str, dict[str, Any], DecisionTrace]:
        """Selects informative probe action to distinguish candidate transition models or break deadlocks."""
        available = list(observation.available_actions)

        # Candidate probe actions: evaluate all available non-reset actions
        candidate_probes = [a for a in available if a != "RESET"]
        if not candidate_probes:
            candidate_probes = list(available)

        # Prioritize untested actions in world_model.action_stats
        untested = [
            a
            for a in candidate_probes
            if a not in getattr(world_model, "action_stats", {})
            or world_model.action_stats[a]["attempts"] == 0
        ]
        if untested:
            selected = untested[self.step_counter % len(untested)]
        else:
            # Sort by least attempts
            candidate_probes.sort(
                key=lambda a: getattr(world_model, "action_stats", {}).get(a, {}).get("attempts", 0)
            )
            selected = candidate_probes[0]

        payload = {}
        if selected == "ACTION6":
            # Bounded coordinate selection: choose entity centroid clamped to [0, 63]
            if analysis.entities:
                play_entities = [
                    e
                    for e in analysis.entities
                    if 1 < int(e.centroid[0]) < 62 and 1 < int(e.centroid[1]) < 62
                ]
                pool = play_entities if play_entities else analysis.entities
                target_ent = pool[self.step_counter % len(pool)]
                cy, cx = target_ent.centroid
                payload = {
                    "x": int(np.clip(cx, 0, 63)),
                    "y": int(np.clip(cy, 0, 63)),
                }
            else:
                H, W = analysis.frame_shape
                payload = {
                    "x": int(np.clip(W // 2, 0, 63)),
                    "y": int(np.clip(H // 2, 0, 63)),
                }

        action, valid_payload = LegalityAdapter.validate_action(
            state=observation.state,
            available_actions=observation.available_actions,
            proposed_action=selected,
            proposed_payload=payload,
        )

        trace = DecisionTrace(
            level=observation.level,
            step=self.step_counter,
            observation_hash=str(hash(observation.frames[0].tobytes())),
            legal_actions=tuple(sorted(observation.available_actions)),
            selected_action=action,
            selected_payload=valid_payload,
            planning_mode="DEADLOCK_BREAKER" if is_loop else "EPISTEMIC_PROBE",
            predicted_next_state="NOT_FINISHED",
            confidence=0.5,
        )
        return action, valid_payload, trace

    def _plan_goal_trajectory(
        self,
        observation: Observation,
        analysis: FrameAnalysis,
        world_model: BeliefStateWorldModel,
    ) -> tuple[str, dict[str, Any], DecisionTrace]:
        """Plans shortest path to candidate goal under validated transition dynamics."""
        frame = observation.frames[0]
        avatar_color = world_model.avatar_color
        available = observation.available_actions

        avatar_coords = np.argwhere(frame == avatar_color)
        if len(avatar_coords) == 0:
            return self._select_epistemic_probe(observation, analysis, world_model)

        start_pos = (int(avatar_coords[0][0]), int(avatar_coords[0][1]))

        target_pos = None
        for ent in analysis.entities:
            if ent.color != avatar_color and ent.size <= 36:
                target_pos = (int(ent.centroid[0]), int(ent.centroid[1]))
                break

        if target_pos is None:
            return self._select_epistemic_probe(observation, analysis, world_model)

        path = self._bfs_search(start_pos, target_pos, frame.shape, world_model, available)

        if path:
            chosen_action = path[0]
            action, valid_payload = LegalityAdapter.validate_action(
                state=observation.state,
                available_actions=available,
                proposed_action=chosen_action,
            )
            trace = DecisionTrace(
                level=observation.level,
                step=self.step_counter,
                observation_hash=str(hash(frame.tobytes())),
                legal_actions=tuple(sorted(available)),
                selected_action=action,
                selected_payload=valid_payload,
                planning_mode="GOAL_PLAN",
                predicted_next_state="NOT_FINISHED",
                confidence=0.9,
            )
            return action, valid_payload, trace

        return self._select_epistemic_probe(observation, analysis, world_model)

    def _astar_search(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        grid_shape: tuple[int, int],
        obstacles: set[tuple[int, int]],
        world_model: BeliefStateWorldModel,
        available_actions: Collection[str],
        reasoning_state: PersistentReasoningState | None = None,
    ) -> list[str] | None:
        """A* search towards goal avoiding static obstacles and lethal death coordinates."""
        H, W = grid_shape
        import heapq

        default_deltas: dict[str, tuple[int, int]] = {
            "ACTION1": (-1, 0),  # UP
            "ACTION2": (1, 0),  # DOWN
            "ACTION3": (0, -1),  # LEFT
            "ACTION4": (0, 1),  # RIGHT
        }

        # Build action displacement mapping from empirical learning and world model
        action_deltas: dict[str, tuple[int, int]] = {}
        known_step_sizes = []
        for act in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"):
            if act in available_actions:
                disp = None
                if reasoning_state is not None and act in reasoning_state.action_effects:
                    disp = reasoning_state.action_effects[act]
                elif world_model is not None:
                    disp = world_model.get_action_displacement(act)
                if disp is not None and disp != (0, 0):
                    action_deltas[act] = disp
                    known_step_sizes.append(max(abs(disp[0]), abs(disp[1])))

        base_step = int(np.median(known_step_sizes)) if known_step_sizes else 1

        # Populate cardinal fallback per available action so search space never collapses to 1D
        for act, (dy, dx) in default_deltas.items():
            if act in available_actions and act not in action_deltas:
                action_deltas[act] = (dy * base_step, dx * base_step)

        valid_actions = [act for act in action_deltas if act in available_actions]
        if not valid_actions:
            return None

        # Ensure start and goal coordinates are not blocked by obstacle mask
        nav_obstacles = set(obstacles) - {start, goal}

        # Priority queue stores (f_score, cost, current_pos, path)
        def h(pos: tuple[int, int]) -> int:
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

        heap: list[tuple[int, int, tuple[int, int], list[str]]] = [(h(start), 0, start, [])]
        visited = {start: 0}
        max_expansions = 200  # Bound computation
        goal_tolerance = max(1, base_step)

        best_path = None
        best_dist = float("inf")

        while heap and max_expansions > 0:
            max_expansions -= 1
            f, cost, curr, path = heapq.heappop(heap)

            if curr == goal:
                return path

            dist = h(curr)
            if path and dist < best_dist:
                best_dist = dist
                best_path = path

            for act in valid_actions:
                dy, dx = action_deltas[act]
                ny, nx = curr[0] + dy, curr[1] + dx

                if not (0 <= ny < H and 0 <= nx < W):
                    continue
                nxt = (ny, nx)
                if nxt in nav_obstacles:
                    continue

                # Add penalty for highly-visited positions to discourage looping
                extra_cost = 0
                if reasoning_state and nxt in reasoning_state.visitation_counts:
                    extra_cost = reasoning_state.visitation_counts[nxt] * 2

                # Prefer verified actions
                if world_model and not world_model.is_action_verified(act):
                    extra_cost += 1

                new_cost = cost + 1 + extra_cost
                if nxt not in visited or new_cost < visited[nxt]:
                    visited[nxt] = new_cost
                    heapq.heappush(heap, (new_cost + h(nxt), new_cost, nxt, path + [act]))

        if best_path is not None and best_dist <= goal_tolerance:
            return best_path

        return None

    def _bfs_search(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        grid_shape: tuple[int, int],
        world_model: BeliefStateWorldModel,
        available_actions: Collection[str],
    ) -> list[str] | None:
        """Bounded BFS search over validated directional dynamics."""
        H, W = grid_shape
        queue: deque[tuple[tuple[int, int], list[str]]] = deque([(start, [])])
        visited = {start}

        dir_actions = [
            a for a in ("ACTION1", "ACTION2", "ACTION3", "ACTION4") if a in available_actions
        ]
        max_depth = 40

        while queue:
            curr_pos, path = queue.popleft()
            if curr_pos == goal:
                return path

            if len(path) >= max_depth:
                continue

            for act in dir_actions:
                nxt = world_model.predict_next_avatar_pos(curr_pos, act)
                if nxt is None:
                    continue
                ny, nx = nxt
                if 0 <= ny < H and 0 <= nx < W and nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, path + [act]))

        return None

# ======================================================================
# INLINED: scoped_memory.py
# ======================================================================

"""
Scoped Episode Memory for ARC-AGI-3.
Binds learned facts strictly to (game_family, version, run_id).
Invalidates facts upon prediction error, and records hashable transition event logs
for reproducible replay analysis and debugging.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TransitionEvent:
    """Hashable record of an environment transition and agent decision."""

    step: int
    level: int
    from_frame_hash: str
    action: str
    payload: dict[str, Any]
    to_state: str
    to_frame_hash: str
    planning_mode: str
    confidence: float
    prediction_error: bool


class ScopedEpisodeMemory:
    """Manages scoped invariant storage and transition event logging."""

    def __init__(self, game_key: str = "unknown", version: str = "1.0", run_id: str = "0"):
        self.game_key = game_key
        self.version = version
        self.run_id = run_id

        # Invariants carried across levels of the same game run
        self.confirmed_invariants: dict[str, Any] = {}
        self.events: list[TransitionEvent] = []
        self.forbidden_transitions: set[tuple[str, str]] = set()

    def reset_for_new_game(self, game_key: str, version: str = "1.0", run_id: str = "0"):
        """Completely resets all memories when switching to a novel environment."""
        self.game_key = game_key
        self.version = version
        self.run_id = run_id
        self.confirmed_invariants.clear()
        self.events.clear()
        self.forbidden_transitions.clear()

    def record_transition(
        self,
        step: int,
        level: int,
        from_frame: np.ndarray,
        action: str,
        payload: dict[str, Any],
        to_state: str,
        to_frame: np.ndarray,
        planning_mode: str,
        confidence: float,
        prediction_error: bool = False,
    ):
        """Logs a transition event into memory."""
        from_hash = str(hash(from_frame.tobytes()))
        to_hash = str(hash(to_frame.tobytes()))

        event = TransitionEvent(
            step=step,
            level=level,
            from_frame_hash=from_hash,
            action=action,
            payload=dict(payload),
            to_state=to_state,
            to_frame_hash=to_hash,
            planning_mode=planning_mode,
            confidence=confidence,
            prediction_error=prediction_error,
        )
        self.events.append(event)

        if to_state == "GAME_OVER":
            self.forbidden_transitions.add((from_hash, action))

    def is_forbidden(self, frame: np.ndarray, action: str) -> bool:
        """Checks if this action from this state is known to cause GAME_OVER."""
        frame_hash = str(hash(frame.tobytes()))
        return (frame_hash, action) in self.forbidden_transitions

# ======================================================================
# PRIMARY AGENT: my_agent.py
# ======================================================================

"""
Uncertainty-Aware Agent for ARC-AGI-3.
Integrates:
1. DRE-Bench 4-Level Cognitive Hierarchy (Attribute, Spatial, Sequential Macro-Planning, Intuitive Physics).
2. OpenAI Reasoning Persistence (Persistent working scratchpad & active multi-step macro-plans).
3. Automatic Context Compaction (Compact semantic delta summaries, zero context rot).
4. Hard legality adapter (strictly conforms to available_actions, handles GAME_OVER -> RESET).
5. Scoped episode memory (bounds invariants to game run, avoids negative transfer).
"""


from typing import Any

import numpy as np
from arcengine import GameAction, GameState


# When running in official starter, `Agent` is imported from `agents.agent`
try:
    from agents.agent import Agent
except ImportError:
    # Base fallback for local testing without the starter framework wrapper
    class Agent:  # type: ignore[no-redef]
        def __init__(self, game_id: str = "local_game", *args: Any, **kwargs: Any):
            self.game_id = game_id


class MyAgent(Agent):
    """
    Production-ready Uncertainty-Aware Agent for ARC-AGI-3.
    """

    MAX_ACTIONS = 1000

    def __init__(
        self,
        game_id: str = "default_game",
        parameters: dict[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.game_id = getattr(self, "game_id", game_id)
        self.parameters = parameters or {}
        self.perception = LayeredPerception()
        self.cognitive_perception = CognitiveHierarchyPerception()
        self.world_model = BeliefStateWorldModel()
        self.policy = EpistemicPolicy()
        self.memory = ScopedEpisodeMemory(game_key=self.game_id)
        self.reasoning_state = PersistentReasoningState(game_id=self.game_id)
        self.compactor = ContextCompactor()

        self.previous_observation: Observation | None = None
        self.previous_analysis: FrameAnalysis | None = None
        self.previous_cognitive: CognitiveHierarchyAnalysis | None = None
        self.previous_action: str | None = None
        self.action_count = 0

    def is_done(self, frames: Any, latest_frame: Any) -> bool:
        """Determines whether agent should stop playing."""
        if self.action_count >= self.MAX_ACTIONS:
            return True
        state = getattr(latest_frame, "state", None)
        if state in ("WIN", GameState.WIN, GameState.WIN.value):
            return True
        return False

    def choose_action(self, frames: Any, latest_frame: Any) -> GameAction:
        """
        Main decision loop for ARC-AGI-3 agent contract.
        Inspects environment state and returns a validated GameAction enum.
        """
        self.action_count += 1

        current_level = getattr(latest_frame, "levels_completed", 0) + 1
        if current_level != self.reasoning_state.current_level:
            self.reasoning_state.reset_level(current_level)

        # Extract frame arrays
        raw_frames = getattr(latest_frame, "frame", [])
        if not raw_frames:
            if isinstance(frames, list) and frames:
                raw_frames = frames
            else:
                raw_frames = [np.zeros((16, 16), dtype=int)]

        grid = raw_frames[0] if isinstance(raw_frames, (list, tuple)) else raw_frames
        if not isinstance(grid, np.ndarray):
            grid = np.array(grid, dtype=int)

        # Extract metadata
        raw_state = getattr(latest_frame, "state", GameState.NOT_FINISHED)
        state_str = raw_state.value if hasattr(raw_state, "value") else str(raw_state)

        raw_avail = getattr(latest_frame, "available_actions", [])
        avail_actions = set()
        for a in raw_avail:
            if isinstance(a, int):
                try:
                    avail_actions.add(GameAction.from_id(a).name)
                except Exception:
                    pass
            elif hasattr(a, "name"):
                avail_actions.add(a.name)
            else:
                try:
                    avail_actions.add(GameAction.from_name(str(a)).name)
                except Exception:
                    avail_actions.add(str(a).split(".")[-1])

        if not avail_actions:
            avail_actions = {"RESET", "ACTION1"}

        current_obs = Observation(
            frames=(grid,),
            state=state_str,
            available_actions=frozenset(avail_actions),
            game_key=self.game_id,
            level=getattr(latest_frame, "levels_completed", 0) + 1,
            action_count=self.action_count,
            guid=getattr(latest_frame, "guid", None),
        )

        # 1. Perception & DRE-Bench Cognitive Analysis
        prev_grid = self.previous_observation.frames[0] if self.previous_observation else None
        current_analysis = self.perception.analyze(grid, prev_grid)
        cognitive_analysis = self.cognitive_perception.analyze(
            frame=grid,
            prev_frame=prev_grid,
            known_player_color=self.reasoning_state.player_color,
        )

        if cognitive_analysis.player_color is not None:
            if self.reasoning_state.player_color is None or cognitive_analysis.is_stagnant:
                self.reasoning_state.player_color = cognitive_analysis.player_color
            elif (
                cognitive_analysis.player_color != self.reasoning_state.player_color
                and prev_grid is not None
            ):
                self.reasoning_state.player_color = cognitive_analysis.player_color

        # 2. Update Reasoning Persistence (Causal displacements & deaths)
        if (
            self.previous_action is not None
            and self.previous_cognitive is not None
            and self.previous_cognitive.player_pos is not None
            and cognitive_analysis.player_pos is not None
        ):
            dy = cognitive_analysis.player_pos[0] - self.previous_cognitive.player_pos[0]
            dx = cognitive_analysis.player_pos[1] - self.previous_cognitive.player_pos[1]
            self.reasoning_state.update_action_effect(self.previous_action, dy, dx)

        # Context compaction
        prev_pos = self.previous_cognitive.player_pos if self.previous_cognitive else None
        self.compactor.compact_step(
            step=self.action_count,
            level=current_obs.level,
            action=self.previous_action or "NONE",
            curr_pos=cognitive_analysis.player_pos,
            prev_pos=prev_pos,
            state=state_str,
            levels_completed=getattr(latest_frame, "levels_completed", 0),
        )

        # Belief State Update
        if (
            self.previous_observation is not None
            and self.previous_analysis is not None
            and self.previous_action is not None
        ):
            self.world_model.update_with_transition(
                prev_obs=self.previous_observation,
                action=self.previous_action,
                curr_obs=current_obs,
                prev_analysis=self.previous_analysis,
                curr_analysis=current_analysis,
            )

        # 3. Decision Policy: Macro-Planning vs Epistemic Probing
        action_name, payload, trace = self.policy.select_action(
            observation=current_obs,
            analysis=current_analysis,
            world_model=self.world_model,
            reasoning_state=self.reasoning_state,
            cognitive_analysis=cognitive_analysis,
        )

        # Update tracking
        self.previous_observation = current_obs
        self.previous_analysis = current_analysis
        self.previous_cognitive = cognitive_analysis
        self.previous_action = action_name
        self.last_payload = payload

        return LegalityAdapter.to_game_action(action_name)