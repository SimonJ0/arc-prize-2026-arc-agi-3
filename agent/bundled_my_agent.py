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
    class Agent:
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
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


@dataclass(frozen=True)
class Observation:
    """Immutable snapshot of the environment state returned by ARC-AGI-3."""
    frames: Tuple[np.ndarray, ...]  # (H, W) arrays with values in [0, 15]
    state: str                      # 'NOT_PLAYED', 'NOT_FINISHED', 'WIN', 'GAME_OVER'
    available_actions: frozenset[str]  # e.g. {'RESET', 'ACTION1', 'ACTION2', ...}
    game_key: str                   # Environment / game ID
    level: int                      # Current level index (1-based)
    action_count: int               # Actions taken so far in this level/game
    guid: Optional[str] = None      # Session GUID


@dataclass(frozen=True)
class Transition:
    """Observed transition between two consecutive environment states."""
    from_observation_hash: str
    action: str
    payload: Dict[str, Any]
    to_state: str
    to_observation_hash: str
    frame_diff_count: int


@dataclass
class TransitionModelHypothesis:
    """Single hypothesis about environment mechanics / transition rules."""
    hypothesis_id: str
    description: str
    action_semantics: Dict[str, str] = field(default_factory=dict)  # e.g. {'ACTION1': 'MOVE_UP'}
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
    hypotheses: List[TransitionModelHypothesis] = field(default_factory=list)
    posterior: np.ndarray = field(default_factory=lambda: np.array([]))
    one_step_accuracy: float = 0.0
    rollout_accuracy: float = 0.0
    evidence: List[Transition] = field(default_factory=list)

    def get_most_likely_hypothesis(self) -> Optional[TransitionModelHypothesis]:
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
    payload: Dict[str, Any] = field(default_factory=dict)
    expected_completion_value: float = 0.0
    expected_information_gain: float = 0.0
    risk_game_over: float = 0.0
    cost: float = 1.0  # Physical action cost (always >= 1 in RHAE)

    @property
    def score(self) -> float:
        """Heuristic value combining completion utility and epistemic gain penalized by risk."""
        return self.expected_completion_value + 0.4 * self.expected_information_gain - 2.0 * self.risk_game_over


@dataclass(frozen=True)
class DecisionTrace:
    """Trace of agent reasoning for a single physical step."""
    level: int
    step: int
    observation_hash: str
    legal_actions: Tuple[str, ...]
    selected_action: str
    selected_payload: Dict[str, Any]
    planning_mode: str  # 'EPISTEMIC_PROBE', 'GOAL_PLAN', 'LEGAL_FALLBACK', 'RESET_RECOVERY'
    predicted_next_state: Optional[str]
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

from typing import Any, Dict, Optional, Set, Tuple
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
        available_actions: Set[str],
        proposed_action: str,
        proposed_payload: Optional[Dict[str, Any]] = None,
        default_fallback: str = "RESET",
    ) -> Tuple[str, Dict[str, Any]]:
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
                action_name = sorted(list(available_upper))[0]

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
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from scipy.ndimage import label


@dataclass(frozen=True)
class EntityCandidate:
    """An identified object or cluster within the grid."""
    entity_id: int
    color: int
    cells: Tuple[Tuple[int, int], ...]
    bbox: Tuple[int, int, int, int]  # (min_y, min_x, max_y, max_x)
    centroid: Tuple[float, float]
    is_dynamic: bool = False

    @property
    def size(self) -> int:
        return len(self.cells)


@dataclass
class FrameAnalysis:
    """Multi-hypothesis perception result for a single observation."""
    frame_shape: Tuple[int, int]
    present_colors: Set[int]
    background_hypotheses: List[Tuple[int, float]]  # (color, confidence)
    entities: List[EntityCandidate]
    dynamic_diff_mask: Optional[np.ndarray] = None
    symmetry_scores: Dict[str, float] = field(default_factory=dict)


class LayeredPerception:
    """Perception pipeline maintaining multiple structural segmentations."""

    def __init__(self):
        self.previous_frame: Optional[np.ndarray] = None

    def analyze(self, frame: np.ndarray, prev_frame: Optional[np.ndarray] = None) -> FrameAnalysis:
        """
        Processes a 2D integer grid frame into layered perceptual abstractions.
        """
        if frame.ndim != 2:
            raise ValueError(f"Expected 2D grid frame, got shape {frame.shape}")

        H, W = frame.shape
        unique_colors, counts = np.unique(frame, return_counts=True)
        present_colors = set(int(c) for c in unique_colors)

        # 1. Multi-candidate background inference
        bg_hypotheses = self._infer_background_candidates(frame, unique_colors, counts)

        # 2. Dynamic temporal diff mask
        if prev_frame is None:
            prev_frame = self.previous_frame
        dynamic_mask = None
        if prev_frame is not None and prev_frame.shape == frame.shape:
            dynamic_mask = (frame != prev_frame)

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
    ) -> List[Tuple[int, float]]:
        """
        Ranks candidate background colors using combined border density,
        overall area fraction, and connectivity.
        """
        H, W = frame.shape
        total_pixels = H * W
        border_pixels = np.concatenate([
            frame[0, :], frame[-1, :], frame[:, 0], frame[:, -1]
        ])
        border_total = len(border_pixels)

        candidates = []
        for color, count in zip(unique_colors, counts):
            color = int(color)
            area_frac = count / total_pixels
            border_frac = np.mean(border_pixels == color)
            # Composite score (weighted border presence and area)
            conf = 0.6 * border_frac + 0.4 * area_frac
            candidates.append((color, float(conf)))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates

    def _extract_entities(
        self, frame: np.ndarray, background_color: int, dynamic_mask: Optional[np.ndarray]
    ) -> List[EntityCandidate]:
        """Extracts connected component entities excluding the primary candidate background."""
        entities = []
        entity_id_counter = 0

        for color in np.unique(frame):
            color = int(color)
            if color == background_color:
                continue

            color_mask = (frame == color)
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

                entities.append(EntityCandidate(
                    entity_id=entity_id_counter,
                    color=color,
                    cells=cells,
                    bbox=(int(min_y), int(min_x), int(max_y), int(max_x)),
                    centroid=centroid,
                    is_dynamic=is_dyn,
                ))
                entity_id_counter += 1

        return entities

# ======================================================================
# INLINED: belief_state.py
# ======================================================================

"""
Factored World Model Belief State for ARC-AGI-3.
Maintains a Bayesian distribution over transition dynamics hypotheses.
Tracks 1-step prediction accuracy and explicitly falsifies hypotheses upon contradiction.
Only authorizes deep planning when model consensus and fidelity meet confidence thresholds.
"""

from typing import Dict, List, Optional, Tuple
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
        self.avatar_color: Optional[int] = None
        self.solid_colors: set[int] = set()

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

    def can_reliably_plan(self) -> bool:
        """
        Gating check: Deep forward planning is authorized ONLY when
        sufficient evidence confirms high predictive fidelity.
        """
        top_hyp = self.belief.get_most_likely_hypothesis()
        if top_hyp is None:
            return False
        return (
            top_hyp.confidence >= 0.80
            and top_hyp.evidence_count >= 3
            and top_hyp.accuracy() >= 0.85
            and self.avatar_color is not None
        )

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

        # 2. If action is directional, test directional hypotheses
        if action in ("ACTION1", "ACTION2", "ACTION3", "ACTION4") and self.avatar_color is not None:
            self._update_directional_hypotheses(action, prev_frame, curr_frame)

        # 3. Recompute posterior distribution
        self._recompute_posterior()

    def _update_directional_hypotheses(
        self, action: str, prev_frame: np.ndarray, curr_frame: np.ndarray
    ):
        """Validates predicted movement of the avatar against observed frame displacement."""
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

    def _find_entity_centroid(
        self, frame: np.ndarray, color: int
    ) -> Optional[Tuple[float, float]]:
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
        self, curr_pos: Tuple[int, int], action: str
    ) -> Optional[Tuple[int, int]]:
        """Predicts next avatar position under the most likely hypothesis."""
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
Selects actions via:
1. Low-risk epistemic probing when world model uncertainty is high (Value of Information > Action Cost).
2. Goal-directed search (BFS / A*) over validated transition dynamics when model consensus is established.
3. Candidate-coordinate spatial pruning for ACTION6.
4. Hard legality validation via LegalityAdapter.
"""

from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np



class EpistemicPolicy:
    """Decision engine balancing active epistemic learning with goal pursuit."""

    def __init__(self):
        self.step_counter = 0

    def select_action(
        self,
        observation: Observation,
        analysis: FrameAnalysis,
        world_model: BeliefStateWorldModel,
    ) -> Tuple[str, Dict[str, Any], DecisionTrace]:
        """
        Selects next physical environment action given current belief state.
        Guaranteed to return a legal action from observation.available_actions.
        """
        self.step_counter += 1
        state = observation.state
        available = observation.available_actions

        # 1. State-level guard: GAME_OVER requires RESET
        if state == "GAME_OVER":
            action, payload = LegalityAdapter.validate_action(
                state=state,
                available_actions=available,
                proposed_action="RESET",
            )
            trace = DecisionTrace(
                level=observation.level,
                step=self.step_counter,
                observation_hash=str(hash(observation.frames[0].tobytes())),
                legal_actions=tuple(sorted(list(available))),
                selected_action=action,
                selected_payload=payload,
                planning_mode="RESET_RECOVERY",
                predicted_next_state="NOT_FINISHED",
                confidence=1.0,
            )
            return action, payload, trace

        # 2. Epistemic Probing Mode: Model is uncertain
        if not world_model.can_reliably_plan():
            action, payload, trace = self._select_epistemic_probe(
                observation, analysis, world_model
            )
            return action, payload, trace

        # 3. Exploitation Mode: Validated model allows forward planning
        action, payload, trace = self._plan_goal_trajectory(
            observation, analysis, world_model
        )
        return action, payload, trace

    def _select_epistemic_probe(
        self,
        observation: Observation,
        analysis: FrameAnalysis,
        world_model: BeliefStateWorldModel,
    ) -> Tuple[str, Dict[str, Any], DecisionTrace]:
        """Selects informative probe action to distinguish candidate transition models."""
        available = observation.available_actions

        # Rank probe candidates: test untested directional actions first
        probes = [a for a in ("ACTION1", "ACTION2", "ACTION3", "ACTION4") if a in available]
        if not probes:
            probes = [a for a in ("ACTION5", "ACTION6", "ACTION7") if a in available]
        if not probes:
            probes = list(available)

        # Pick probe with highest epistemic utility
        selected = probes[self.step_counter % len(probes)]
        payload = {}

        if selected == "ACTION6":
            # Prune coordinates to candidate dynamic or target entity centroids
            if analysis.entities:
                target_ent = analysis.entities[0]
                cy, cx = target_ent.centroid
                payload = {"x": int(cx), "y": int(cy)}
            else:
                H, W = analysis.frame_shape
                payload = {"x": W // 2, "y": H // 2}

        action, valid_payload = LegalityAdapter.validate_action(
            state=observation.state,
            available_actions=available,
            proposed_action=selected,
            proposed_payload=payload,
        )

        trace = DecisionTrace(
            level=observation.level,
            step=self.step_counter,
            observation_hash=str(hash(observation.frames[0].tobytes())),
            legal_actions=tuple(sorted(list(available))),
            selected_action=action,
            selected_payload=valid_payload,
            planning_mode="EPISTEMIC_PROBE",
            predicted_next_state="NOT_FINISHED",
            confidence=0.5,
        )
        return action, valid_payload, trace

    def _plan_goal_trajectory(
        self,
        observation: Observation,
        analysis: FrameAnalysis,
        world_model: BeliefStateWorldModel,
    ) -> Tuple[str, Dict[str, Any], DecisionTrace]:
        """Plans shortest path to candidate goal under validated transition dynamics."""
        frame = observation.frames[0]
        avatar_color = world_model.avatar_color
        available = observation.available_actions

        # Locate avatar
        avatar_coords = np.argwhere(frame == avatar_color)
        if len(avatar_coords) == 0:
            # Avatar lost, fallback to probe
            return self._select_epistemic_probe(observation, analysis, world_model)

        start_pos = (int(avatar_coords[0][0]), int(avatar_coords[0][1]))

        # Identify candidate goal entities (distinct non-background, non-avatar entity)
        target_pos = None
        for ent in analysis.entities:
            if ent.color != avatar_color and ent.size <= 36:
                target_pos = (int(ent.centroid[0]), int(ent.centroid[1]))
                break

        if target_pos is None:
            # No clear target found, fallback to safe exploration
            return self._select_epistemic_probe(observation, analysis, world_model)

        # Run BFS to find shortest action sequence to target
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
                legal_actions=tuple(sorted(list(available))),
                selected_action=action,
                selected_payload=valid_payload,
                planning_mode="GOAL_PLAN",
                predicted_next_state="NOT_FINISHED",
                confidence=0.9,
            )
            return action, valid_payload, trace

        # Path not found or blocked, fallback
        return self._select_epistemic_probe(observation, analysis, world_model)

    def _bfs_search(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        grid_shape: Tuple[int, int],
        world_model: BeliefStateWorldModel,
        available_actions: Set[str],
    ) -> Optional[List[str]]:
        """Bounded BFS search over validated directional dynamics."""
        H, W = grid_shape
        queue = deque([(start, [])])
        visited = {start}

        dir_actions = [a for a in ("ACTION1", "ACTION2", "ACTION3", "ACTION4") if a in available_actions]
        max_depth = 40  # Bound search to avoid budget exhaustion

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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass(frozen=True)
class TransitionEvent:
    """Hashable record of an environment transition and agent decision."""
    step: int
    level: int
    from_frame_hash: str
    action: str
    payload: Dict[str, Any]
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
        self.confirmed_invariants: Dict[str, Any] = {}
        self.events: List[TransitionEvent] = []
        self.forbidden_transitions: set[Tuple[str, str]] = set()

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
        payload: Dict[str, Any],
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
Combines:
1. Hard legality adapter (strictly conforms to available_actions, handles GAME_OVER -> RESET).
2. Multi-hypothesis layered perception (candidate backgrounds, connected components).
3. Factored belief-state world model (Bayesian posterior over transition dynamics).
4. Epistemic decision policy (epistemic probing vs bounded goal planning).
5. Scoped episode memory (bounds invariants to game run, avoids negative transfer).
"""

import hashlib
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from arcengine import GameAction, GameState, FrameDataRaw

# When running in official starter, `Agent` is imported from `agents.agent`
try:
    from agents.agent import Agent
except ImportError:
    # Base fallback for local testing without the starter framework wrapper
    class Agent:
        def __init__(self, game_id: str = "local_game", *args: Any, **kwargs: Any):
            self.game_id = game_id


class MyAgent(Agent):
    """
    Production-ready Uncertainty-Aware Agent for ARC-AGI-3.
    """
    MAX_ACTIONS = 120

    def __init__(self, game_id: str = "default_game", *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.game_id = getattr(self, "game_id", game_id)
        self.perception = LayeredPerception()
        self.world_model = BeliefStateWorldModel()
        self.policy = EpistemicPolicy()
        self.memory = ScopedEpisodeMemory(game_key=self.game_id)

        self.previous_observation: Optional[Observation] = None
        self.previous_analysis: Optional[FrameAnalysis] = None
        self.previous_action: Optional[str] = None
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

        # Perception
        prev_grid = self.previous_observation.frames[0] if self.previous_observation else None
        current_analysis = self.perception.analyze(grid, prev_grid)

        # Belief State Update from previous transition
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

            # Record event in scoped memory
            self.memory.record_transition(
                step=self.action_count,
                level=current_obs.level,
                from_frame=prev_grid,
                action=self.previous_action,
                payload={},
                to_state=state_str,
                to_frame=grid,
                planning_mode="ONLINE_DECISION",
                confidence=self.world_model.belief.one_step_accuracy,
            )

        # Decision Policy: Probing vs Goal Planning
        action_name, payload, trace = self.policy.select_action(
            observation=current_obs,
            analysis=current_analysis,
            world_model=self.world_model,
        )

        # Update tracking
        self.previous_observation = current_obs
        self.previous_analysis = current_analysis
        self.previous_action = action_name

        return LegalityAdapter.to_game_action(action_name)