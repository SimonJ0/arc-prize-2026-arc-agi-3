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
from enum import Enum
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np
from scipy.ndimage import label
from arcengine import GameAction, GameState, FrameDataRaw


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
    def to_game_action(
        cls, action_str: str, payload: dict[str, Any] | None = None
    ) -> GameAction:
        """Converts string action to arcengine GameAction enum, setting data for complex actions."""
        key = action_str.upper().split(".")[-1]
        try:
            action = GameAction.from_name(key)
        except Exception:
            action = GameAction.RESET

        if payload:
            if action == GameAction.ACTION6:
                x = int(payload.get("x", 0))
                y = int(payload.get("y", 0))
                action.set_data({"x": x, "y": y})
            if "reasoning" in payload:
                setattr(action, "reasoning", payload["reasoning"])

        return action

# ======================================================================
# INLINED: replay_logger.py
# ======================================================================

"""
ARC-AGI-3 Step-Level Replay Diagnostic Logger.
Instruments agent execution to capture:
1. Per-step observations, predictions, and decomposed prediction errors.
2. Active hypothesis distributions P(H_i).
3. Action rationale and risk tiers (Levels 0-5).
4. Deadlock classifications (Types 1-6).
5. Automated post-hoc failure attribution (8-tier taxonomy).
"""


import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


class ActionRiskTier(int, Enum):
    LEVEL_0_SAFE_OBSERVATION = 0
    LEVEL_1_REVERSIBLE_MOVE = 1
    LEVEL_2_LOW_RISK_PROBE = 2
    LEVEL_3_ACTIVE_INTERVENTION = 3
    LEVEL_4_IRREVERSIBLE_ACTION = 4
    LEVEL_5_GOAL_COMMITMENT = 5


class FailureTaxonomy(str, Enum):
    PERCEPTION_FAILURE = "PERCEPTION_FAILURE"
    MODEL_FAILURE = "MODEL_FAILURE"
    GOAL_FAILURE = "GOAL_FAILURE"
    EXPLORATION_FAILURE = "EXPLORATION_FAILURE"
    PLANNING_FAILURE = "PLANNING_FAILURE"
    EXECUTION_FAILURE = "EXECUTION_FAILURE"
    MEMORY_FAILURE = "MEMORY_FAILURE"
    TRANSFER_FAILURE = "TRANSFER_FAILURE"
    SUCCESS = "SUCCESS"


@dataclass
class PredictionError:
    """Structured decomposed prediction error vector."""
    position_error: float = 0.0
    appearance_error: float = 0.0
    disappearance_error: float = 0.0
    color_error: float = 0.0
    collision_error: float = 0.0
    goal_progress_error: float = 0.0

    @property
    def total_magnitude(self) -> float:
        return float(
            np.sqrt(
                self.position_error**2
                + self.appearance_error**2
                + self.disappearance_error**2
                + self.color_error**2
                + self.collision_error**2
                + self.goal_progress_error**2
            )
        )

    def to_dict(self) -> dict[str, float]:
        d = asdict(self)
        d["total_magnitude"] = round(self.total_magnitude, 4)
        return d


@dataclass
class StepReplayRecord:
    """Detailed per-step telemetry record."""
    step: int
    level: int
    observation_hash: str
    action: str
    payload: dict[str, Any]
    risk_tier: int
    rationale: str
    predicted_avatar_pos: tuple[int, int] | None
    actual_avatar_pos: tuple[int, int] | None
    prediction_error: PredictionError
    active_hypotheses: dict[str, float]  # hypothesis_id -> posterior
    active_goal: tuple[int, int] | None
    goal_confidence: float
    deadlock_type: str | None = None
    state: str = "NOT_FINISHED"
    levels_completed: int = 0
    decision_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "level": self.level,
            "observation_hash": self.observation_hash,
            "action": self.action,
            "payload": self.payload,
            "risk_tier": self.risk_tier,
            "rationale": self.rationale,
            "predicted_avatar_pos": self.predicted_avatar_pos,
            "actual_avatar_pos": self.actual_avatar_pos,
            "prediction_error": self.prediction_error.to_dict(),
            "active_hypotheses": {k: round(v, 4) for k, v in self.active_hypotheses.items()},
            "active_goal": self.active_goal,
            "goal_confidence": round(self.goal_confidence, 4),
            "deadlock_type": self.deadlock_type,
            "state": self.state,
            "levels_completed": self.levels_completed,
            "decision_latency_ms": round(self.decision_latency_ms, 2),
        }


class ReplayLogger:
    """Captures, serializes, and analyzes execution replays for diagnostic feedback."""

    def __init__(self, game_id: str, output_dir: str = "reports/replays"):
        self.game_id = game_id
        self.output_dir = ROOT / output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[StepReplayRecord] = []
        self.start_time = datetime.now(timezone.utc)
        self.hypotheses_falsified = 0
        self.goals_falsified = 0
        self.deadlocks_encountered = 0

    def log_step(
        self,
        step: int,
        level: int,
        observation_hash: str,
        action: str,
        payload: dict[str, Any],
        risk_tier: int = ActionRiskTier.LEVEL_1_REVERSIBLE_MOVE.value,
        rationale: str = "epistemic_probe",
        predicted_avatar_pos: tuple[int, int] | None = None,
        actual_avatar_pos: tuple[int, int] | None = None,
        prediction_error: PredictionError | None = None,
        active_hypotheses: dict[str, float] | None = None,
        active_goal: tuple[int, int] | None = None,
        goal_confidence: float = 0.5,
        deadlock_type: str | None = None,
        state: str = "NOT_FINISHED",
        levels_completed: int = 0,
        decision_latency_ms: float = 0.0,
    ) -> StepReplayRecord:
        record = StepReplayRecord(
            step=step,
            level=level,
            observation_hash=observation_hash,
            action=action,
            payload=payload,
            risk_tier=risk_tier,
            rationale=rationale,
            predicted_avatar_pos=predicted_avatar_pos,
            actual_avatar_pos=actual_avatar_pos,
            prediction_error=prediction_error or PredictionError(),
            active_hypotheses=active_hypotheses or {},
            active_goal=active_goal,
            goal_confidence=goal_confidence,
            deadlock_type=deadlock_type,
            state=state,
            levels_completed=levels_completed,
            decision_latency_ms=decision_latency_ms,
        )
        self.records.append(record)
        if deadlock_type:
            self.deadlocks_encountered += 1
        return record

    def classify_failure(
        self,
        final_state: str,
        total_actions: int,
        max_actions: int,
        levels_completed: int,
        initial_levels_completed: int = 0,
    ) -> FailureTaxonomy:
        """Determines the primary failure root cause using execution evidence."""
        if final_state in ("WIN", "GameState.WIN") or levels_completed > initial_levels_completed:
            return FailureTaxonomy.SUCCESS

        if not self.records:
            return FailureTaxonomy.PERCEPTION_FAILURE

        # 1. Check for perceptual failure: avatar position was never detected
        detected_avatar = any(r.actual_avatar_pos is not None for r in self.records)
        if not detected_avatar:
            return FailureTaxonomy.PERCEPTION_FAILURE

        # 2. Check for memory / transfer failure across level advancement
        if initial_levels_completed > 0 and levels_completed == initial_levels_completed:
            return FailureTaxonomy.TRANSFER_FAILURE

        # 3. Check for execution / planning failure (frequent collisions or deadlocks)
        deadlock_ratio = self.deadlocks_encountered / max(1, len(self.records))
        if deadlock_ratio > 0.35:
            return FailureTaxonomy.PLANNING_FAILURE

        # 4. Check for high model prediction error
        mean_pred_error = float(
            np.mean([r.prediction_error.total_magnitude for r in self.records])
        )
        if mean_pred_error > 1.5:
            return FailureTaxonomy.MODEL_FAILURE

        # 5. Check if goals were constantly falsified without progress
        if self.goals_falsified >= 4:
            return FailureTaxonomy.GOAL_FAILURE

        # 6. Default to exploration exhaustion
        if total_actions >= max_actions:
            return FailureTaxonomy.EXPLORATION_FAILURE

        return FailureTaxonomy.EXECUTION_FAILURE

    def save_replay(
        self,
        final_state: str = "NOT_FINISHED",
        levels_completed: int = 0,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Saves JSONL replay file with structured diagnostics."""
        timestamp_str = self.start_time.strftime("%Y%m%d_%H%M%S")
        safe_game_id = self.game_id.replace("/", "_").replace("\\", "_")
        replay_path = self.output_dir / f"{safe_game_id}_{timestamp_str}.jsonl"

        failure_cause = self.classify_failure(
            final_state=final_state,
            total_actions=len(self.records),
            max_actions=len(self.records),
            levels_completed=levels_completed,
        )

        metadata = {
            "game_id": self.game_id,
            "timestamp": self.start_time.isoformat(),
            "total_steps": len(self.records),
            "levels_completed": levels_completed,
            "final_state": final_state,
            "failure_taxonomy": failure_cause.value,
            "deadlocks_count": self.deadlocks_encountered,
            "hypotheses_falsified": self.hypotheses_falsified,
            "goals_falsified": self.goals_falsified,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        with open(replay_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"_metadata": metadata}) + "\n")
            for r in self.records:
                f.write(json.dumps(r.to_dict()) + "\n")

        return replay_path

# ======================================================================
# INLINED: effect_taxonomy.py
# ======================================================================

"""
Structured Effect Taxonomy for ARC-AGI-3 (Baseline 3.0).
Replaces coarse binary delta threshold (diff_count > 2) with a rich 8-class taxonomy:
  NONE, UI_NOISE, LOCAL_MUTATION, STRUCTURAL_MUTATION, GLOBAL_MUTATION,
  DELAYED_EFFECT, LEVEL_ADVANCE, GAME_OVER.
"""


from enum import Enum


class EffectType(str, Enum):
    """Classification of state transition consequences."""

    NONE = "NONE"                          # Δ == 0: no observable effect
    UI_NOISE = "UI_NOISE"                  # 0 < Δ <= 2: counter tick, animation flicker
    LOCAL_MUTATION = "LOCAL_MUTATION"      # 2 < Δ <= 20: single object/handle changed
    STRUCTURAL_MUTATION = "STRUCTURAL"     # 20 < Δ <= 100: board geometry / puzzle rotation
    GLOBAL_MUTATION = "GLOBAL_MUTATION"    # Δ > 100: massive multi-object transformation / clear
    DELAYED_EFFECT = "DELAYED_EFFECT"      # Δ == 0 immediately, consequence manifests later
    LEVEL_ADVANCE = "LEVEL_ADVANCE"        # Goal condition met, advancing level
    GAME_OVER = "GAME_OVER"               # Terminal failure, lethal trap, or collision quota exceeded


# Threshold boundaries
UI_NOISE_CEIL: int = 2
LOCAL_CEIL: int = 20
STRUCTURAL_CEIL: int = 100


def classify_effect(
    diff_count: int,
    state: str = "NOT_FINISHED",
    level_advanced: bool = False,
    is_lethal: bool = False,
    prev_diff: int = 0,
) -> EffectType:
    """
    Categorizes the observable impact of an action into EffectType.

    Args:
        diff_count: Number of pixels altered between pre- and post-action observations.
        state: Environment state string ("NOT_FINISHED", "WIN", "GAME_OVER", etc.).
        level_advanced: True if this action triggered level advancement.
        is_lethal: True if the action triggered a lethal obstacle penalty or fatal reset.
        prev_diff: Diff count from previous step for delayed effect correlation.

    Returns:
        EffectType enumeration member.
    """
    # 1. Terminal / Fatal Outcomes
    if is_lethal or state in ("GAME_OVER", "GameState.GAME_OVER"):
        return EffectType.GAME_OVER

    # 2. Level Advancement
    if level_advanced or state in ("WIN", "GameState.WIN"):
        return EffectType.LEVEL_ADVANCE

    # 3. Observable pixel delta classification
    if diff_count == 0:
        return EffectType.NONE

    if diff_count <= UI_NOISE_CEIL:
        return EffectType.UI_NOISE

    if diff_count <= LOCAL_CEIL:
        return EffectType.LOCAL_MUTATION

    if diff_count <= STRUCTURAL_CEIL:
        return EffectType.STRUCTURAL_MUTATION

    return EffectType.GLOBAL_MUTATION

# ======================================================================
# INLINED: mechanism_memory.py
# ======================================================================

"""
Causal Mechanism Memory for ARC-AGI-3 (Baseline 3.0 B3.02).
Maintains an indexed ledger of causal interventions, mapping:
  (EntitySignature, Action) -> (EffectType, LatentVariableDelta, ChangedEntities, Reversibility).
Enables the agent to infer causal control systems rather than raw coordinate blacklists.
"""


from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any



@dataclass(frozen=True)
class EntitySignature:
    """
    Abstract invariant representation of an entity's structural morphology.
    Allows transfer and equivalence grouping regardless of transient pixel coordinates.
    """

    color: int
    size_bucket: str          # "tiny" (<10), "small" (10-50), "medium" (50-120), "large" (>120)
    aspect_ratio_bucket: str  # "square" (0.8-1.2), "wide" (>1.2), "tall" (<0.8)
    solidity_bucket: str      # "solid" (>=0.8), "sparse" (<0.8)

    @classmethod
    def from_entity(cls, entity: Any) -> EntitySignature:
        """Constructs an invariant signature from an EntityCandidate or AttributeProfile."""
        color = getattr(entity, "color", 0)
        size = getattr(entity, "size", 1)

        # Size bucket
        if size < 10:
            size_b = "tiny"
        elif size <= 50:
            size_b = "small"
        elif size <= 120:
            size_b = "medium"
        else:
            size_b = "large"

        # Aspect ratio & solidity
        bbox = getattr(entity, "bbox", (0, 0, 1, 1))
        h = max(1, bbox[2] - bbox[0])
        w = max(1, bbox[3] - bbox[1])
        ar = w / h
        if 0.8 <= ar <= 1.2:
            ar_b = "square"
        elif ar > 1.2:
            ar_b = "wide"
        else:
            ar_b = "tall"

        bbox_area = max(1, h * w)
        solidity = size / bbox_area
        sol_b = "solid" if solidity >= 0.8 else "sparse"

        return cls(
            color=color,
            size_bucket=size_b,
            aspect_ratio_bucket=ar_b,
            solidity_bucket=sol_b,
        )


@dataclass
class MechanismRecord:
    """Detailed record of an intervention outcome in the causal ledger."""

    step: int
    action: str
    target_coord: tuple[int, int]
    target_entity: EntitySignature | None
    effect_type: EffectType
    diff_count: int
    changed_entity_signatures: list[EntitySignature] = field(default_factory=list)
    latent_variable_id: str | None = None
    latent_delta: int | None = None
    level_advanced: bool = False
    is_lethal: bool = False
    reversible: bool | None = None
    inverse_action_coord: tuple[int, int] | None = None
    pre_frame_hash: str = ""
    post_frame_hash: str = ""


@dataclass
class MechanismHypothesis:
    """Separates causal mechanism certainty from goal relevance certainty."""

    mechanism_id: str
    target_coord: tuple[int, int]
    entity_signature: EntitySignature | None
    dominant_effect: EffectType
    causal_confidence: float = 0.5  # "I know what this interaction does"
    goal_relevance: float = 0.5     # "I know this interaction predicts winning"
    observations: int = 0
    concordant_effects: int = 0
    progress_correlations: int = 0
    is_lethal: bool = False
    reversible: bool = False

    def update(self, effect: EffectType, level_advanced: bool, is_lethal: bool = False):
        self.observations += 1
        if is_lethal:
            self.is_lethal = True
            self.goal_relevance = 0.0
            self.causal_confidence = 1.0
            return

        if effect == self.dominant_effect:
            self.concordant_effects += 1
        elif self.concordant_effects == 0:
            self.dominant_effect = effect
            self.concordant_effects = 1

        # Causal confidence: how consistently does it produce this effect?
        self.causal_confidence = (self.concordant_effects + 1.0) / (self.observations + 2.0)

        # Goal relevance: does this effect advance the level or transform structural/global board geometry?
        if level_advanced or effect in (
            EffectType.STRUCTURAL_MUTATION,
            EffectType.GLOBAL_MUTATION,
            EffectType.LEVEL_ADVANCE,
        ):
            self.progress_correlations += 1
        self.goal_relevance = (self.progress_correlations + 1.0) / (self.observations + 2.0)

    def can_reliably_exploit(self) -> bool:
        """Only exploit when both mechanism and goal relevance confidence exceed threshold."""
        return (
            not self.is_lethal
            and self.causal_confidence >= 0.60
            and self.goal_relevance >= 0.50
        )


class MechanismMemory:
    """
    Indexed causal transition ledger.
    Indexes interactions by entity signature, effect type, and spatial coordinates.
    """

    def __init__(self, max_records: int = 500):
        self.max_records = max_records
        self.records: list[MechanismRecord] = []
        self.signature_to_effects: dict[EntitySignature, list[EffectType]] = defaultdict(list)
        self.active_controllers: set[tuple[int, int]] = set()
        self.inert_signatures: set[EntitySignature] = set()
        self.active_signatures: set[EntitySignature] = set()
        self.hypotheses: dict[tuple[int, int], MechanismHypothesis] = {}

    def record_transition(
        self,
        step: int,
        action: str,
        target_coord: tuple[int, int],
        target_entity: EntitySignature | None,
        effect_type: EffectType,
        diff_count: int,
        changed_entity_signatures: list[EntitySignature] | None = None,
        latent_variable_id: str | None = None,
        latent_delta: int | None = None,
        level_advanced: bool = False,
        is_lethal: bool = False,
        pre_frame_hash: str = "",
        post_frame_hash: str = "",
    ) -> MechanismRecord:
        """Appends and indexes a causal transition in the ledger."""
        rec = MechanismRecord(
            step=step,
            action=action,
            target_coord=target_coord,
            target_entity=target_entity,
            effect_type=effect_type,
            diff_count=diff_count,
            changed_entity_signatures=changed_entity_signatures or [],
            latent_variable_id=latent_variable_id,
            latent_delta=latent_delta,
            level_advanced=level_advanced,
            is_lethal=is_lethal,
            pre_frame_hash=pre_frame_hash,
            post_frame_hash=post_frame_hash,
        )
        self.records.append(rec)
        if len(self.records) > self.max_records:
            self.records.pop(0)

        # Indexing
        if target_entity is not None:
            self.signature_to_effects[target_entity].append(effect_type)

            if effect_type in (EffectType.NONE, EffectType.UI_NOISE):
                self.inert_signatures.add(target_entity)
            elif effect_type in (
                EffectType.LOCAL_MUTATION,
                EffectType.STRUCTURAL_MUTATION,
                EffectType.GLOBAL_MUTATION,
                EffectType.LEVEL_ADVANCE,
            ):
                self.active_signatures.add(target_entity)
                self.active_controllers.add(target_coord)
                if target_entity in self.inert_signatures:
                    self.inert_signatures.remove(target_entity)

        # Update hypothesis for this coordinate
        if target_coord not in self.hypotheses:
            self.hypotheses[target_coord] = MechanismHypothesis(
                mechanism_id=f"mech_{target_coord[0]}_{target_coord[1]}",
                target_coord=target_coord,
                entity_signature=target_entity,
                dominant_effect=effect_type,
            )
        self.hypotheses[target_coord].update(
            effect=effect_type,
            level_advanced=level_advanced,
            is_lethal=is_lethal,
        )

        return rec

    def get_hypothesis(self, coord: tuple[int, int]) -> MechanismHypothesis | None:
        """Returns the MechanismHypothesis for this coordinate if observed."""
        return self.hypotheses.get(coord)

    def get_exploitable_controllers(self) -> list[tuple[int, int]]:
        """Returns coordinates where both causal and goal relevance confidence permit exploitation."""
        return [c for c, hyp in self.hypotheses.items() if hyp.can_reliably_exploit()]

    def get_by_entity_signature(self, sig: EntitySignature) -> list[MechanismRecord]:
        """Returns all historical transitions targeting entities with matching signature."""
        return [r for r in self.records if r.target_entity == sig]

    def get_by_effect_type(self, effect_type: EffectType) -> list[MechanismRecord]:
        """Returns all historical transitions yielding the specified effect type."""
        return [r for r in self.records if r.effect_type == effect_type]

    def get_active_controllers(self) -> list[tuple[int, int]]:
        """Returns verified active coordinate controllers."""
        return list(self.active_controllers)

    def get_inert_signatures(self) -> set[EntitySignature]:
        """Returns entity signatures confirmed to produce no meaningful state change."""
        return set(self.inert_signatures)

    def is_signature_inert(self, sig: EntitySignature) -> bool:
        """Returns True if the entity signature is confirmed inert."""
        return sig in self.inert_signatures

    def is_signature_active(self, sig: EntitySignature) -> bool:
        """Returns True if the entity signature is confirmed active."""
        return sig in self.active_signatures

    def compute_equivalence_classes(self, entities: Sequence[Any]) -> dict[EntitySignature, list[Any]]:
        """Groups candidate entities into morphological equivalence classes."""
        classes: dict[EntitySignature, list[Any]] = defaultdict(list)
        for ent in entities:
            sig = EntitySignature.from_entity(ent)
            classes[sig].append(ent)
        return dict(classes)

    def filter_inert_classes(self, entities: Sequence[Any]) -> list[Any]:
        """
        Prunes all entities belonging to an equivalence class already confirmed inert.
        If all entities would be pruned, returns the original list as fallback.
        """
        if not self.inert_signatures:
            return list(entities)

        viable = []
        for ent in entities:
            sig = EntitySignature.from_entity(ent)
            if sig not in self.inert_signatures:
                viable.append(ent)

        return viable if viable else list(entities)

    def compress_to_symbolic(self) -> str:
        """
        Compresses the causal ledger into compact Astra-style symbolic shorthand.
        Example: 'CTRL=[(32,5):STRU,(32,58):STRU] INERT_SIGS=1 ACTIVE_SIGS=1'
        """
        ctrl_parts = [
            f"({c[0]},{c[1]}):{self.hypotheses[c].dominant_effect.value[:4]}"
            for c in sorted(self.active_controllers)
            if c in self.hypotheses
        ]
        ctrl_str = ",".join(ctrl_parts) if ctrl_parts else "none"
        inert_count = len(self.inert_signatures)
        active_count = len(self.active_signatures)

        return f"CTRL=[{ctrl_str}] INERT_SIGS={inert_count} ACTIVE_SIGS={active_count}"

    def reset_level(self, keep_abstract_signatures: bool = True):
        """
        Resets level-scoped coordinates while optionally preserving abstract entity signatures.
        """
        self.active_controllers.clear()
        self.hypotheses.clear()
        if not keep_abstract_signatures:
            self.records.clear()
            self.signature_to_effects.clear()
            self.inert_signatures.clear()
            self.active_signatures.clear()

# ======================================================================
# INLINED: structured_belief.py
# ======================================================================

"""
5-Tier Structured Memory & Belief State Architecture for ARC-AGI-3.
Replaces unstructured scratchpad with 5 dedicated cognitive memory modules:
  A. Perceptual Memory (Entities, positions, morphology, spatial diffs)
  B. Causal Memory (Action-delta transitions, causal intervention attribution)
  C. Hypothesis Memory (Bayesian posteriors over kinematics, mechanics, goals)
  D. Goal Memory (Candidate goal coordinates, evidence, precondition chains)
  E. Skill Memory (Cross-level invariant relational rules)
"""


import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np



# ---------------------------------------------------------------------------
# A. Perceptual Memory
# ---------------------------------------------------------------------------

@dataclass
class PerceptualEntityRecord:
    entity_id: int
    color: int
    size: int
    centroid: tuple[int, int]
    bbox: tuple[int, int, int, int]
    is_dynamic: bool
    last_seen_step: int


class PerceptualMemory:
    """Tracks spatio-temporal entity profiles over a sliding window."""

    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.frame_history: deque[np.ndarray] = deque(maxlen=window_size)
        self.player_pos_history: deque[tuple[int, int]] = deque(maxlen=window_size)
        self.entities_by_id: dict[int, PerceptualEntityRecord] = {}
        self.avatar_color: int | None = None
        self.background_color: int = 0

    def update(
        self,
        grid: np.ndarray,
        entities: list[Any],
        player_pos: tuple[int, int] | None,
        player_color: int | None,
        bg_color: int,
        step: int,
    ):
        self.frame_history.append(grid.copy())
        self.background_color = bg_color
        if player_color is not None:
            self.avatar_color = player_color
        if player_pos is not None:
            self.player_pos_history.append(player_pos)

        # Update entity records
        for ent in entities:
            eid = getattr(ent, "entity_id", id(ent))
            color = getattr(ent, "color", 0)
            size = getattr(ent, "size", 1)
            centroid = getattr(ent, "centroid", (0, 0))
            bbox = getattr(ent, "bbox", (0, 0, 0, 0))
            is_dyn = getattr(ent, "is_dynamic", False)
            cy, cx = int(round(centroid[0])), int(round(centroid[1]))

            self.entities_by_id[eid] = PerceptualEntityRecord(
                entity_id=eid,
                color=color,
                size=size,
                centroid=(cy, cx),
                bbox=bbox,
                is_dynamic=is_dyn,
                last_seen_step=step,
            )

    def get_current_avatar_pos(self) -> tuple[int, int] | None:
        return self.player_pos_history[-1] if self.player_pos_history else None


# ---------------------------------------------------------------------------
# B. Causal Memory
# ---------------------------------------------------------------------------

@dataclass
class CausalTransition:
    step: int
    action: str
    payload: dict[str, Any]
    prev_player_pos: tuple[int, int] | None
    next_player_pos: tuple[int, int] | None
    avatar_delta: tuple[int, int]  # (dy, dx)
    diff_pixel_count: int
    is_direct_intervention: bool


class CausalMemory:
    """Maintains transition tuples and causal attribution flags."""

    def __init__(self, max_records: int = 200):
        self.transitions: list[CausalTransition] = []
        self.max_records = max_records
        self.action_success_counts: dict[str, int] = defaultdict(int)
        self.action_stationary_counts: dict[str, int] = defaultdict(int)

    def record(
        self,
        step: int,
        action: str,
        payload: dict[str, Any],
        prev_pos: tuple[int, int] | None,
        curr_pos: tuple[int, int] | None,
        diff_pixel_count: int = 0,
    ) -> CausalTransition:
        dy, dx = (0, 0)
        if prev_pos is not None and curr_pos is not None:
            dy = curr_pos[0] - prev_pos[0]
            dx = curr_pos[1] - prev_pos[1]

        is_direct = (dy != 0 or dx != 0 or action == "ACTION6")

        if (dy, dx) != (0, 0):
            self.action_success_counts[action] += 1
        else:
            self.action_stationary_counts[action] += 1

        trans = CausalTransition(
            step=step,
            action=action,
            payload=payload,
            prev_player_pos=prev_pos,
            next_player_pos=curr_pos,
            avatar_delta=(dy, dx),
            diff_pixel_count=diff_pixel_count,
            is_direct_intervention=is_direct,
        )
        self.transitions.append(trans)
        if len(self.transitions) > self.max_records:
            self.transitions.pop(0)
        return trans


# ---------------------------------------------------------------------------
# C. Hypothesis Memory
# ---------------------------------------------------------------------------

@dataclass
class HypothesisRecord:
    hypothesis_id: str
    category: str  # "kinematics", "barrier", "goal_trigger", "precondition"
    description: str
    posterior: float = 0.5
    evidence_for: int = 0
    evidence_against: int = 0
    falsified: bool = False
    details: dict[str, Any] = field(default_factory=dict)


class HypothesisMemory:
    """Bayesian hypothesis tracker with normalized posteriors and falsification."""

    def __init__(self):
        self.hypotheses: dict[str, HypothesisRecord] = {}

    def register(
        self,
        hypothesis_id: str,
        category: str,
        description: str,
        prior: float = 0.5,
        details: dict[str, Any] | None = None,
    ):
        if hypothesis_id not in self.hypotheses:
            self.hypotheses[hypothesis_id] = HypothesisRecord(
                hypothesis_id=hypothesis_id,
                category=category,
                description=description,
                posterior=prior,
                details=details or {},
            )

    def update_evidence(self, hypothesis_id: str, supported: bool, weight: float = 1.0):
        if hypothesis_id not in self.hypotheses:
            return
        hyp = self.hypotheses[hypothesis_id]
        if hyp.falsified:
            return

        if supported:
            hyp.evidence_for += 1
            # Multiplicative Bayesian odds update
            odds = (hyp.posterior / max(1e-6, 1.0 - hyp.posterior)) * (1.0 + 0.5 * weight)
            hyp.posterior = min(0.99, odds / (1.0 + odds))
        else:
            hyp.evidence_against += 1
            odds = (hyp.posterior / max(1e-6, 1.0 - hyp.posterior)) * (1.0 / (1.0 + 0.8 * weight))
            hyp.posterior = max(0.01, odds / (1.0 + odds))

    def falsify(self, hypothesis_id: str):
        if hypothesis_id in self.hypotheses:
            self.hypotheses[hypothesis_id].falsified = True
            self.hypotheses[hypothesis_id].posterior = 0.0

    def compute_entropy(self, category: str | None = None) -> float:
        """Computes Shannon entropy over active hypotheses."""
        active = [h for h in self.hypotheses.values() if not h.falsified]
        if category:
            active = [h for h in active if h.category == category]
        if not active:
            return 0.0

        probs = np.array([h.posterior for h in active], dtype=float)
        sum_p = probs.sum()
        if sum_p <= 1e-8:
            return 0.0
        norm_p = probs / sum_p
        ent = -float(np.sum(norm_p * np.log2(norm_p + 1e-12)))
        return max(0.0, ent)

    def get_distribution(self) -> dict[str, float]:
        return {hid: h.posterior for hid, h in self.hypotheses.items() if not h.falsified}


# ---------------------------------------------------------------------------
# D. Goal Memory
# ---------------------------------------------------------------------------

@dataclass
class CandidateGoalRecord:
    coord: tuple[int, int]
    color: int
    entity_id: int | None
    confidence: float = 0.5
    visit_count: int = 0
    falsified: bool = False
    requires_precondition: bool = False
    precondition_key_color: int | None = None


@dataclass
class GoalVariableHypothesis:
    """Hypothesis that manipulating a specific state variable advances toward goal state."""

    variable_name: str
    target_coord: tuple[int, int] | None = None
    observations: int = 0
    progress_correlations: int = 0
    confidence: float = 0.5

    @property
    def posterior(self) -> float:
        """Laplace-smoothed posterior probability P(goal | variable)."""
        if self.observations == 0:
            return 0.5
        return (self.progress_correlations + 1.0) / (self.observations + 2.0)


class GoalMemory:
    """Tracks goal candidates, evidence accumulators, and prerequisite chains."""

    def __init__(self):
        self.candidate_goals: dict[tuple[int, int], CandidateGoalRecord] = {}
        self.active_goal_coord: tuple[int, int] | None = None
        self.variable_hypotheses: dict[str, GoalVariableHypothesis] = {}

    def register_variable(
        self,
        var_name: str,
        target_coord: tuple[int, int] | None = None,
        prior: float = 0.5,
    ) -> GoalVariableHypothesis:
        """Registers a latent goal-variable hypothesis."""
        if var_name not in self.variable_hypotheses:
            self.variable_hypotheses[var_name] = GoalVariableHypothesis(
                variable_name=var_name,
                target_coord=target_coord,
                confidence=prior,
            )
        return self.variable_hypotheses[var_name]

    def record_variable_transition(self, var_name: str, progress_occurred: bool):
        """Updates Bayesian posterior P(goal | variable) upon observing state change."""
        hyp = self.register_variable(var_name)
        hyp.observations += 1
        if progress_occurred:
            hyp.progress_correlations += 1
        hyp.confidence = hyp.posterior

    def get_top_goal_variable(self) -> GoalVariableHypothesis | None:
        """Returns the variable with the highest posterior correlation with goal progress."""
        if not self.variable_hypotheses:
            return None
        return max(self.variable_hypotheses.values(), key=lambda h: h.confidence)

    def add_candidate(self, coord: tuple[int, int], color: int, entity_id: int | None = None):
        if coord not in self.candidate_goals:
            self.candidate_goals[coord] = CandidateGoalRecord(
                coord=coord, color=color, entity_id=entity_id
            )

    def record_visit(self, coord: tuple[int, int], level_advanced: bool) -> bool:
        if coord not in self.candidate_goals:
            return False
        goal = self.candidate_goals[coord]
        goal.visit_count += 1
        if level_advanced:
            goal.confidence = 1.0
            return True
        else:
            # Reached goal without completing level
            goal.confidence = max(0.0, goal.confidence - 0.35)
            if goal.visit_count >= 2:
                goal.falsified = True
                if self.active_goal_coord == coord:
                    self.active_goal_coord = None
            return False

    def mark_precondition(self, coord: tuple[int, int], key_color: int | None = None):
        if coord in self.candidate_goals:
            self.candidate_goals[coord].requires_precondition = True
            self.candidate_goals[coord].precondition_key_color = key_color

    def falsify(self, coord: tuple[int, int]):
        if coord in self.candidate_goals:
            self.candidate_goals[coord].falsified = True
            if self.active_goal_coord == coord:
                self.active_goal_coord = None

    def get_best_candidate(self, current_pos: tuple[int, int] | None) -> tuple[int, int] | None:
        viable = [g for g in self.candidate_goals.values() if not g.falsified]
        if not viable:
            return None

        if current_pos is None:
            return max(viable, key=lambda g: g.confidence).coord

        # Balance confidence with Manhattan distance
        def score(g: CandidateGoalRecord) -> float:
            dist = abs(g.coord[0] - current_pos[0]) + abs(g.coord[1] - current_pos[1])
            return g.confidence - 0.02 * dist

        best = max(viable, key=score)
        self.active_goal_coord = best.coord
        return best.coord


# ---------------------------------------------------------------------------
# E. Skill Memory
# ---------------------------------------------------------------------------

@dataclass
class AbstractSkill:
    skill_id: str
    origin_game: str
    origin_level: int
    rule_type: str  # "kinematic_mapping", "unlock_barrier", "coordinate_targeting"
    schema: dict[str, Any]
    confidence: float = 0.8


class SkillMemory:
    """Stores and retrieves abstract relational invariant rules across levels."""

    def __init__(self):
        self.skills: list[AbstractSkill] = []

    def save_skill(
        self,
        skill_id: str,
        origin_game: str,
        origin_level: int,
        rule_type: str,
        schema: dict[str, Any],
        confidence: float = 0.8,
    ):
        self.skills.append(
            AbstractSkill(
                skill_id=skill_id,
                origin_game=origin_game,
                origin_level=origin_level,
                rule_type=rule_type,
                schema=schema,
                confidence=confidence,
            )
        )

    def get_skills_by_type(self, rule_type: str) -> list[AbstractSkill]:
        return [s for s in self.skills if s.rule_type == rule_type]


# ---------------------------------------------------------------------------
# Unified Structured Belief State
# ---------------------------------------------------------------------------

class StructuredBeliefState:
    """Unified 5-tier memory manager for Baseline 2.0 (FD-NSA)."""

    def __init__(self, game_id: str):
        self.game_id = game_id
        self.perceptual = PerceptualMemory()
        self.causal = CausalMemory()
        self.hypotheses = HypothesisMemory()
        self.goals = GoalMemory()
        self.skills = SkillMemory()
        self.mechanisms = MechanismMemory()
        self.current_level = 1
        self.step_count = 0

    def reset_level(self, new_level: int):
        self.current_level = new_level
        # Reset level-local goal and perceptual memory, but preserve causal & skill invariants
        self.goals = GoalMemory()
        self.perceptual.frame_history.clear()
        self.perceptual.player_pos_history.clear()
        self.mechanisms.reset_level(keep_abstract_signatures=True)

# ======================================================================
# INLINED: falsification_engine.py
# ======================================================================

"""
Causal Prediction-Error World Model & Falsification Engine for ARC-AGI-3.
Implements:
1. 1-Step Forward State Expectation Generator: Predicts avatar position, collisions, and entity deltas.
2. Multi-Aspect Decomposed Prediction Error Vector:
     e_t = (e_pos, e_appear, e_disappear, e_color, e_collision, e_goal)
3. Bayesian Posterior Updating with Conditional Falsification:
     Distinguishes hard invariant contradictions from dormant conditional prerequisites.
4. Historical Replay Verification:
     Backtests candidate transition rules against historical causal transitions.
"""


from dataclasses import dataclass
from typing import Any

import numpy as np



@dataclass
class ForwardPrediction:
    predicted_player_pos: tuple[int, int] | None
    predicted_delta: tuple[int, int]  # (dy, dx)
    expected_collision: bool
    expected_interactive_trigger: bool
    expected_goal_reach: bool


class FalsificationEngine:
    """Computes prediction errors, updates Bayesian posteriors, and verifies hypotheses against history."""

    def __init__(self):
        self.learned_barriers: set[tuple[int, int]] = set()
        self.confirmed_passable: set[tuple[int, int]] = set()
        self.kinematic_mappings: dict[str, tuple[int, int]] = {}
        self.dormant_preconditions: dict[tuple[int, int], str] = {}

    def predict_next_state(
        self,
        action: str,
        current_pos: tuple[int, int] | None,
        hypotheses: HypothesisMemory,
        grid_shape: tuple[int, int] = (16, 16),
        target_pos: tuple[int, int] | None = None,
    ) -> ForwardPrediction:
        """Generates an explicit 1-step prediction before the environment executes the action."""
        if current_pos is None:
            return ForwardPrediction(
                predicted_player_pos=None,
                predicted_delta=(0, 0),
                expected_collision=False,
                expected_interactive_trigger=False,
                expected_goal_reach=False,
            )

        H, W = grid_shape
        # Kinematic prediction from active hypothesis
        dy, dx = (0, 0)
        hyp_key = f"kinematics_{action}"
        if hyp_key in hypotheses.hypotheses and not hypotheses.hypotheses[hyp_key].falsified:
            dy = hypotheses.hypotheses[hyp_key].details.get("dy", 0)
            dx = hypotheses.hypotheses[hyp_key].details.get("dx", 0)
        elif action in self.kinematic_mappings:
            dy, dx = self.kinematic_mappings[action]
        else:
            # Cardinal default prior
            cardinals = {
                "ACTION1": (-1, 0),
                "ACTION2": (1, 0),
                "ACTION3": (0, -1),
                "ACTION4": (0, 1),
            }
            dy, dx = cardinals.get(action, (0, 0))

        ny = current_pos[0] + dy
        nx = current_pos[1] + dx

        # Check bounds and learned barrier collisions
        collision = False
        if ny < 0 or ny >= H or nx < 0 or nx >= W or (ny, nx) in self.learned_barriers:
            collision = True
            predicted_pos = current_pos
            pred_delta = (0, 0)
        else:
            predicted_pos = (ny, nx)
            pred_delta = (dy, dx)

        reaches_target = (target_pos is not None and predicted_pos == target_pos)

        return ForwardPrediction(
            predicted_player_pos=predicted_pos,
            predicted_delta=pred_delta,
            expected_collision=collision,
            expected_interactive_trigger=(action == "ACTION6"),
            expected_goal_reach=reaches_target,
        )

    def evaluate_prediction_error(
        self,
        prediction: ForwardPrediction,
        actual_pos: tuple[int, int] | None,
        prev_pos: tuple[int, int] | None,
        level_advanced: bool = False,
        grid_diff_count: int = 0,
    ) -> PredictionError:
        """Calculates multi-aspect decomposed prediction error vector."""
        # 1. Position error (Manhattan distance)
        pos_err = 0.0
        if prediction.predicted_player_pos and actual_pos:
            pos_err = float(
                abs(prediction.predicted_player_pos[0] - actual_pos[0])
                + abs(prediction.predicted_player_pos[1] - actual_pos[1])
            )
        elif prediction.predicted_player_pos != actual_pos:
            pos_err = 1.0

        # 2. Collision error
        coll_err = 0.0
        actual_moved = (prev_pos is not None and actual_pos is not None and actual_pos != prev_pos)
        if prediction.expected_collision and actual_moved:
            coll_err = 1.0  # Hallucinated barrier
        elif not prediction.expected_collision and prev_pos == actual_pos and prediction.predicted_delta != (0, 0):
            coll_err = 1.0  # Unexpected barrier

        # 3. Goal progress error
        goal_err = 0.0
        if prediction.expected_goal_reach and not level_advanced:
            goal_err = 1.0  # Reached expected goal but level didn't advance
        elif not prediction.expected_goal_reach and level_advanced:
            goal_err = 0.5  # Serendipitous level completion

        # 4. Appearance / Disappearance errors (from pixel diffs)
        appear_err = 0.1 * min(10, grid_diff_count) if grid_diff_count > 2 else 0.0
        disappear_err = 0.0

        return PredictionError(
            position_error=pos_err,
            appearance_error=appear_err,
            disappearance_error=disappear_err,
            color_error=0.0,
            collision_error=coll_err,
            goal_progress_error=goal_err,
        )

    def update_and_falsify(
        self,
        action: str,
        prev_pos: tuple[int, int] | None,
        actual_pos: tuple[int, int] | None,
        prediction: ForwardPrediction,
        error: PredictionError,
        hypotheses: HypothesisMemory,
    ):
        """Bayesian update and conditional falsification based on prediction error."""
        if prev_pos is None or actual_pos is None:
            return

        dy = actual_pos[0] - prev_pos[0]
        dx = actual_pos[1] - prev_pos[1]

        # 1. Kinematic verification / updating
        hyp_key = f"kinematics_{action}"
        if (dy, dx) != (0, 0):
            norm_dy = 1 if dy > 0 else (-1 if dy < 0 else 0)
            norm_dx = 1 if dx > 0 else (-1 if dx < 0 else 0)
            self.kinematic_mappings[action] = (norm_dy, norm_dx)
            self.confirmed_passable.add(actual_pos)

            if hyp_key in hypotheses.hypotheses:
                expected_delta = hypotheses.hypotheses[hyp_key].details.get("delta")
                if expected_delta == (norm_dy, norm_dx):
                    hypotheses.update_evidence(hyp_key, supported=True, weight=1.5)
                else:
                    # Invariant contradiction
                    hypotheses.falsify(hyp_key)
            else:
                hypotheses.register(
                    hypothesis_id=hyp_key,
                    category="kinematics",
                    description=f"Action {action} produces translation ({norm_dy}, {norm_dx})",
                    prior=0.75,
                    details={"dy": norm_dy, "dx": norm_dx, "delta": (norm_dy, norm_dx)},
                )
        else:
            # Stationary outcome: Did we attempt a move into a barrier?
            if prediction.predicted_delta != (0, 0):
                target_cell = (prev_pos[0] + prediction.predicted_delta[0], prev_pos[1] + prediction.predicted_delta[1])
                self.learned_barriers.add(target_cell)

    def verify_against_history(
        self,
        candidate_rule: tuple[int, int],  # (dy, dx)
        action: str,
        causal_memory: CausalMemory,
    ) -> bool:
        """Historical replay verification: checks if candidate rule contradicts any past observations."""
        relevant = [t for t in causal_memory.transitions if t.action == action and t.prev_player_pos is not None]
        if not relevant:
            return True

        contradictions = 0
        for t in relevant:
            actual_delta = t.avatar_delta
            if actual_delta != (0, 0) and actual_delta != candidate_rule:
                contradictions += 1

        return contradictions == 0

# ======================================================================
# INLINED: deadlock_taxonomy.py
# ======================================================================

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

# ======================================================================
# INLINED: level_transfer.py
# ======================================================================

"""
Cross-Level Skill Transfer & Invariant Schema Extraction for ARC-AGI-3.
Prevents negative transfer by separating relational invariant schemas from concrete tokens:
  1. Preserves verified cardinal kinematic mappings across levels (e.g. ACTION1 = UP).
  2. Preserves avatar morphology priors (e.g. singleton vs multi-pixel entity).
  3. Resets and dynamically re-binds concrete color tokens and coordinates per level.
"""


from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from typing import Any



@dataclass
class LevelTransferPackage:
    verified_kinematics: dict[str, tuple[int, int]]  # Action -> (dy, dx)
    avatar_is_singleton: bool
    successful_action_modalities: set[str]  # e.g. {"ACTION1", "ACTION2", "ACTION6"}
    total_levels_completed: int
    transferred_inert_signatures: set[EntitySignature] = field(default_factory=set)
    transferred_active_signatures: set[EntitySignature] = field(default_factory=set)


class LevelTransferManager:
    """Manages cross-level knowledge retention while safeguarding against color/token inversion."""

    def __init__(self):
        self.verified_kinematics: dict[str, tuple[int, int]] = {}
        self.avatar_is_singleton: bool = True
        self.confirmed_modalities: set[str] = set()
        self.completed_level_count: int = 0
        self.inert_signatures: set[EntitySignature] = set()
        self.active_signatures: set[EntitySignature] = set()

    def extract_level_schema(
        self,
        level_index: int,
        kinematic_mappings: dict[str, tuple[int, int]],
        avatar_size: int,
        actions_used: set[str],
        skill_memory: SkillMemory,
        game_id: str,
        mechanism_memory: Any | None = None,
    ) -> LevelTransferPackage:
        """Called immediately upon completing a level to compile reusable invariant schemas."""
        self.completed_level_count += 1
        self.verified_kinematics.update(kinematic_mappings)
        self.avatar_is_singleton = (avatar_size <= 1)
        self.confirmed_modalities.update(actions_used)

        if mechanism_memory is not None and hasattr(mechanism_memory, "get_inert_signatures"):
            self.inert_signatures.update(mechanism_memory.get_inert_signatures())
            self.active_signatures.update(getattr(mechanism_memory, "active_signatures", set()))

        pkg = LevelTransferPackage(
            verified_kinematics=dict(self.verified_kinematics),
            avatar_is_singleton=self.avatar_is_singleton,
            successful_action_modalities=set(self.confirmed_modalities),
            total_levels_completed=self.completed_level_count,
            transferred_inert_signatures=set(self.inert_signatures),
            transferred_active_signatures=set(self.active_signatures),
        )

        skill_memory.save_skill(
            skill_id=f"level_{level_index}_kinematic_transfer",
            origin_game=game_id,
            origin_level=level_index,
            rule_type="kinematic_mapping",
            schema={"kinematics": dict(self.verified_kinematics)},
            confidence=0.95,
        )

        return pkg

    def apply_transfer_priors(
        self,
        available_actions: list[str],
    ) -> dict[str, tuple[int, int]]:
        """Provides verified kinematic priors to the new level without carrying over concrete colors."""
        priors: dict[str, tuple[int, int]] = {}
        for act in available_actions:
            if act in self.verified_kinematics:
                priors[act] = self.verified_kinematics[act]
        return priors

    def apply_mechanism_priors(self, mechanism_memory: Any):
        """Populates new level's mechanism memory with verified abstract entity priors."""
        if hasattr(mechanism_memory, "inert_signatures"):
            mechanism_memory.inert_signatures.update(self.inert_signatures)
        if hasattr(mechanism_memory, "active_signatures"):
            mechanism_memory.active_signatures.update(self.active_signatures)

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
# INLINED: action_algebra.py
# ======================================================================

"""
Action Algebra Engine for ARC-AGI-3 (Baseline 3.0 B3.03).
Infers algebraic relationships between actions from observed transitions:
  - Inverse operations: f(f(s, A), B) == s  ==>  B = A^(-1)
  - Self-inverse toggles: f(f(s, A), A) == s
  - Additive compositions: f(s, A) yields monotonic state variable progress
Prevents detrimental oscillation (e.g., alternating +1 / -1 buttons) and enables
shortest-sequence planning under known group/monoid dynamics.
"""


from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence



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

# ======================================================================
# INLINED: model_gate.py
# ======================================================================

"""
Conditional Model-Use Gate for ARC-AGI-3 (Baseline 3.0 B3.07).
Decides whether to act directly, simulate/plan through the world model, or perform
a discriminating epistemic probe. Avoids spending unnecessary computation or actions
constructing models when direct execution is already validated.
"""


from enum import Enum


class ModelUseDecision(str, Enum):
    """Decision modes for action generation."""

    ACT_DIRECTLY = "ACT_DIRECTLY"       # High confidence or active plan in flight; execute immediately
    SIMULATE_MODEL = "SIMULATE_MODEL"   # Verified transition dynamics available; plan/simulate trajectory
    PERFORM_PROBE = "PERFORM_PROBE"     # Epistemic uncertainty high; execute discriminating experiment


class ModelGate:
    """Arbitrates among direct execution, world-model planning, and epistemic probing."""

    def __init__(self, confidence_threshold: float = 0.75):
        self.confidence_threshold = confidence_threshold

    def evaluate_decision(
        self,
        has_active_plan: bool,
        can_reliably_plan: bool,
        has_exploitable_controller: bool,
        is_loop: bool = False,
    ) -> ModelUseDecision:
        """
        Determines the optimal computational and action strategy for the current turn.

        Args:
            has_active_plan: True if a verified multi-step macro-plan is already in flight.
            can_reliably_plan: True if world model kinematics and barriers are verified.
            has_exploitable_controller: True if a high-confidence goal-advancing controller is identified.
            is_loop: True if stagnation/deadlock loop was detected.

        Returns:
            ModelUseDecision member.
        """
        # Deadlock / loop breaks force epistemic probing
        if is_loop:
            return ModelUseDecision.PERFORM_PROBE

        # 1. Direct Execution: macro-plan in flight or verified exploitable controller
        if has_active_plan or has_exploitable_controller:
            return ModelUseDecision.ACT_DIRECTLY

        # 2. World Model Simulation: kinematics verified, plan shortest path to goal
        if can_reliably_plan:
            return ModelUseDecision.SIMULATE_MODEL

        # 3. Epistemic Probing: explore untested dynamics or mechanisms
        return ModelUseDecision.PERFORM_PROBE

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
        self.algebra = ActionAlgebraEngine()
        self.model_gate = ModelGate()
        self.recent_coords: deque[tuple[int, int]] = deque(maxlen=6)

    def select_action(
        self,
        observation: Observation,
        analysis: FrameAnalysis,
        world_model: BeliefStateWorldModel,
        reasoning_state: PersistentReasoningState | None = None,
        cognitive_analysis: CognitiveHierarchyAnalysis | None = None,
        structured_belief: Any | None = None,
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

        # Model-Use Gate: Determine execution strategy (direct vs simulation vs probe)
        has_plan = bool(reasoning_state and reasoning_state.has_active_plan())
        can_plan = world_model.can_reliably_plan()
        has_exploit = False
        if structured_belief is not None and hasattr(structured_belief, "mechanisms"):
            has_exploit = len(structured_belief.mechanisms.get_exploitable_controllers()) > 0

        model_decision = self.model_gate.evaluate_decision(
            has_active_plan=has_plan,
            can_reliably_plan=can_plan,
            has_exploitable_controller=has_exploit,
            is_loop=is_loop,
        )

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
            observation, analysis, world_model, reasoning_state, is_loop, structured_belief
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
        structured_belief: Any | None = None,
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
            # Update action algebra from mechanism transition ledger if available
            if structured_belief is not None and hasattr(structured_belief, "mechanisms"):
                self.algebra.update_from_records(structured_belief.mechanisms.records)

            # Affordance-driven coordinate selection
            last_coord = getattr(reasoning_state, "last_affordance_coord", None) if reasoning_state else None
            last_diff = getattr(reasoning_state, "last_affordance_diff", 0) if reasoning_state else 0
            repeat_count = getattr(reasoning_state, "affordance_repeat_count", 0) if reasoning_state else 0
            lethal_set = getattr(reasoning_state, "lethal_affordances", set()) if reasoning_state else set()
            inert_set = getattr(reasoning_state, "inert_affordances", set()) if reasoning_state else set()
            active_list = getattr(reasoning_state, "active_affordances", []) if reasoning_state else []

            # 1. Filter entities: compact play entities (avoiding huge obstacle blobs and letterbox/UI bounds)
            cands = []
            if analysis.entities:
                for e in analysis.entities:
                    cy, cx = int(round(e.centroid[0])), int(round(e.centroid[1]))
                    # Discard letterbox border pixels and top UI indicator rows (cy <= 1)
                    if not (2 <= cy <= 61 and 2 <= cx <= 61):
                        continue
                    if e.size > 120:  # Massive entities are obstacles or static backgrounds
                        continue
                    if (cy, cx) in lethal_set:
                        continue
                    cands.append(e)

            chosen_coord: tuple[int, int] | None = None

            # 2. Decision logic:
            # Check mechanism confidence vs goal relevance before repeating active controller
            mech_permits_exploit = True
            if structured_belief is not None and hasattr(structured_belief, "mechanisms") and last_coord:
                hyp = structured_belief.mechanisms.get_hypothesis(last_coord)
                if hyp is not None and not hyp.can_reliably_exploit():
                    mech_permits_exploit = False

            if last_coord is not None and last_diff > 2 and repeat_count < 5 and last_coord not in lethal_set and mech_permits_exploit:
                chosen_coord = last_coord
            elif cands:
                # Prioritize candidates not confirmed inert by coordinate
                non_inert = [
                    e
                    for e in cands
                    if (int(round(e.centroid[0])), int(round(e.centroid[1]))) not in inert_set
                ]
                # B3.04: Causal equivalence class filtering (prune entire classes of inert entities)
                if structured_belief is not None and hasattr(structured_belief, "mechanisms"):
                    non_inert = structured_belief.mechanisms.filter_inert_classes(non_inert)

                pool = non_inert if non_inert else cands
                # Filter out oscillating inverse actions
                coords_pool = [(int(round(e.centroid[0])), int(round(e.centroid[1]))) for e in pool]
                pruned_coords = self.algebra.filter_oscillating_actions(coords_pool, list(self.recent_coords))
                cands_to_score = [
                    e
                    for e in pool
                    if (int(round(e.centroid[0])), int(round(e.centroid[1]))) in set(pruned_coords)
                ] or pool

                # B3.11: RHAE-aware utility ranking U(a) = αG + βI + γC - λK - μR
                cand_scores = [
                    (
                        e,
                        self._compute_action_utility(
                            coord=(int(round(e.centroid[0])), int(round(e.centroid[1]))),
                            entity=e,
                            structured_belief=structured_belief,
                            reasoning_state=reasoning_state,
                        ),
                    )
                    for e in cands_to_score
                ]
                max_u = max(s[1] for s in cand_scores)
                top_cands = [s[0] for s in cand_scores if abs(s[1] - max_u) < 0.05]
                best_ent = top_cands[self.step_counter % len(top_cands)]
                chosen_coord = (int(round(best_ent.centroid[0])), int(round(best_ent.centroid[1])))
            elif active_list:
                pruned_active = self.algebra.filter_oscillating_actions(active_list, list(self.recent_coords))
                chosen_coord = (pruned_active or active_list)[self.step_counter % len(pruned_active or active_list)]
            else:
                H, W = analysis.frame_shape
                chosen_coord = (int(np.clip(H // 2, 0, 63)), int(np.clip(W // 2, 0, 63)))

            if chosen_coord is not None:
                self.recent_coords.append(chosen_coord)

            payload = {
                "x": int(np.clip(chosen_coord[1], 0, 63)),
                "y": int(np.clip(chosen_coord[0], 0, 63)),
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

    def _compute_action_utility(
        self,
        coord: tuple[int, int],
        entity: Any,
        structured_belief: Any | None,
        reasoning_state: PersistentReasoningState | None,
    ) -> float:
        """
        Computes action utility: U(a) = α*G(a) + β*I(a) + γ*C(a) - λ*K(a) - μ*R(a)
        Maximizes goal progress and information gain while penalizing action cost and risk.
        """
        G = 0.5
        C = 0.5
        I = 1.0
        R = 0.0
        K = 1.0

        if structured_belief is not None and hasattr(structured_belief, "mechanisms"):
            hyp = structured_belief.mechanisms.get_hypothesis(coord)
            if hyp is not None:
                G = hyp.goal_relevance
                C = hyp.causal_confidence
                I = 1.0 / (hyp.observations + 1.0)
                if hyp.is_lethal:
                    R = 1.0
            else:
                I = 1.0

        if reasoning_state is not None and coord in reasoning_state.death_coords:
            R = 1.0

        # Weights: α=3.5 (goal progress), β=1.0 (information gain), γ=1.0 (causal confidence), λ=0.5 (action cost), μ=5.0 (risk)
        return 3.5 * G + 1.0 * I + 1.0 * C - 0.5 * K - 5.0 * R

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
    Production-ready Uncertainty-Aware Agent for ARC-AGI-3 (Baseline 2.0: FD-NSA).
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
        self.game_id = game_id
        self.parameters = parameters or {}
        self.perception = LayeredPerception()
        self.cognitive_perception = CognitiveHierarchyPerception()
        self.world_model = BeliefStateWorldModel()
        self.policy = EpistemicPolicy()
        self.memory = ScopedEpisodeMemory(game_key=self.game_id)
        self.reasoning_state = PersistentReasoningState(game_id=self.game_id)
        self.compactor = ContextCompactor()

        # Baseline 2.0: 5-Tier Memory, Falsification Engine, Deadlock Taxonomy, Transfer Manager
        self.structured_belief = StructuredBeliefState(game_id=self.game_id)
        self.falsification_engine = FalsificationEngine()
        self.deadlock_engine = DeadlockTaxonomyEngine()
        self.level_transfer = LevelTransferManager()
        self.replay_logger = ReplayLogger(game_id=self.game_id)
        self.previous_prediction: ForwardPrediction | None = None

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
        if current_level != self.structured_belief.current_level:
            # 0. Cross-level invariant extraction upon level progression
            self.level_transfer.extract_level_schema(
                level_index=self.structured_belief.current_level,
                kinematic_mappings=self.falsification_engine.kinematic_mappings,
                avatar_size=1,
                actions_used=set(self.falsification_engine.kinematic_mappings.keys()),
                skill_memory=self.structured_belief.skills,
                game_id=self.game_id,
                mechanism_memory=self.structured_belief.mechanisms,
            )
            self.structured_belief.reset_level(current_level)
            self.deadlock_engine.reset_level()
            self.reasoning_state.reset_level(current_level)
            # Apply transferred mechanism priors to the new level
            self.level_transfer.apply_mechanism_priors(self.structured_belief.mechanisms)

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
            level=current_level,
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

        # 2. Prediction-Error Evaluation & Bayesian Belief Revision
        curr_pos = cognitive_analysis.player_pos if cognitive_analysis else None
        prev_pos = self.previous_cognitive.player_pos if self.previous_cognitive else None

        if self.previous_action is not None and self.previous_prediction is not None:
            diff_count = (
                int(np.sum(current_analysis.dynamic_diff_mask))
                if current_analysis.dynamic_diff_mask is not None
                else 0
            )

            # Compute decomposed prediction error vector
            pred_error = self.falsification_engine.evaluate_prediction_error(
                prediction=self.previous_prediction,
                actual_pos=curr_pos,
                prev_pos=prev_pos,
                level_advanced=(current_level > self.structured_belief.current_level),
                grid_diff_count=diff_count,
            )

            # Bayesian update and conditional falsification
            self.falsification_engine.update_and_falsify(
                action=self.previous_action,
                prev_pos=prev_pos,
                actual_pos=curr_pos,
                prediction=self.previous_prediction,
                error=pred_error,
                hypotheses=self.structured_belief.hypotheses,
            )

            # Record transition in causal memory
            self.structured_belief.causal.record(
                step=self.action_count - 1,
                action=self.previous_action,
                payload=getattr(self, "last_payload", {}),
                prev_pos=prev_pos,
                curr_pos=curr_pos,
                diff_pixel_count=diff_count,
            )

            # Deadlock classification & recovery check
            tgt_coord = None
            if hasattr(self, "last_payload") and self.last_payload:
                tgt_coord = (self.last_payload.get("y"), self.last_payload.get("x"))

            deadlock = self.deadlock_engine.record_step(
                current_pos=curr_pos,
                action=self.previous_action,
                state=state_str,
                active_goal=self.structured_belief.goals.active_goal_coord,
                goal_reached_without_win=(
                    curr_pos == self.structured_belief.goals.active_goal_coord
                    and state_str not in ("WIN", "GameState.WIN")
                ),
                diff_count=diff_count,
                target_coord=tgt_coord,
            )
            if deadlock and deadlock.deadlock_type == DeadlockType.TYPE_2_WRONG_GOAL:
                if curr_pos:
                    self.structured_belief.goals.falsify(curr_pos)
                    self.reasoning_state.falsify_goal(curr_pos)

            # Coordinate affordance & mechanism ledger registration for ACTION6
            if self.previous_action == "ACTION6" and hasattr(self, "last_payload") and self.last_payload:
                px = self.last_payload.get("x")
                py = self.last_payload.get("y")
                if px is not None and py is not None:
                    is_lethal = state_str in ("GAME_OVER", "GameState.GAME_OVER") or (
                        deadlock is not None and deadlock.deadlock_type == DeadlockType.TYPE_5_LETHAL_TRAP
                    )
                    level_adv = (current_level > self.structured_belief.current_level)
                    self.reasoning_state.register_affordance_result(
                        coord=(py, px),
                        diff_count=diff_count,
                        is_lethal=bool(is_lethal),
                        state_str=state_str,
                        level_advanced=level_adv,
                    )

                    # B3.02: Resolve target entity and record in Causal Mechanism Ledger
                    target_sig = None
                    cand_pool = self.previous_analysis.entities if self.previous_analysis and self.previous_analysis.entities else current_analysis.entities
                    if cand_pool:
                        for ent in cand_pool:
                            if (py, px) in getattr(ent, "cells", ()):
                                target_sig = EntitySignature.from_entity(ent)
                                break
                        if target_sig is None:
                            best_d = float("inf")
                            best_ent = None
                            for ent in cand_pool:
                                d = abs(ent.centroid[0] - py) + abs(ent.centroid[1] - px)
                                if d < best_d and d <= 4.0:
                                    best_d = d
                                    best_ent = ent
                            if best_ent is not None:
                                target_sig = EntitySignature.from_entity(best_ent)

                    # Identify changed entity signatures
                    changed_sigs = []
                    if current_analysis.entities:
                        for ent in current_analysis.entities:
                            if getattr(ent, "is_dynamic", False):
                                changed_sigs.append(EntitySignature.from_entity(ent))

                    eff_type = self.reasoning_state.last_effect_type or classify_effect(
                        diff_count=diff_count,
                        state=state_str,
                        level_advanced=level_adv,
                        is_lethal=bool(is_lethal),
                    )
                    pre_hash = str(hash(prev_grid.tobytes())) if prev_grid is not None else ""
                    post_hash = str(hash(grid.tobytes()))

                    self.structured_belief.mechanisms.record_transition(
                        step=self.action_count - 1,
                        action=self.previous_action,
                        target_coord=(py, px),
                        target_entity=target_sig,
                        effect_type=eff_type,
                        diff_count=diff_count,
                        changed_entity_signatures=changed_sigs,
                        level_advanced=level_adv,
                        is_lethal=bool(is_lethal),
                        pre_frame_hash=pre_hash,
                        post_frame_hash=post_hash,
                    )

                    # B3.05: Update Goal-Variable Hypotheses
                    if eff_type in (
                        EffectType.LOCAL_MUTATION,
                        EffectType.STRUCTURAL_MUTATION,
                        EffectType.GLOBAL_MUTATION,
                        EffectType.LEVEL_ADVANCE,
                    ):
                        var_key = f"var_{target_sig.size_bucket if target_sig else 'unknown'}_{eff_type.value}"
                        self.structured_belief.goals.record_variable_transition(
                            var_name=var_key,
                            progress_occurred=bool(level_adv or eff_type == EffectType.STRUCTURAL_MUTATION),
                        )

            # Log step telemetry to ReplayLogger
            self.replay_logger.log_step(
                step=self.action_count - 1,
                level=self.structured_belief.current_level,
                observation_hash=str(hash(grid.tobytes())),
                action=self.previous_action,
                payload=getattr(self, "last_payload", {}),
                predicted_avatar_pos=self.previous_prediction.predicted_player_pos,
                actual_avatar_pos=curr_pos,
                prediction_error=pred_error,
                active_hypotheses=self.structured_belief.hypotheses.get_distribution(),
                active_goal=self.structured_belief.goals.active_goal_coord,
                deadlock_type=deadlock.deadlock_type.value if deadlock else None,
                state=state_str,
                levels_completed=getattr(latest_frame, "levels_completed", 0),
            )

        # Update perceptual and goal memories
        bg_col = (
            current_analysis.background_hypotheses[0][0]
            if current_analysis.background_hypotheses
            else 0
        )
        self.structured_belief.perceptual.update(
            grid=grid,
            entities=current_analysis.entities,
            player_pos=cognitive_analysis.player_pos,
            player_color=cognitive_analysis.player_color,
            bg_color=bg_col,
            step=self.action_count,
        )
        for goal_coord in cognitive_analysis.candidate_goals:
            self.structured_belief.goals.add_candidate(goal_coord, color=0)

        # Update Reasoning Persistence displacements
        if self.previous_action is not None and prev_pos is not None and curr_pos is not None:
            dy = curr_pos[0] - prev_pos[0]
            dx = curr_pos[1] - prev_pos[1]
            self.reasoning_state.update_action_effect(self.previous_action, dy, dx)

        # Context compaction
        self.compactor.compact_step(
            step=self.action_count,
            level=current_obs.level,
            action=self.previous_action or "NONE",
            curr_pos=curr_pos,
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
            structured_belief=self.structured_belief,
        )

        # 4. Generate 1-Step Forward Prediction for chosen action
        self.previous_prediction = self.falsification_engine.predict_next_state(
            action=action_name,
            current_pos=curr_pos,
            hypotheses=self.structured_belief.hypotheses,
            grid_shape=grid.shape,
            target_pos=self.structured_belief.goals.active_goal_coord,
        )

        # Update tracking
        self.previous_observation = current_obs
        self.previous_analysis = current_analysis
        self.previous_cognitive = cognitive_analysis
        self.previous_action = action_name
        self.last_payload = payload
        return LegalityAdapter.to_game_action(action_name, payload)