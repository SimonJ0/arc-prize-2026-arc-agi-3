"""
ARC-AGI-3 Step-Level Replay Diagnostic Logger.
Instruments agent execution to capture:
1. Per-step observations, predictions, and decomposed prediction errors.
2. Active hypothesis distributions P(H_i).
3. Action rationale and risk tiers (Levels 0-5).
4. Deadlock classifications (Types 1-6).
5. Automated post-hoc failure attribution (8-tier taxonomy).
"""

from __future__ import annotations

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
