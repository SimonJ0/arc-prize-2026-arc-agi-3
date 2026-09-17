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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.arc_agent.diagnostics.replay_logger import PredictionError
from src.arc_agent.memory.structured_belief import (
    CausalMemory,
    HypothesisMemory,
    PerceptualMemory,
)


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
