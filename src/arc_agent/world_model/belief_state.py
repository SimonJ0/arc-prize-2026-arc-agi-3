"""
Factored World Model Belief State for ARC-AGI-3.
Maintains a Bayesian distribution over transition dynamics hypotheses.
Tracks 1-step prediction accuracy and explicitly falsifies hypotheses upon contradiction.
Only authorizes deep planning when model consensus and fidelity meet confidence thresholds.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from src.arc_core.contracts import (
    Observation,
    Transition,
    TransitionModelHypothesis,
    WorldModelBelief,
)
from src.arc_agent.perception.layered_perception import FrameAnalysis


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
