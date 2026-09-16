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
