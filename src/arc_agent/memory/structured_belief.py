"""
5-Tier Structured Memory & Belief State Architecture for ARC-AGI-3.
Replaces unstructured scratchpad with 5 dedicated cognitive memory modules:
  A. Perceptual Memory (Entities, positions, morphology, spatial diffs)
  B. Causal Memory (Action-delta transitions, causal intervention attribution)
  C. Hypothesis Memory (Bayesian posteriors over kinematics, mechanics, goals)
  D. Goal Memory (Candidate goal coordinates, evidence, precondition chains)
  E. Skill Memory (Cross-level invariant relational rules)
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.arc_agent.memory.mechanism_memory import MechanismMemory


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
