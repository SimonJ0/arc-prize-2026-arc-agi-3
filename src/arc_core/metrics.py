"""
Mathematical implementation of Relative Human Action Efficiency (RHAE)
and local proxy metrics for ARC-AGI-3.

Matches the official ARC-AGI-3 scoring methodology:
  level_score = min(1.15, (h_{e,l} / a_{e,l})^2)
  R_e = sum(l * level_score) / sum(l)
  C_e = sum(l * 1[completed]) / sum(l)
  E_e = min(R_e, C_e) * 100
  Total = mean_{e in D}(E_e)
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LevelMetric:
    level_index: int
    completed: bool
    actions_taken: int
    baseline_actions: int | None = None

    @property
    def level_score(self) -> float:
        """Level score in [0.0, 1.15] as a percentage fraction (0.0 to 1.15)."""
        if not self.completed or self.actions_taken <= 0:
            return 0.0
        if self.baseline_actions is None or self.baseline_actions <= 0:
            return 1.0  # Fallback completion indicator
        ratio = self.baseline_actions / self.actions_taken
        return float(min(1.15, ratio**2))


@dataclass
class EnvironmentEvaluation:
    game_id: str
    levels: list[LevelMetric] = field(default_factory=list)
    resets: int = 0

    @property
    def levels_completed(self) -> int:
        return sum(1 for lvl in self.levels if lvl.completed)

    @property
    def total_actions(self) -> int:
        return sum(lvl.actions_taken for lvl in self.levels)

    def compute_rhae(self) -> dict[str, float]:
        """
        Computes exact official RHAE environment score.
        Returns:
            dict with 'score', 'raw_score', 'completion_cap', 'levels_completed'
        """
        if not self.levels:
            return {"score": 0.0, "raw_score": 0.0, "completion_cap": 0.0, "levels_completed": 0}

        total_weights = 0
        total_weighted_score = 0.0
        completed_weights = 0

        for lvl in self.levels:
            w = lvl.level_index
            total_weights += w
            if lvl.completed:
                completed_weights += w
                total_weighted_score += (lvl.level_score * 100.0) * w

        if total_weights == 0:
            return {"score": 0.0, "raw_score": 0.0, "completion_cap": 0.0, "levels_completed": 0}

        raw_score = total_weighted_score / total_weights
        completion_cap = (completed_weights / total_weights) * 100.0
        final_score = min(raw_score, completion_cap)

        return {
            "score": round(final_score, 4),
            "raw_score": round(raw_score, 4),
            "completion_cap": round(completion_cap, 4),
            "levels_completed": self.levels_completed,
            "total_actions": self.total_actions,
            "resets": self.resets,
        }

    def compute_local_proxies(self) -> dict[str, float]:
        """
        Computes proxy performance metrics when human baseline actions are unknown.
        Prioritizes level depth, completion rate, action efficiency, and low resets.
        """
        num_levels = len(self.levels)
        if num_levels == 0:
            return {"completion_rate": 0.0, "depth": 0, "avg_actions_per_level": 0.0}

        completed_indices = [lvl.level_index for lvl in self.levels if lvl.completed]
        max_depth = max(completed_indices) if completed_indices else 0
        total_weights = sum(lvl.level_index for lvl in self.levels)
        completed_weights = sum(lvl.level_index for lvl in self.levels if lvl.completed)
        weighted_completion = (completed_weights / total_weights) if total_weights > 0 else 0.0

        return {
            "completion_rate": round(self.levels_completed / num_levels, 4),
            "weighted_completion": round(weighted_completion, 4),
            "max_depth": max_depth,
            "total_actions": self.total_actions,
            "resets": self.resets,
        }


def compute_benchmark_rhae(evaluations: list[EnvironmentEvaluation]) -> float:
    """Computes aggregate benchmark RHAE: T = (1 / |D|) * sum(E_e)."""
    if not evaluations:
        return 0.0
    scores = [e.compute_rhae()["score"] for e in evaluations]
    return float(np.mean(scores))
