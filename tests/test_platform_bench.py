"""
Unit tests for the official ARC-AGI PlatformBenchmarkSuite.
"""

from pathlib import Path
import pytest
from src.arc_core.platform_bench import (
    PlatformBenchmarkSuite,
    TRAIN_GAMES,
    HOLDOUT_GAMES,
)


def test_split_disjointness():
    """Verify train and holdout splits are completely disjoint."""
    train_set = set(TRAIN_GAMES)
    holdout_set = set(HOLDOUT_GAMES)
    assert len(train_set) == 15
    assert len(holdout_set) == 10
    assert train_set.isdisjoint(holdout_set), "Train and holdout sets must be strictly disjoint!"


def test_platform_bench_initialization():
    """Verify platform benchmark suite connects to Arcade and maps environments."""
    suite = PlatformBenchmarkSuite()
    games = suite.list_available_games()
    assert len(games) >= 20, "Should have access to official platform games."
    first = games[0]
    assert "game_id" in first
    assert "baseline_actions" in first
    assert isinstance(first["baseline_actions"], list)


def test_platform_bench_single_game_evaluation():
    """Verify evaluating a real game runs and returns valid diagnostics."""
    suite = PlatformBenchmarkSuite()
    # Evaluate 1 training game with a tiny action limit (5 steps)
    test_game = TRAIN_GAMES[0]  # ls20-9607627b
    env_eval, diag = suite.evaluate_game(
        game_id=test_game,
        max_actions=5,
        seed=42,
    )

    assert diag["game_id"] == test_game
    assert "score" in diag
    assert "legality_rate" in diag
    assert diag["legality_rate"] == 1.0, "All actions must be legal!"
    assert diag["total_actions"] <= 5
    assert len(diag["level_details"]) > 0


def test_platform_suite_report_generation(tmp_path):
    """Verify suite evaluation report writing."""
    suite = PlatformBenchmarkSuite(reports_dir=str(tmp_path))
    test_game = TRAIN_GAMES[0]
    summary = suite.evaluate_suite(
        split="custom",
        games_subset=[test_game],
        max_actions=4,
        seed=42,
    )

    assert "mean_rhae_score" in summary
    assert "report_path" in summary
    report_file = Path(summary["report_path"])
    assert report_file.exists()


def test_platform_multi_budget_evaluation(tmp_path):
    """Verify multi-budget evaluation method produces budget curves and valid report."""
    suite = PlatformBenchmarkSuite(reports_dir=str(tmp_path))
    test_game = TRAIN_GAMES[0]
    multi_summary = suite.evaluate_multi_budget(
        split="custom",
        games_subset=[test_game],
        budgets=[3, 6],
        seed=42,
    )

    assert "budget_curve" in multi_summary
    assert 3 in multi_summary["budget_curve"]
    assert 6 in multi_summary["budget_curve"]
    assert "report_path" in multi_summary
    report_file = Path(multi_summary["report_path"])
    assert report_file.exists()

