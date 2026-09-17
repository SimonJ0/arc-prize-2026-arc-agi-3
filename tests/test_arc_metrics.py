"""
Unit tests for ARC-AGI-3 RHAE metrics and local proxy evaluators.
"""

from src.arc_core.metrics import EnvironmentEvaluation, LevelMetric


def test_level_metric_score_and_cap():
    # Exactly matching human baseline -> 1.0 (100%)
    lvl_exact = LevelMetric(level_index=1, completed=True, actions_taken=10, baseline_actions=10)
    assert lvl_exact.level_score == 1.0

    # Half the human actions -> capped at 1.15
    lvl_super = LevelMetric(level_index=1, completed=True, actions_taken=5, baseline_actions=10)
    assert lvl_super.level_score == 1.15

    # Double human actions -> 0.25
    lvl_slow = LevelMetric(level_index=1, completed=True, actions_taken=20, baseline_actions=10)
    assert lvl_slow.level_score == 0.25

    # Incomplete level -> 0.0
    lvl_fail = LevelMetric(level_index=1, completed=False, actions_taken=10, baseline_actions=10)
    assert lvl_fail.level_score == 0.0


def test_environment_completion_cap_enforcement():
    # If an agent completes only Level 1 out of 3 levels with 1.15x efficiency:
    # Level 1 weight = 1, Level 2 weight = 2, Level 3 weight = 3 (Total = 6)
    # raw_score = (115 * 1 + 0 + 0) / 6 = 19.1667
    # completion_cap = (1 / 6) * 100 = 16.6667
    # final_score must be min(raw_score, completion_cap) = 16.6667
    env = EnvironmentEvaluation(
        game_id="test_game",
        levels=[
            LevelMetric(level_index=1, completed=True, actions_taken=5, baseline_actions=10),
            LevelMetric(level_index=2, completed=False, actions_taken=20, baseline_actions=15),
            LevelMetric(level_index=3, completed=False, actions_taken=10, baseline_actions=20),
        ],
    )
    res = env.compute_rhae()
    assert res["completion_cap"] == 16.6667
    assert res["score"] == 16.6667  # Strictly bounded by completion cap
    assert res["raw_score"] > res["completion_cap"]


def test_local_proxy_metrics():
    env = EnvironmentEvaluation(
        game_id="test_game",
        levels=[
            LevelMetric(level_index=1, completed=True, actions_taken=10),
            LevelMetric(level_index=2, completed=True, actions_taken=15),
            LevelMetric(level_index=3, completed=False, actions_taken=25),
        ],
    )
    proxies = env.compute_local_proxies()
    assert proxies["max_depth"] == 2
    assert proxies["completion_rate"] == round(2 / 3, 4)
    assert proxies["total_actions"] == 50
