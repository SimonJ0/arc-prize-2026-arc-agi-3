"""
ARC-AGI-3 Official Platform Benchmarking Suite.
Connects to the ARC-AGI Platform API via arc_agi.Arcade to execute agents
on real platform environments with exact human baselines, computing official RHAE scores.
Supports family-disjoint train/holdout splits to prevent public-game overfitting.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import arc_agi
import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState
from dotenv import load_dotenv

from agent.my_agent import MyAgent
from src.arc_core.metrics import EnvironmentEvaluation, LevelMetric

# Load environment variables (ARC_API_KEY)
load_dotenv()

ROOT = Path(__file__).resolve().parents[2]

# Disjoint Train / Holdout Partition of the 25 official platform games
# 15 Training / Diagnostic Games (for development & regression testing)
TRAIN_GAMES: list[str] = [
    "ls20-9607627b",
    "sc25-635fd71a",
    "m0r0-492f87ba",
    "tu93-0768757b",
    "s5i5-18d95033",
    "bp35-0a0ad940",
    "cn04-2fe56bfb",
    "sk48-d8078629",
    "lf52-271a04aa",
    "su15-1944f8ab",
    "vc33-5430563c",
    "cd82-fb555c5d",
    "ft09-0d8bbf25",
    "wa30-ee6fef47",
    "tn36-ef4dde99",
]

# 10 Frozen Holdout Games (strictly reserved for LCB promotion gates, never tuned against)
HOLDOUT_GAMES: list[str] = [
    "sp80-589a99af",
    "sb26-7fbdac44",
    "ka59-38d34dbb",
    "ar25-0c556536",
    "tr87-cd924810",
    "re86-8af5384d",
    "g50t-5849a774",
    "lp85-305b61c3",
    "dc22-fdcac232",
    "r11l-495a7899",
]


class PlatformBenchmarkSuite:
    """Official Platform Benchmark Suite for ARC-AGI-3."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        reports_dir: str = "reports",
    ):
        self.api_key = api_key or os.getenv("ARC_API_KEY")
        self.base_url = base_url or os.getenv("ARC_BASE_URL")
        self.reports_dir = ROOT / reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        kwargs: dict[str, Any] = {"arc_api_key": self.api_key}
        if self.base_url:
            kwargs["arc_base_url"] = self.base_url.rstrip("/")

        # Initialize Arcade interface
        self.arcade = arc_agi.Arcade(**kwargs)
        self.env_map: dict[str, Any] = {
            env.game_id: env for env in self.arcade.available_environments
        }

    def list_available_games(self) -> list[dict[str, Any]]:
        """Returns summary of all 25 platform environments."""
        return [
            {
                "game_id": env.game_id,
                "title": env.title,
                "tags": env.tags,
                "total_levels": len(env.baseline_actions),
                "baseline_actions": env.baseline_actions,
            }
            for env in self.arcade.available_environments
        ]

    def evaluate_game(
        self,
        game_id: str,
        agent_factory: Callable[[str], Any] | None = None,
        max_actions: int = 120,
        seed: int = 42,
    ) -> tuple[EnvironmentEvaluation, dict[str, Any]]:
        """
        Executes an agent on a single platform environment.
        Tracks levels completed, action counts, decision latencies, and legality.
        """
        env_info = self.env_map.get(game_id)
        if not env_info:
            raise ValueError(f"Game '{game_id}' not found in available platform environments.")

        baseline_actions: list[int] = env_info.baseline_actions or [50]
        total_levels = len(baseline_actions)

        # Instantiate environment via Arcade
        env = self.arcade.make(game_id=game_id, seed=seed)
        if env is None:
            raise RuntimeError(f"Failed to create environment for {game_id}")

        frame: FrameDataRaw = env.reset()

        # Instantiate agent
        if agent_factory is None:
            agent = MyAgent(game_id=game_id)
        else:
            agent = agent_factory(game_id)

        # Tracking state
        actions_per_level: dict[int, int] = {lvl: 0 for lvl in range(1, total_levels + 1)}
        level_completed_flags: dict[int, bool] = {lvl: False for lvl in range(1, total_levels + 1)}
        latencies_ms: list[float] = []
        legal_actions_count = 0
        illegal_actions_count = 0
        total_actions = 0
        resets_count = 0

        current_level_idx = getattr(frame, "levels_completed", 0) + 1
        frames_history = [frame.frame[0]]

        while total_actions < max_actions:
            if hasattr(agent, "is_done") and agent.is_done(frames_history, frame):
                break

            state = getattr(frame, "state", GameState.NOT_FINISHED)
            if state in (GameState.WIN, "WIN"):
                break

            # Choose action with latency profiling
            t0 = time.perf_counter()
            action = agent.choose_action(frames_history, frame)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(lat_ms)

            # Legality validation
            avail = getattr(frame, "available_actions", [])
            action_id = action.value[0] if isinstance(action.value, tuple) else int(action.value)
            if action_id in avail or (hasattr(GameAction, "RESET") and action == GameAction.RESET):
                legal_actions_count += 1
            else:
                illegal_actions_count += 1

            if action_id == 0 or (hasattr(GameAction, "RESET") and action == GameAction.RESET):
                resets_count += 1

            # Step environment
            payload = getattr(agent, "last_payload", None)
            next_frame: FrameDataRaw | None
            if payload:
                next_frame = env.step(action, data=payload)
            else:
                next_frame = env.step(action)
            if next_frame is None:
                break

            total_actions += 1
            actions_per_level[current_level_idx] = actions_per_level.get(current_level_idx, 0) + 1

            # Check level transition
            new_levels_completed = getattr(next_frame, "levels_completed", 0)
            if new_levels_completed >= current_level_idx:
                for completed_lvl in range(current_level_idx, new_levels_completed + 1):
                    level_completed_flags[completed_lvl] = True
                current_level_idx = new_levels_completed + 1

            # Update observation
            frame = next_frame
            if hasattr(frame, "frame") and frame.frame:
                frames_history = [frame.frame[0]]

            if getattr(frame, "state", None) in (GameState.WIN, "WIN"):
                level_completed_flags[current_level_idx] = True
                break

        # Build EnvironmentEvaluation metric object
        level_metrics: list[LevelMetric] = []
        for lvl in range(1, total_levels + 1):
            h_bl = baseline_actions[lvl - 1] if lvl - 1 < len(baseline_actions) else 50
            completed = level_completed_flags.get(lvl, False)
            actions = actions_per_level.get(lvl, 0)
            level_metrics.append(
                LevelMetric(
                    level_index=lvl,
                    completed=completed,
                    actions_taken=actions,
                    baseline_actions=h_bl,
                )
            )

        env_eval = EnvironmentEvaluation(
            game_id=game_id,
            levels=level_metrics,
            resets=resets_count,
        )

        rhae_dict = env_eval.compute_rhae()
        p50 = float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0
        p95 = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0
        legality_rate = (
            legal_actions_count / (legal_actions_count + illegal_actions_count)
            if (legal_actions_count + illegal_actions_count) > 0
            else 1.0
        )

        diagnostics = {
            "game_id": game_id,
            "title": getattr(env_info, "title", game_id),
            "score": rhae_dict["score"],
            "raw_score": rhae_dict["raw_score"],
            "completion_cap": rhae_dict["completion_cap"],
            "levels_completed": rhae_dict["levels_completed"],
            "total_levels": total_levels,
            "total_actions": total_actions,
            "resets": resets_count,
            "legality_rate": round(legality_rate, 4),
            "latency_p50_ms": round(p50, 2),
            "latency_p95_ms": round(p95, 2),
            "level_details": [
                {
                    "level": lm.level_index,
                    "completed": lm.completed,
                    "actions_taken": lm.actions_taken,
                    "baseline_actions": lm.baseline_actions,
                    "level_score": round(lm.level_score, 4),
                }
                for lm in level_metrics
            ],
        }

        return env_eval, diagnostics

    def evaluate_suite(
        self,
        split: str = "train",
        games_subset: list[str] | None = None,
        max_actions: int = 100,
        agent_factory: Callable[[str], Any] | None = None,
        seed: int = 42,
    ) -> dict[str, Any]:
        """
        Runs evaluation across the designated split or subset of platform games.
        Computes aggregate RHAE, bootstrap 95% LCB, and produces structured scorecard.
        """
        if games_subset:
            target_games = games_subset
        elif split == "train":
            target_games = TRAIN_GAMES
        elif split == "holdout":
            target_games = HOLDOUT_GAMES
        elif split == "all":
            target_games = TRAIN_GAMES + HOLDOUT_GAMES
        else:
            raise ValueError(f"Unknown split '{split}'. Use 'train', 'holdout', or 'all'.")

        print(f"\n{'=' * 75}")
        print(
            f"OFFICIAL ARC-AGI PLATFORM BENCHMARK ({split.upper()} SUITE - {len(target_games)} GAMES)"
        )
        print(f"{'=' * 75}")

        evaluations: list[EnvironmentEvaluation] = []
        diagnostics_list: list[dict[str, Any]] = []

        for idx, game_id in enumerate(target_games, 1):
            title = self.env_map.get(game_id, {}).title if game_id in self.env_map else game_id
            print(f"\n[{idx}/{len(target_games)}] Running game: {title} ({game_id})...")
            try:
                env_eval, diag = self.evaluate_game(
                    game_id=game_id,
                    agent_factory=agent_factory,
                    max_actions=max_actions,
                    seed=seed,
                )
                evaluations.append(env_eval)
                diagnostics_list.append(diag)
                print(
                    f"   Result: Score={diag['score']} (Levels: {diag['levels_completed']}/{diag['total_levels']}, "
                    f"Actions: {diag['total_actions']}, Legality: {diag['legality_rate'] * 100:.1f}%)"
                )
            except Exception as e:
                print(f"   ERROR running game {game_id}: {e}")
                # Log failed game with 0 score
                empty_eval = EnvironmentEvaluation(game_id=game_id, levels=[])
                evaluations.append(empty_eval)
                diagnostics_list.append(
                    {"game_id": game_id, "error": str(e), "score": 0.0, "levels_completed": 0}
                )

        # Aggregate Statistics
        scores = [d["score"] for d in diagnostics_list if "score" in d]
        mean_score = float(np.mean(scores)) if scores else 0.0
        total_levels_comp = sum(d.get("levels_completed", 0) for d in diagnostics_list)
        total_levels_avail = sum(d.get("total_levels", 0) for d in diagnostics_list)

        # Bootstrap 95% LCB
        if len(scores) >= 3:
            rng = np.random.default_rng(seed)
            boot_means = [
                np.mean(rng.choice(scores, size=len(scores), replace=True)) for _ in range(1000)
            ]
            lcb_95 = float(np.percentile(boot_means, 5))
        else:
            lcb_95 = mean_score

        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "split": split,
            "games_evaluated": len(target_games),
            "mean_rhae_score": round(mean_score, 4),
            "bootstrap_95_lcb": round(lcb_95, 4),
            "total_levels_completed": total_levels_comp,
            "total_levels_available": total_levels_avail,
            "overall_completion_rate": (
                round(total_levels_comp / total_levels_avail, 4) if total_levels_avail > 0 else 0.0
            ),
            "per_game_results": diagnostics_list,
        }

        # Save report
        report_path = self._save_report(summary, split)
        summary["report_path"] = str(report_path)

        print(f"\n{'=' * 75}")
        print("SUITE COMPLETE:")
        print(f"  - Mean RHAE Score:       {summary['mean_rhae_score']}")
        print(f"  - Bootstrap 95% LCB:     {summary['bootstrap_95_lcb']}")
        print(
            f"  - Levels Completed:      {total_levels_comp}/{total_levels_avail} ({summary['overall_completion_rate'] * 100:.1f}%)"
        )
        print(f"  - Scorecard Saved At:    {report_path}")
        print(f"{'=' * 75}\n")

        return summary

    def _save_report(self, summary: dict[str, Any], split: str) -> Path:
        """Saves evaluation JSON and updates markdown leaderboard."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.reports_dir / f"platform_eval_{split}_{ts}.json"
        report_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        # Update or append to reports/arc_leaderboard.md
        board_file = self.reports_dir / "arc_leaderboard.md"
        row = (
            f"| {ts} | PLATFORM_{split.upper()} | {summary['mean_rhae_score']} | "
            f"{summary['bootstrap_95_lcb']} | {summary['total_levels_completed']}/{summary['total_levels_available']} | "
            f"{summary['games_evaluated']} | [JSON]({report_file.name}) |\n"
        )

        if not board_file.exists():
            header = (
                "# ARC-AGI-3 Evaluation Leaderboard\n\n"
                "| Timestamp | Model / Split | Mean RHAE | 95% LCB | Levels Completed | Games | Artifact |\n"
                "|---|---|---|---|---|---|---|\n"
            )
            board_file.write_text(header + row, encoding="utf-8")
        else:
            board_file.write_text(board_file.read_text(encoding="utf-8") + row, encoding="utf-8")

        return report_file

    def evaluate_multi_budget(
        self,
        split: str = "holdout",
        games_subset: list[str] | None = None,
        budgets: list[int] | None = None,
        agent_factory: Callable[[str], Any] | None = None,
        seed: int = 42,
    ) -> dict[str, Any]:
        """
        Runs multi-budget evaluation across designated split (e.g. holdout)
        for multiple action budget limits (e.g. 50, 100, 200).
        Calculates performance scaling curves, levels completed, and bootstrap LCB.
        """
        if budgets is None:
            budgets = [50, 100, 200]

        print(f"\n{'=' * 75}")
        print(
            f"OFFICIAL ARC-AGI MULTI-BUDGET BENCHMARK ({split.upper()} SUITE - BUDGETS: {budgets})"
        )
        print(f"{'=' * 75}")

        budget_results = {}
        for b in budgets:
            print(f"\n--- Commencing evaluation for Action Budget: {b} ---")
            summary = self.evaluate_suite(
                split=split,
                games_subset=games_subset,
                max_actions=b,
                agent_factory=agent_factory,
                seed=seed,
            )
            budget_results[str(b)] = summary

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        multi_summary: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "split": split,
            "budgets": budgets,
            "results_by_budget": budget_results,
            "budget_curve": {
                b: {
                    "mean_rhae": budget_results[str(b)]["mean_rhae_score"],
                    "bootstrap_95_lcb": budget_results[str(b)]["bootstrap_95_lcb"],
                    "levels_completed": budget_results[str(b)]["total_levels_completed"],
                    "total_levels": budget_results[str(b)]["total_levels_available"],
                    "completion_rate": budget_results[str(b)]["overall_completion_rate"],
                }
                for b in budgets
            },
        }

        # Save multi-budget report
        multi_report_file = self.reports_dir / f"platform_multi_budget_{split}_{ts}.json"
        multi_report_file.write_text(json.dumps(multi_summary, indent=2), encoding="utf-8")
        multi_summary["report_path"] = str(multi_report_file)

        # Append each budget's result to arc_leaderboard.md
        board_file = self.reports_dir / "arc_leaderboard.md"
        if board_file.exists():
            rows = ""
            for b in budgets:
                res = budget_results[str(b)]
                rows += (
                    f"| {ts} | PLATFORM_{split.upper()}_B{b} | {res['mean_rhae_score']} | "
                    f"{res['bootstrap_95_lcb']} | {res['total_levels_completed']}/{res['total_levels_available']} | "
                    f"{res['games_evaluated']} | [JSON]({multi_report_file.name}) |\n"
                )
            board_file.write_text(board_file.read_text(encoding="utf-8") + rows, encoding="utf-8")

        print(f"\n{'=' * 75}")
        print("MULTI-BUDGET EVALUATION COMPLETE:")
        for b in budgets:
            c = multi_summary["budget_curve"][b]
            print(
                f"  Budget {b:3d}: Mean RHAE={c['mean_rhae']} | 95% LCB={c['bootstrap_95_lcb']} | Levels={c['levels_completed']}/{c['total_levels']}"
            )
        print(f"  Multi-Budget Scorecard Saved At: {multi_report_file}")
        print(f"{'=' * 75}\n")

        return multi_summary
