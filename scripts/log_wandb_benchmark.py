"""
Wandb Logger for ARC-AGI-3 Benchmarks and Baselines.
Logs official RHAE scores, level completion, prediction fidelity,
and Kaggle submission provenance to Weights & Biases project 'arc-prize-2026-arc-agi-3'.
"""

from __future__ import annotations

import json
from pathlib import Path
import wandb

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "hypotheses_benchmark_report.json"
ENTITY = "sjosiah-org"
PROJECT = "arc-prize-2026-arc-agi-3"


def log_all_benchmarks():
    if not REPORT_PATH.exists():
        raise FileNotFoundError(f"Report file not found: {REPORT_PATH}")

    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)

    print(f"Logging benchmark runs to W&B: {ENTITY}/{PROJECT}...")
    run_urls = {}

    # 1. Historical Baseline 1.0 (Initial DRE-Bench baseline)
    run_b1 = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name="baseline-1.0-dre",
        job_type="baseline-progression",
        config={
            "architecture": "Baseline 1.0 (DRE-Bench Incumbent)",
            "version": "1.0",
            "effect_taxonomy": False,
            "action_algebra": False,
            "causal_equivalence": False,
            "goal_inference": False,
        },
        reinit=True,
    )
    run_b1.log({
        "rhae_score_pct": 0.3409,
        "levels_completed": 1,
        "total_levels": 70,
        "completion_rate": 1 / 70,
        "legality_rate": 1.0,
        "micro_worlds_solved": 4,
        "micro_worlds_total": 4,
        "prediction_fidelity": 0.70,
    })
    run_b1.summary["best_rhae_score_pct"] = 0.3409
    run_b1.summary["best_levels_completed"] = 1
    run_urls["baseline-1.0"] = run_b1.url
    run_b1.finish()

    # 2. Historical Baseline 2.1 (Affordance Causal Verification)
    run_b2 = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name="baseline-2.1-affordance-causal",
        job_type="baseline-progression",
        config={
            "architecture": "Baseline 2.1 (Affordance Causal Verification)",
            "version": "2.1",
            "delta_filter": ">2",
            "affordance_memory": True,
            "action_algebra": False,
        },
        reinit=True,
    )
    run_b2.log({
        "rhae_score_pct": 0.7540,
        "levels_completed": 2,
        "total_levels": 70,
        "completion_rate": 2 / 70,
        "legality_rate": 1.0,
        "micro_worlds_solved": 4,
        "micro_worlds_total": 4,
        "prediction_fidelity": 0.70,
        "lp85_level1_actions": 14,
        "r11l_level1_actions": 19,
    })
    run_b2.summary["best_rhae_score_pct"] = 0.7540
    run_b2.summary["best_levels_completed"] = 2
    run_urls["baseline-2.1"] = run_b2.url
    run_b2.finish()

    # 3. Baseline 3.0 (Causal Mechanism Memory B3.01 - B3.12)
    b3_details = report["details"]["baseline"]
    b3_summary = next(s for s in report["summary"] if s["id"] == "baseline")

    run_b3 = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name="baseline-3.0-causal-mechanism-memory",
        job_type="baseline-progression",
        tags=["baseline-3.0", "kaggle-submission", "causal-mechanism", "iron-rule"],
        config={
            "architecture": "Baseline 3.0 Causal Mechanism Memory (B3.01-B3.12)",
            "version": "3.0",
            "git_commit": "5247fcefb68e6e56c2a3095380d049f347671166",
            "git_clean": True,
            "notebook_sha256": "5f6b6f6e78f2f1767602ae70af91d5590d5b5076105bb8b8e484db3235f841bd",
            "agent_sha256": "29e92600cbd5ff7e6eb9bd2cd85411db78c7a6994263537a6e27a08809a9ec26",
            "kaggle_submission_ref": "56315568",
            "kaggle_submission_file": "submission.parquet",
            "kaggle_kernel": "simonjosiah/arc-prize-2026-arc-agi-3",
            "ablations": [
                "B3.01 Effect Taxonomy (8 Tiers)",
                "B3.02 Causal Transition Ledger",
                "B3.03 Action Algebra (Inverses & Toggles)",
                "B3.04 Causal Equivalence Classes",
                "B3.05 Goal-Variable Inference",
                "B3.06 Mechanism vs Goal Confidence",
                "B3.07 Model-Use Gate",
                "B3.08 Symbolic Compression",
                "B3.09 Counterfactual Simulation",
                "B3.10 Cross-Level Mechanism Transfer",
                "B3.11 RHAE Utility Planner",
                "B3.12 Deadlock Engine Type 3",
            ],
            "max_actions": report["max_actions"],
            "seed": report["seed"],
        },
        reinit=True,
    )

    # Log overall metrics
    run_b3.log({
        "rhae_score_pct": b3_summary["holdout_rhae_pct"],
        "levels_completed": b3_summary["levels_completed"],
        "total_levels": b3_summary["total_levels"],
        "completion_rate": b3_summary["completion_rate"],
        "legality_rate": b3_summary["legality_rate"],
        "prediction_fidelity": b3_summary["prediction_fidelity"],
        "micro_worlds_solved": b3_details["micro_worlds"]["solved_count"],
        "micro_worlds_total": b3_details["micro_worlds"]["total_envs"],
        "micro_worlds_steps": b3_details["micro_worlds"]["total_steps"],
        "micro_solve_rate": b3_details["micro_worlds"]["solve_rate"],
        "unit_tests_passed": 75,
        "unit_tests_total": 75,
        "lp85_score": 2.7778,
        "lp85_level1_actions": 14,
        "lp85_human_baseline": 17,
        "r11l_score": 0.9219,
        "r11l_level1_actions": 50,
        "r11l_human_baseline": 22,
        "latency_p50_ms": b3_summary["latency_p50_ms"],
        "latency_p95_ms": b3_summary["latency_p95_ms"],
        "eval_duration_sec": b3_summary["eval_duration_sec"],
    })

    # Log per-game table
    per_game = b3_details["platform_holdout"]["per_game_results"]
    game_table = wandb.Table(
        columns=["game_id", "title", "score", "levels_completed", "total_levels", "total_actions", "legality_rate", "latency_p50_ms"]
    )
    for g in per_game:
        game_table.add_data(
            g["game_id"],
            g["title"],
            g["score"],
            g["levels_completed"],
            g["total_levels"],
            g["total_actions"],
            g["legality_rate"],
            g["latency_p50_ms"],
        )
    run_b3.log({"holdout_per_game_results": game_table})

    # Log micro-worlds table
    micro_table = wandb.Table(columns=["micro_env", "solved", "steps", "legality_rate", "latency_p50_ms"])
    for m_name, m_val in b3_details["micro_worlds"]["details"].items():
        micro_table.add_data(
            m_name,
            m_val["solved"],
            m_val["steps"],
            m_val["legality_rate"],
            m_val["latency_p50_ms"],
        )
    run_b3.log({"micro_worlds_breakdown": micro_table})

    run_b3.summary["best_rhae_score_pct"] = b3_summary["holdout_rhae_pct"]
    run_b3.summary["best_levels_completed"] = b3_summary["levels_completed"]
    run_b3.summary["legality_rate"] = 1.0
    run_b3.summary["kaggle_ref"] = 56315568
    run_urls["baseline-3.0"] = run_b3.url
    run_b3.finish()

    # 4. Comparative Hypotheses (A, B, C, D)
    hyp_ids = ["hyp_a_latent_world_model", "hyp_b_object_dsl", "hyp_c_epistemic_curiosity", "hyp_d_hd_nsa"]
    for hid in hyp_ids:
        s_data = next((s for s in report["summary"] if s["id"] == hid), None)
        if not s_data:
            continue
        cand_meta = report["details"][hid]["metadata"]
        run_h = wandb.init(
            entity=ENTITY,
            project=PROJECT,
            name=hid,
            job_type="hypothesis-ablation",
            config={
                "candidate_id": hid,
                "candidate_name": cand_meta["name"],
                "description": cand_meta["desc"],
                "max_actions": report["max_actions"],
                "seed": report["seed"],
            },
            reinit=True,
        )
        run_h.log({
            "rhae_score_pct": s_data["holdout_rhae_pct"],
            "levels_completed": s_data["levels_completed"],
            "total_levels": s_data["total_levels"],
            "completion_rate": s_data["completion_rate"],
            "legality_rate": s_data["legality_rate"],
            "prediction_fidelity": s_data["prediction_fidelity"],
            "micro_solve_rate": s_data["micro_solve_rate"],
            "micro_steps": s_data["micro_steps"],
            "eval_duration_sec": s_data["eval_duration_sec"],
        })
        run_h.summary["best_rhae_score_pct"] = s_data["holdout_rhae_pct"]
        run_h.summary["best_levels_completed"] = s_data["levels_completed"]
        run_urls[hid] = run_h.url
        run_h.finish()

    print("\n" + "=" * 80)
    print("ALL RUNS SUCCESSFULLY LOGGED TO WANDB:")
    print("=" * 80)
    for name, url in run_urls.items():
        print(f"  - {name:<35}: {url}")
    print("=" * 80 + "\n")
    return run_urls


if __name__ == "__main__":
    log_all_benchmarks()
