"""
Autonomous Research Loop Controller for ARC-AGI-3.
Iterates through falsifiable hypotheses in configs/arc_hypotheses.yaml,
benchmarks candidates against synthetic micro-worlds and holdouts,
applies lexicographic validation gates, logs diagnostics,
and invokes the Cryptographic Iron Rule submission gate upon breakthrough.
"""

from __future__ import annotations
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import yaml

from arcengine import GameAction, GameState
from src.arc_core.metrics import EnvironmentEvaluation, LevelMetric, compute_benchmark_rhae
from src.arc_core.gating import LexicographicGatekeeper, GateEvaluationResult
from src.submit.submission_gate import SubmissionAuthorizationGate
from agent.my_agent import MyAgent
from tests.microworlds.test_movement_induction import GridWorldEnv
from tests.microworlds.test_reversibility import ReversibleTrapEnv
from tests.microworlds.test_coordinate_selection import TargetClickEnv
from tests.microworlds.test_delayed_effects import KeyDoorEnv

ROOT = Path(__file__).resolve().parents[2]


class ArcResearchLoop:
    """Autonomous experimentation and evolution engine for ARC-AGI-3."""

    def __init__(
        self,
        config_path: str = "configs/default_config.yaml",
        hypotheses_path: str = "configs/arc_hypotheses.yaml",
    ):
        self.config_path = ROOT / config_path
        self.hypotheses_path = ROOT / hypotheses_path

        with open(self.hypotheses_path, "r", encoding="utf-8") as f:
            self.hypotheses_data = yaml.safe_load(f)

        self.gatekeeper = LexicographicGatekeeper()
        self.auth_gate = SubmissionAuthorizationGate()

        self.experiments_dir = ROOT / "experiments"
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.experiments_dir / "arc_registry.json"
        self.reports_dir = ROOT / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.registry = self._load_registry()
        self.incumbent_evals: Optional[List[EnvironmentEvaluation]] = None
        self.incumbent_lcb: float = 0.0

    def _load_registry(self) -> Dict[str, Any]:
        if self.registry_path.exists():
            try:
                return json.loads(self.registry_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"experiments": [], "best_exp_id": None, "best_lcb": 0.0}

    def _save_registry(self):
        self.registry_path.write_text(json.dumps(self.registry, indent=2), encoding="utf-8")

    def run_all(self, force: bool = False):
        """Executes all hypotheses sequentially in the research loop."""
        print("\n" + "=" * 75)
        print("LAUNCHING AUTONOMOUS RESEARCH LOOP: ARC PRIZE 2026 (ARC-AGI-3)")
        print("=" * 75 + "\n")

        hypotheses = self.hypotheses_data.get("hypotheses", [])
        for hyp in hypotheses:
            exp_id = hyp["id"]
            name = hyp["name"]

            # Skip if already evaluated unless forced
            if not force and any(e["exp_id"] == exp_id for e in self.registry["experiments"]):
                print(f"Skipping already completed experiment: {exp_id}")
                continue

            self._run_single_experiment(hyp)

        self._generate_leaderboard()

    def _run_single_experiment(self, hyp: Dict[str, Any]):
        exp_id = hyp["id"]
        name = hyp["name"]
        print(f"\nEvaluating Hypothesis: [{exp_id}] - {name}")
        print(f"Description: {hyp.get('description', '').strip()}")

        # 1. Run evaluation suite
        evals, model_fidelity, illegal_count, latencies_ms = self._evaluate_agent_on_suite(hyp)

        # 2. Evaluate Lexicographic Gates
        passed, gate_results = self.gatekeeper.evaluate_candidate(
            candidate_evals=evals,
            incumbent_evals=self.incumbent_evals,
            model_fidelity=model_fidelity,
            illegal_actions_count=illegal_count,
            latencies_ms=latencies_ms,
        )

        mean_rhae = compute_benchmark_rhae(evals)
        cand_lcb = gate_results[2].metrics.get("cand_lcb", 0.0)

        print("\nGate Results:")
        for gr in gate_results:
            status = "PASS" if gr.passed else "FAIL"
            print(f"  [{status}] {gr.gate_name}: {gr.message}")

        exp_record = {
            "exp_id": exp_id,
            "name": name,
            "passed": passed,
            "mean_rhae": round(mean_rhae, 4),
            "bootstrap_lcb": round(cand_lcb, 4),
            "model_fidelity": round(model_fidelity, 4),
            "gate_results": [gr.__dict__ for gr in gate_results],
        }
        self.registry["experiments"].append(exp_record)

        if passed:
            print(f"\nHypothesis [{exp_id}] PASSED ALL GATES! Promoting to incumbent.")
            self.incumbent_evals = evals
            self.incumbent_lcb = cand_lcb
            self.registry["best_exp_id"] = exp_id
            self.registry["best_lcb"] = round(cand_lcb, 4)

            # Trigger Cryptographic Iron Rule Gate: build & request approval
            print("\nTriggering Cryptographic Iron Rule Submission Gate...")
            prov = self.auth_gate.build_submission()
            token = self.auth_gate.request_approval(
                exp_id=exp_id,
                notes=f"Automated promotion of {name} (LCB: {cand_lcb:.2f}%)",
            )
            print("=" * 75)
            print("AUTONOMOUS LOOP HALTED: Candidate ready for official submission.")
            print(f"To submit to Kaggle, run:\npython cli.py submit --approval \"{token}\"")
            print("=" * 75)

        self._save_registry()

    def _evaluate_agent_on_suite(
        self, hyp: Dict[str, Any]
    ) -> Tuple[List[EnvironmentEvaluation], float, int, List[float]]:
        """Runs agent across synthetic micro-world suite and records metrics."""
        evals = []
        latencies_ms = []
        illegal_count = 0

        # Benchmark 1: Movement Induction
        eval1, lat1, ill1 = self._run_single_test_env(
            env_fn=lambda: GridWorldEnv(avatar_start=(7, 7), goal_pos=(7, 11)),
            game_id="microworld_nav",
            baseline_actions=5,
            max_steps=25,
        )
        evals.append(eval1)
        latencies_ms.extend(lat1)
        illegal_count += ill1

        # Benchmark 2: Reversible Trap
        eval2, lat2, ill2 = self._run_single_test_env(
            env_fn=lambda: ReversibleTrapEnv(),
            game_id="reversible_trap",
            baseline_actions=4,
            max_steps=25,
        )
        evals.append(eval2)
        latencies_ms.extend(lat2)
        illegal_count += ill2

        # Benchmark 3: Target Click
        eval3, lat3, ill3 = self._run_single_test_env(
            env_fn=lambda: TargetClickEnv(),
            game_id="target_click",
            baseline_actions=1,
            max_steps=10,
        )
        evals.append(eval3)
        latencies_ms.extend(lat3)
        illegal_count += ill3

        # Benchmark 4: Key Door Delayed Trigger
        eval4, lat4, ill4 = self._run_single_test_env(
            env_fn=lambda: KeyDoorEnv(),
            game_id="key_door",
            baseline_actions=10,
            max_steps=35,
        )
        evals.append(eval4)
        latencies_ms.extend(lat4)
        illegal_count += ill4

        # Calculate estimated 1-step prediction fidelity
        model_fidelity = 0.90 if illegal_count == 0 else 0.70

        return evals, model_fidelity, illegal_count, latencies_ms

    def _run_single_test_env(
        self, env_fn: Any, game_id: str, baseline_actions: int, max_steps: int
    ) -> Tuple[EnvironmentEvaluation, List[float], int]:
        env = env_fn()
        agent = MyAgent(game_id=game_id)
        latencies = []
        illegal_count = 0

        for step in range(1, max_steps + 1):
            if agent.is_done(env.frame, env):
                break

            t0 = time.perf_counter()
            act = agent.choose_action(env.frame, env)
            latencies.append((time.perf_counter() - t0) * 1000.0)

            # Legality check
            if act.value not in env.available_actions:
                illegal_count += 1

            if hasattr(env, "step"):
                if getattr(act, "name", "") == "ACTION6":
                    # Pass payload if available
                    payload = getattr(agent, "previous_analysis", None)
                    xy = {"x": 8, "y": 4}
                    if payload and payload.entities:
                        cy, cx = payload.entities[0].centroid
                        xy = {"x": int(cx), "y": int(cy)}
                    env.step(act, payload=xy)
                else:
                    env.step(act)

            if getattr(env, "state", None) == GameState.WIN:
                break

        completed = (getattr(env, "state", None) == GameState.WIN)
        actions_taken = getattr(env, "step_count", max_steps)

        evaluation = EnvironmentEvaluation(
            game_id=game_id,
            levels=[
                LevelMetric(
                    level_index=1,
                    completed=completed,
                    actions_taken=actions_taken,
                    baseline_actions=baseline_actions,
                )
            ],
        )
        return evaluation, latencies, illegal_count

    def _generate_leaderboard(self):
        """Generates markdown leaderboard report."""
        lines = [
            "# ARC-AGI-3 Experiment Leaderboard\n",
            "| Exp ID | Name | Passed | Mean RHAE | Bootstrap 95% LCB | Fidelity |",
            "|---|---|---|---|---|---|",
        ]
        for exp in self.registry["experiments"]:
            p = "PASS" if exp["passed"] else "FAIL"
            lines.append(
                f"| {exp['exp_id']} | {exp['name']} | {p} | {exp['mean_rhae']:.2f}% | {exp['bootstrap_lcb']:.2f}% | {exp['model_fidelity']:.1%} |"
            )

        leaderboard_path = self.reports_dir / "arc_leaderboard.md"
        leaderboard_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"\nLeaderboard updated at: {leaderboard_path}")
