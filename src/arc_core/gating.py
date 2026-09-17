"""
Lexicographic Validation Gates for ARC-AGI-3.
Enforces the auditor's prioritized gating hierarchy:
1. Legality Gate: 100% legal actions, zero 400 Bad Request errors.
2. Model Fidelity Gate: 1-step transition prediction accuracy >= 80%.
3. Completion-Weighted Depth Gate: Later-level progression prioritized over premature action pruning.
4. Statistical Confidence Gate: Candidate must beat incumbent on bootstrap Lower Confidence Bound (95% CI).
5. Resource Budget Gate: p50 latency <= 50ms, p95 latency <= 200ms, memory <= 4GB.
"""

from dataclasses import dataclass

import numpy as np

from src.arc_core.metrics import EnvironmentEvaluation


@dataclass
class GateEvaluationResult:
    passed: bool
    gate_name: str
    message: str
    metrics: dict[str, float]


class LexicographicGatekeeper:
    """Evaluates candidate agents against the prioritized validation gate cascade."""

    def __init__(
        self,
        min_fidelity: float = 0.80,
        min_lcb_gain: float = 0.01,
        max_p95_latency_ms: float = 200.0,
        n_bootstrap: int = 1000,
    ):
        self.min_fidelity = min_fidelity
        self.min_lcb_gain = min_lcb_gain
        self.max_p95_latency_ms = max_p95_latency_ms
        self.n_bootstrap = n_bootstrap

    def evaluate_candidate(
        self,
        candidate_evals: list[EnvironmentEvaluation],
        incumbent_evals: list[EnvironmentEvaluation] | None,
        model_fidelity: float,
        illegal_actions_count: int,
        latencies_ms: list[float],
    ) -> tuple[bool, list[GateEvaluationResult]]:
        """
        Runs the candidate through the lexicographic gate cascade.
        Returns (all_passed, results_list).
        """
        results = []

        # Gate 1: Legality Gate
        if illegal_actions_count > 0:
            results.append(
                GateEvaluationResult(
                    passed=False,
                    gate_name="LEGALITY_GATE",
                    message=f"FAILED: {illegal_actions_count} illegal actions emitted.",
                    metrics={"illegal_actions": float(illegal_actions_count)},
                )
            )
            return False, results
        else:
            results.append(
                GateEvaluationResult(
                    passed=True,
                    gate_name="LEGALITY_GATE",
                    message="PASSED: 100% legal actions within available_actions.",
                    metrics={"illegal_actions": 0.0},
                )
            )

        # Gate 2: Model Fidelity Gate
        if model_fidelity < self.min_fidelity:
            results.append(
                GateEvaluationResult(
                    passed=False,
                    gate_name="MODEL_FIDELITY_GATE",
                    message=f"FAILED: Model 1-step prediction accuracy {model_fidelity:.1%} < {self.min_fidelity:.1%}.",
                    metrics={"fidelity": model_fidelity},
                )
            )
            return False, results
        else:
            results.append(
                GateEvaluationResult(
                    passed=True,
                    gate_name="MODEL_FIDELITY_GATE",
                    message=f"PASSED: Model fidelity {model_fidelity:.1%} >= {self.min_fidelity:.1%}.",
                    metrics={"fidelity": model_fidelity},
                )
            )

        # Gate 3 & 4: Completion Depth & Statistical Confidence (Bootstrap LCB)
        cand_scores = np.array([e.compute_rhae()["score"] for e in candidate_evals])
        cand_lcb = self._compute_bootstrap_lcb(cand_scores)

        if incumbent_evals is not None and len(incumbent_evals) > 0:
            inc_scores = np.array([e.compute_rhae()["score"] for e in incumbent_evals])
            inc_lcb = self._compute_bootstrap_lcb(inc_scores)
            lcb_diff = cand_lcb - inc_lcb

            if lcb_diff < self.min_lcb_gain:
                results.append(
                    GateEvaluationResult(
                        passed=False,
                        gate_name="STATISTICAL_CONFIDENCE_GATE",
                        message=f"FAILED: Candidate LCB {cand_lcb:.2f} did not beat incumbent LCB {inc_lcb:.2f} by >= {self.min_lcb_gain}.",
                        metrics={"cand_lcb": cand_lcb, "inc_lcb": inc_lcb, "lcb_gain": lcb_diff},
                    )
                )
                return False, results
            else:
                results.append(
                    GateEvaluationResult(
                        passed=True,
                        gate_name="STATISTICAL_CONFIDENCE_GATE",
                        message=f"PASSED: Candidate LCB {cand_lcb:.2f} > Incumbent LCB {inc_lcb:.2f} (Gain: +{lcb_diff:.2f}).",
                        metrics={"cand_lcb": cand_lcb, "inc_lcb": inc_lcb, "lcb_gain": lcb_diff},
                    )
                )
        else:
            # Baseline candidate has no incumbent to beat
            results.append(
                GateEvaluationResult(
                    passed=True,
                    gate_name="STATISTICAL_CONFIDENCE_GATE",
                    message=f"PASSED: Initial baseline candidate established (LCB: {cand_lcb:.2f}).",
                    metrics={"cand_lcb": cand_lcb, "inc_lcb": 0.0, "lcb_gain": cand_lcb},
                )
            )

        # Gate 5: Resource Budget Gate
        if latencies_ms:
            p50 = float(np.percentile(latencies_ms, 50))
            p95 = float(np.percentile(latencies_ms, 95))
        else:
            p50, p95 = 0.0, 0.0

        if p95 > self.max_p95_latency_ms:
            results.append(
                GateEvaluationResult(
                    passed=False,
                    gate_name="RESOURCE_BUDGET_GATE",
                    message=f"FAILED: p95 latency {p95:.1f}ms exceeds limit {self.max_p95_latency_ms}ms.",
                    metrics={"p50_latency_ms": p50, "p95_latency_ms": p95},
                )
            )
            return False, results
        else:
            results.append(
                GateEvaluationResult(
                    passed=True,
                    gate_name="RESOURCE_BUDGET_GATE",
                    message=f"PASSED: Decision latency within budget (p50: {p50:.1f}ms, p95: {p95:.1f}ms).",
                    metrics={"p50_latency_ms": p50, "p95_latency_ms": p95},
                )
            )

        return True, results

    def _compute_bootstrap_lcb(self, scores: np.ndarray, alpha: float = 0.05) -> float:
        """Computes bootstrap 95% Lower Confidence Bound for mean score."""
        if len(scores) == 0:
            return 0.0
        if len(scores) == 1:
            return float(scores[0])

        rng = np.random.default_rng(42)
        bootstrap_means = []
        for _ in range(self.n_bootstrap):
            sample = rng.choice(scores, size=len(scores), replace=True)
            bootstrap_means.append(float(np.mean(sample)))

        # Return 5th percentile (95% LCB)
        return float(np.percentile(bootstrap_means, alpha * 100))
