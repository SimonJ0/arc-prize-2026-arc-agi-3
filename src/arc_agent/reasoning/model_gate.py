"""
Conditional Model-Use Gate for ARC-AGI-3 (Baseline 3.0 B3.07).
Decides whether to act directly, simulate/plan through the world model, or perform
a discriminating epistemic probe. Avoids spending unnecessary computation or actions
constructing models when direct execution is already validated.
"""

from __future__ import annotations

from enum import Enum


class ModelUseDecision(str, Enum):
    """Decision modes for action generation."""

    ACT_DIRECTLY = "ACT_DIRECTLY"       # High confidence or active plan in flight; execute immediately
    SIMULATE_MODEL = "SIMULATE_MODEL"   # Verified transition dynamics available; plan/simulate trajectory
    PERFORM_PROBE = "PERFORM_PROBE"     # Epistemic uncertainty high; execute discriminating experiment


class ModelGate:
    """Arbitrates among direct execution, world-model planning, and epistemic probing."""

    def __init__(self, confidence_threshold: float = 0.75):
        self.confidence_threshold = confidence_threshold

    def evaluate_decision(
        self,
        has_active_plan: bool,
        can_reliably_plan: bool,
        has_exploitable_controller: bool,
        is_loop: bool = False,
    ) -> ModelUseDecision:
        """
        Determines the optimal computational and action strategy for the current turn.

        Args:
            has_active_plan: True if a verified multi-step macro-plan is already in flight.
            can_reliably_plan: True if world model kinematics and barriers are verified.
            has_exploitable_controller: True if a high-confidence goal-advancing controller is identified.
            is_loop: True if stagnation/deadlock loop was detected.

        Returns:
            ModelUseDecision member.
        """
        # Deadlock / loop breaks force epistemic probing
        if is_loop:
            return ModelUseDecision.PERFORM_PROBE

        # 1. Direct Execution: macro-plan in flight or verified exploitable controller
        if has_active_plan or has_exploitable_controller:
            return ModelUseDecision.ACT_DIRECTLY

        # 2. World Model Simulation: kinematics verified, plan shortest path to goal
        if can_reliably_plan:
            return ModelUseDecision.SIMULATE_MODEL

        # 3. Epistemic Probing: explore untested dynamics or mechanisms
        return ModelUseDecision.PERFORM_PROBE
