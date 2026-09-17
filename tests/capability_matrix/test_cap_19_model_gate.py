"""
Capability Matrix Test 19: Conditional Model-Use Gate (Baseline 3.0 B3.07).
Verifies:
- Direct execution prioritization when macro-plans or exploitable controllers exist
- Simulation/model planning routing when kinematics are verified
- Epistemic probing fallback under uncertainty
- Deadlock loop override forcing probe mode
"""

import pytest
from src.arc_agent.reasoning.model_gate import ModelGate, ModelUseDecision


def test_model_gate_routing_logic():
    """ModelGate accurately routes decision strategy based on epistemic and execution readiness."""
    gate = ModelGate()

    # 1. Macro-plan in flight -> ACT_DIRECTLY
    decision = gate.evaluate_decision(
        has_active_plan=True,
        can_reliably_plan=True,
        has_exploitable_controller=False,
    )
    assert decision == ModelUseDecision.ACT_DIRECTLY

    # 2. Exploitable controller identified -> ACT_DIRECTLY
    decision = gate.evaluate_decision(
        has_active_plan=False,
        can_reliably_plan=False,
        has_exploitable_controller=True,
    )
    assert decision == ModelUseDecision.ACT_DIRECTLY

    # 3. Validated world model dynamics -> SIMULATE_MODEL
    decision = gate.evaluate_decision(
        has_active_plan=False,
        can_reliably_plan=True,
        has_exploitable_controller=False,
    )
    assert decision == ModelUseDecision.SIMULATE_MODEL

    # 4. Unknown dynamics and no active controllers -> PERFORM_PROBE
    decision = gate.evaluate_decision(
        has_active_plan=False,
        can_reliably_plan=False,
        has_exploitable_controller=False,
    )
    assert decision == ModelUseDecision.PERFORM_PROBE

    # 5. Stagnation / deadlock loop forces probe regardless of plan or model state
    decision = gate.evaluate_decision(
        has_active_plan=True,
        can_reliably_plan=True,
        has_exploitable_controller=True,
        is_loop=True,
    )
    assert decision == ModelUseDecision.PERFORM_PROBE
