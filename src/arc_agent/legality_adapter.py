"""
Hard Legality Adapter for ARC-AGI-3.
Ensures zero 400 Bad Request errors by strictly enforcing:
1. When state == 'GAME_OVER', strictly emit RESET.
2. Only actions in the current frame's `available_actions` are emitted.
3. ACTION6 strictly requires integer coordinates (x, y) in range [0, 63].
4. Non-ACTION6 actions strip all coordinates.
"""

from collections.abc import Collection
from typing import Any

from arcengine import GameAction, GameState


class LegalityAdapter:
    """Hard guardrail ensuring every emitted action strictly adheres to environment contracts."""

    VALID_ACTIONS = {
        "RESET": GameAction.RESET,
        "ACTION1": GameAction.ACTION1,
        "ACTION2": GameAction.ACTION2,
        "ACTION3": GameAction.ACTION3,
        "ACTION4": GameAction.ACTION4,
        "ACTION5": GameAction.ACTION5,
        "ACTION6": GameAction.ACTION6,
        "ACTION7": GameAction.ACTION7,
    }

    @classmethod
    def validate_action(
        cls,
        state: str,
        available_actions: Collection[str],
        proposed_action: str,
        proposed_payload: dict[str, Any] | None = None,
        default_fallback: str = "RESET",
    ) -> tuple[str, dict[str, Any]]:
        """
        Validates and sanitizes proposed action against current environment state.

        Returns:
            Tuple of (sanitized_action_str, sanitized_payload_dict)
        """
        payload = dict(proposed_payload or {})

        # Rule 1: In GAME_OVER, the ONLY legal action is RESET
        if state in ("GAME_OVER", GameState.GAME_OVER.value, GameState.GAME_OVER):
            return "RESET", {}

        # Normalize action name string
        action_name = str(proposed_action).upper().strip()
        if action_name.startswith("GAMEACTION."):
            action_name = action_name.split(".")[-1]

        # Rule 2: Must be present in available_actions
        available_upper = {str(a).upper().split(".")[-1] for a in available_actions}
        if not available_upper:
            # If no available actions indicated, fallback to RESET
            return "RESET", {}

        if action_name not in available_upper:
            # Fallback to legal action
            if "ACTION1" in available_upper:
                action_name = "ACTION1"
            elif "RESET" in available_upper:
                action_name = "RESET"
            else:
                action_name = sorted(list(available_upper))[0]

        # Rule 3: Parameter rules
        if action_name == "ACTION6":
            # Must have valid x, y in [0, 63]
            raw_x = payload.get("x", 0)
            raw_y = payload.get("y", 0)
            try:
                x = int(raw_x)
                y = int(raw_y)
            except (ValueError, TypeError):
                x, y = 0, 0
            x = max(0, min(63, x))
            y = max(0, min(63, y))
            sanitized_payload = {"x": x, "y": y}
            if "game_id" in payload:
                sanitized_payload["game_id"] = payload["game_id"]
            return action_name, sanitized_payload
        else:
            # Strip coordinates for all simple actions & RESET
            sanitized_payload = {}
            if "reasoning" in payload:
                sanitized_payload["reasoning"] = payload["reasoning"]
            return action_name, sanitized_payload

    @classmethod
    def to_game_action(cls, action_str: str) -> GameAction:
        """Converts string action to arcengine GameAction enum."""
        key = action_str.upper().split(".")[-1]
        try:
            return GameAction.from_name(key)
        except Exception:
            return GameAction.RESET
