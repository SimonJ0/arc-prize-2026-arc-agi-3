"""
Scoped Episode Memory for ARC-AGI-3.
Binds learned facts strictly to (game_family, version, run_id).
Invalidates facts upon prediction error, and records hashable transition event logs
for reproducible replay analysis and debugging.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass(frozen=True)
class TransitionEvent:
    """Hashable record of an environment transition and agent decision."""
    step: int
    level: int
    from_frame_hash: str
    action: str
    payload: Dict[str, Any]
    to_state: str
    to_frame_hash: str
    planning_mode: str
    confidence: float
    prediction_error: bool


class ScopedEpisodeMemory:
    """Manages scoped invariant storage and transition event logging."""

    def __init__(self, game_key: str = "unknown", version: str = "1.0", run_id: str = "0"):
        self.game_key = game_key
        self.version = version
        self.run_id = run_id

        # Invariants carried across levels of the same game run
        self.confirmed_invariants: Dict[str, Any] = {}
        self.events: List[TransitionEvent] = []
        self.forbidden_transitions: set[Tuple[str, str]] = set()

    def reset_for_new_game(self, game_key: str, version: str = "1.0", run_id: str = "0"):
        """Completely resets all memories when switching to a novel environment."""
        self.game_key = game_key
        self.version = version
        self.run_id = run_id
        self.confirmed_invariants.clear()
        self.events.clear()
        self.forbidden_transitions.clear()

    def record_transition(
        self,
        step: int,
        level: int,
        from_frame: np.ndarray,
        action: str,
        payload: Dict[str, Any],
        to_state: str,
        to_frame: np.ndarray,
        planning_mode: str,
        confidence: float,
        prediction_error: bool = False,
    ):
        """Logs a transition event into memory."""
        from_hash = str(hash(from_frame.tobytes()))
        to_hash = str(hash(to_frame.tobytes()))

        event = TransitionEvent(
            step=step,
            level=level,
            from_frame_hash=from_hash,
            action=action,
            payload=dict(payload),
            to_state=to_state,
            to_frame_hash=to_hash,
            planning_mode=planning_mode,
            confidence=confidence,
            prediction_error=prediction_error,
        )
        self.events.append(event)

        if to_state == "GAME_OVER":
            self.forbidden_transitions.add((from_hash, action))

    def is_forbidden(self, frame: np.ndarray, action: str) -> bool:
        """Checks if this action from this state is known to cause GAME_OVER."""
        frame_hash = str(hash(frame.tobytes()))
        return (frame_hash, action) in self.forbidden_transitions
