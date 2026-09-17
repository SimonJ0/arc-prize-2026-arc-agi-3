"""
Hypothesis C: Pure Epistemic Curiosity Agent (Random Network Distillation / Free Energy) for ARC-AGI-3.
Components:
1. Target Network: Fixed random projection of 2D grid states into a 32-dim target embedding.
2. Predictor Network: Online learned linear/MLP model trained to predict target embeddings.
3. Intrinsic Reward / Free Energy: Prediction error ||f_pred(s') - f_target(s')||^2.
4. Active Epistemic Exploration: Directs moves that maximize intrinsic surprise.
5. Goal Latching: When a high-magnitude state delta spike occurs (door open, key collect),
   latches onto exploiting the triggered locus.
6. Hard Legality Guardrail: Conforms strictly to available_actions via LegalityAdapter.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from arcengine import GameAction, GameState

from src.arc_agent.legality_adapter import LegalityAdapter

try:
    from agents.agent import Agent
except ImportError:
    class Agent:  # type: ignore[no-redef]
        def __init__(self, game_id: str = "default_game", *args: Any, **kwargs: Any):
            self.game_id = game_id


class RndCuriosityEngine:
    """Random Network Distillation (RND) intrinsic curiosity module."""

    def __init__(self, feature_dim: int = 32, lr: float = 0.05):
        self.feature_dim = feature_dim
        self.lr = lr

        rng = np.random.default_rng(1337)
        # Fixed random target projection
        self.target_matrix = rng.standard_normal((16 + 16, feature_dim)) / np.sqrt(feature_dim)
        # Trainable online predictor
        self.pred_matrix = np.zeros((16 + 16, feature_dim), dtype=float)
        # Per-action historical surprise scores
        self.action_surprise: dict[str, float] = {}

    def extract_features(self, grid: np.ndarray) -> np.ndarray:
        if grid.ndim != 2:
            grid = np.squeeze(grid)
        H, W = grid.shape
        # Feature 1: Color distribution (16)
        color_counts = np.bincount(grid.ravel(), minlength=16)[:16]
        hist = color_counts / max(1, grid.size)

        # Feature 2: 4x4 spatial patch densities (16)
        qh, qw = max(1, H // 4), max(1, W // 4)
        quads = []
        for i in range(4):
            for j in range(4):
                patch = grid[i * qh : (i + 1) * qh, j * qw : (j + 1) * qw]
                quads.append(float(np.mean(patch > 0)) if patch.size > 0 else 0.0)

        return np.concatenate([hist, np.array(quads, dtype=float)])

    def compute_intrinsic_reward(self, raw_feat: np.ndarray) -> float:
        target_vec = np.tanh(raw_feat @ self.target_matrix)
        pred_vec = np.tanh(raw_feat @ self.pred_matrix)
        # Prediction error: MSE
        err = float(np.mean((target_vec - pred_vec) ** 2))
        # Update predictor towards target
        grad = raw_feat[:, None] * (pred_vec - target_vec)[None, :]
        self.pred_matrix -= self.lr * grad
        return err


class EpistemicCuriosityAgent(Agent):
    """
    Hypothesis C Agent: Epistemic exploration driven by prediction error (RND),
    with state-change latching to solve discovered goals.
    """

    def __init__(
        self,
        game_id: str = "default_game",
        parameters: dict[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.game_id = game_id
        self.parameters = parameters or {}
        self.rnd = RndCuriosityEngine()

        self.previous_frame: np.ndarray | None = None
        self.previous_action: str | None = None
        self.latched_action: str | None = None
        self.latch_remaining = 0
        self.last_payload: dict[str, Any] = {}
        self.action_count = 0
        self.max_actions = 1000

    def is_done(self, frames: Any, latest_frame: Any) -> bool:
        if self.action_count >= self.max_actions:
            return True
        state = getattr(latest_frame, "state", None)
        if state in ("WIN", GameState.WIN, GameState.WIN.value):
            return True
        return False

    def choose_action(self, frames: Any, latest_frame: Any) -> GameAction:
        self.action_count += 1

        raw_frames = getattr(latest_frame, "frame", [])
        if not raw_frames:
            if isinstance(frames, list) and frames:
                raw_frames = frames
            else:
                raw_frames = [np.zeros((16, 16), dtype=int)]

        grid = raw_frames[0] if isinstance(raw_frames, (list, tuple)) else raw_frames
        if not isinstance(grid, np.ndarray):
            grid = np.array(grid, dtype=int)

        # 1. Compute RND intrinsic curiosity / surprise
        raw_feat = self.rnd.extract_features(grid)
        intrinsic_surprise = self.rnd.compute_intrinsic_reward(raw_feat)

        # 2. Track surprise for previous action
        if self.previous_action is not None:
            prev_surp = self.rnd.action_surprise.get(self.previous_action, 0.0)
            self.rnd.action_surprise[self.previous_action] = 0.7 * prev_surp + 0.3 * intrinsic_surprise

        # 3. Detect state transition spikes (goal latching)
        diff_count = 0
        if self.previous_frame is not None and self.previous_frame.shape == grid.shape:
            diff_count = int(np.sum(self.previous_frame != grid))

        if diff_count > 10 and self.previous_action is not None:
            # Significant environment alteration: latch onto repeating or pursuing this action
            self.latched_action = self.previous_action
            self.latch_remaining = 3

        # 4. Normalize available actions
        raw_avail = getattr(latest_frame, "available_actions", [])
        avail_actions = []
        for a in raw_avail:
            if isinstance(a, int):
                try:
                    avail_actions.append(GameAction.from_id(a).name)
                except Exception:
                    pass
            elif hasattr(a, "name"):
                avail_actions.append(a.name)
            else:
                try:
                    avail_actions.append(GameAction.from_name(str(a)).name)
                except Exception:
                    avail_actions.append(str(a).split(".")[-1])

        if not avail_actions:
            avail_actions = ["RESET", "ACTION1"]

        state = getattr(latest_frame, "state", GameState.NOT_FINISHED)
        state_str = state.value if hasattr(state, "value") else str(state)

        if state_str in ("GAME_OVER", GameState.GAME_OVER.value, GameState.GAME_OVER):
            chosen_action, payload = LegalityAdapter.validate_action(
                state=state_str,
                available_actions=avail_actions,
                proposed_action="RESET",
            )
            self.latched_action = None
            self.latch_remaining = 0
        else:
            payload = {}
            non_reset = [a for a in avail_actions if a != "RESET"] or avail_actions

            if self.latch_remaining > 0 and self.latched_action in non_reset:
                proposed_act = self.latched_action
                self.latch_remaining -= 1
            else:
                # Pick action with highest epistemic surprise or least tried
                proposed_act = max(
                    non_reset,
                    key=lambda a: self.rnd.action_surprise.get(a, 1.0)
                    + (1.0 / (1 + self.action_count)),
                )

            if proposed_act == "ACTION6":
                unique_colors, counts = np.unique(grid, return_counts=True)
                bg_color = int(unique_colors[np.argmax(counts)])
                fg_coords = np.argwhere(grid != bg_color)
                if len(fg_coords) > 0:
                    cy = int(round(fg_coords[:, 0].mean()))
                    cx = int(round(fg_coords[:, 1].mean()))
                    payload = {"x": cx, "y": cy}
                else:
                    payload = {"x": grid.shape[1] // 2, "y": grid.shape[0] // 2}

            chosen_action, payload = LegalityAdapter.validate_action(
                state=state_str,
                available_actions=avail_actions,
                proposed_action=proposed_act,
                proposed_payload=payload,
            )

        self.previous_frame = grid.copy()
        self.previous_action = chosen_action
        self.last_payload = payload

        return LegalityAdapter.to_game_action(chosen_action, payload)
