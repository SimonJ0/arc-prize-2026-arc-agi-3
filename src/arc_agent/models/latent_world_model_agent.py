"""
Hypothesis A: Pure Latent World Model (Dreamer/MuZero Style) for ARC-AGI-3.
Components:
1. Spatial Grid Encoder: Projects 2D grid into latent representation z_t in R^32.
2. Latent Transition Model: Dynamics network predicting z_{t+1} given (z_t, a_t).
3. Latent Value Estimator: Estimates prospective task utility / progress.
4. Latent Monte Carlo Tree Search (MCTS): Plans in latent space to pick actions.
5. Hard Legality Guardrail: Conforms strictly to available_actions via LegalityAdapter.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
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


class LatentEncoder:
    """Projects 2D integer grids into a continuous latent vector in R^32."""

    def __init__(self, latent_dim: int = 32):
        self.latent_dim = latent_dim
        # Deterministic projection matrix for spatial patches
        rng = np.random.default_rng(42)
        self.proj = rng.standard_normal((16 + 16, latent_dim)) / math.sqrt(32)

    def encode(self, grid: np.ndarray) -> np.ndarray:
        if grid.ndim != 2:
            grid = np.squeeze(grid)
        H, W = grid.shape
        # Feature 1: Color histogram (16 bins, normalized)
        color_counts = np.bincount(grid.ravel(), minlength=16)[:16]
        hist = color_counts / max(1, grid.size)

        # Feature 2: Spatial density across 4x4 quadrants (16 bins)
        qh, qw = max(1, H // 4), max(1, W // 4)
        quadrants = []
        for i in range(4):
            for j in range(4):
                patch = grid[i * qh : (i + 1) * qh, j * qw : (j + 1) * qw]
                # Dominance ratio (non-zero or non-background density)
                quadrants.append(float(np.mean(patch > 0)) if patch.size > 0 else 0.0)

        raw_feat = np.concatenate([hist, np.array(quadrants, dtype=float)])
        z = np.tanh(raw_feat @ self.proj)
        # Normalize to unit sphere
        norm = np.linalg.norm(z)
        return z / (norm + 1e-8)


class LatentTransitionModel:
    """Predicts next latent state z_{t+1} = z_t + delta_a."""

    def __init__(self, latent_dim: int = 32, lr: float = 0.1):
        self.latent_dim = latent_dim
        self.lr = lr
        # Per-action displacement weights
        self.deltas: dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(self.latent_dim, dtype=float)
        )
        self.counts: dict[str, int] = defaultdict(int)

    def predict(self, z: np.ndarray, action: str) -> np.ndarray:
        pred_z = z + self.deltas[action]
        norm = np.linalg.norm(pred_z)
        return pred_z / (norm + 1e-8)

    def update(self, z_t: np.ndarray, action: str, z_next: np.ndarray):
        observed_delta = z_next - z_t
        self.counts[action] += 1
        # Online running average of displacement vector
        self.deltas[action] += self.lr * (observed_delta - self.deltas[action])


class LatentMCTS:
    """Monte Carlo Tree Search operating strictly inside latent state space."""

    def __init__(
        self,
        dynamics: LatentTransitionModel,
        rollout_depth: int = 4,
        num_simulations: int = 15,
        exploration_c: float = 1.414,
    ):
        self.dynamics = dynamics
        self.rollout_depth = rollout_depth
        self.num_simulations = num_simulations
        self.c = exploration_c

    def plan(
        self,
        root_z: np.ndarray,
        available_actions: list[str],
        target_z: np.ndarray | None = None,
    ) -> str:
        if not available_actions:
            return "RESET"
        if len(available_actions) == 1:
            return available_actions[0]

        # Action statistics
        visit_counts: dict[str, int] = defaultdict(int)
        total_rewards: dict[str, float] = defaultdict(float)

        for _ in range(self.num_simulations):
            # Select root candidate
            total_visits = sum(visit_counts.values())
            best_action = None
            best_ucb = -1e9

            for a in available_actions:
                if visit_counts[a] == 0:
                    best_action = a
                    break
                q = total_rewards[a] / visit_counts[a]
                u = self.c * math.sqrt(math.log(total_visits + 1) / visit_counts[a])
                score = q + u
                if score > best_ucb:
                    best_ucb = score
                    best_action = a

            assert best_action is not None

            # Rollout in latent space
            sim_z = root_z.copy()
            sim_reward = 0.0
            act = best_action
            for step in range(self.rollout_depth):
                sim_z = self.dynamics.predict(sim_z, act)
                # Intrinsic novelty reward: magnitude of state displacement
                disp = float(np.linalg.norm(sim_z - root_z))
                # If target available, reward alignment
                if target_z is not None:
                    target_sim = float(np.dot(sim_z, target_z))
                    sim_reward += target_sim * (0.9**step)
                else:
                    sim_reward += disp * (0.9**step)

                # Next action in rollout (greedy on known displacement magnitude)
                act = max(
                    available_actions,
                    key=lambda a_next: float(np.linalg.norm(self.dynamics.deltas[a_next])),
                )

            visit_counts[best_action] += 1
            total_rewards[best_action] += sim_reward

        # Pick most visited action
        return max(available_actions, key=lambda a: visit_counts[a])


class LatentWorldModelAgent(Agent):
    """
    Hypothesis A Agent: End-to-end discrete/continuous latent dynamics model
    with Latent MCTS decision planning.
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
        self.encoder = LatentEncoder(latent_dim=32)
        self.dynamics = LatentTransitionModel(latent_dim=32)
        self.mcts = LatentMCTS(
            dynamics=self.dynamics,
            rollout_depth=self.parameters.get("mcts_depth", 4),
            num_simulations=self.parameters.get("mcts_sims", 12),
        )

        self.previous_z: np.ndarray | None = None
        self.previous_action: str | None = None
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

        # Extract 2D grid
        raw_frames = getattr(latest_frame, "frame", [])
        if not raw_frames:
            if isinstance(frames, list) and frames:
                raw_frames = frames
            else:
                raw_frames = [np.zeros((16, 16), dtype=int)]

        grid = raw_frames[0] if isinstance(raw_frames, (list, tuple)) else raw_frames
        if not isinstance(grid, np.ndarray):
            grid = np.array(grid, dtype=int)

        # 1. Encode grid into latent representation
        current_z = self.encoder.encode(grid)

        # 2. Update transition dynamics with previous transition (z_{t-1}, a_{t-1}, z_t)
        if self.previous_z is not None and self.previous_action is not None:
            self.dynamics.update(self.previous_z, self.previous_action, current_z)

        # 3. Available actions normalization
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

        # 4. If GAME_OVER, reset immediately
        if state_str in ("GAME_OVER", GameState.GAME_OVER.value, GameState.GAME_OVER):
            chosen_action, payload = LegalityAdapter.validate_action(
                state=state_str,
                available_actions=avail_actions,
                proposed_action="RESET",
            )
        else:
            # 5. Latent MCTS Planning
            # Prefer non-reset moves during active gameplay
            non_reset_actions = [a for a in avail_actions if a != "RESET"] or avail_actions
            proposed_act = self.mcts.plan(current_z, non_reset_actions)

            # ACTION6 centroid click on foreground entity if proposed
            payload = {}
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

        self.previous_z = current_z
        self.previous_action = chosen_action
        self.last_payload = payload

        return LegalityAdapter.to_game_action(chosen_action, payload)
