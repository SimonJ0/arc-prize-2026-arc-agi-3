"""
Uncertainty-Aware Agent for ARC-AGI-3.
Combines:
1. Hard legality adapter (strictly conforms to available_actions, handles GAME_OVER -> RESET).
2. Multi-hypothesis layered perception (candidate backgrounds, connected components).
3. Factored belief-state world model (Bayesian posterior over transition dynamics).
4. Epistemic decision policy (epistemic probing vs bounded goal planning).
5. Scoped episode memory (bounds invariants to game run, avoids negative transfer).
"""

from __future__ import annotations
import hashlib
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from arcengine import GameAction, GameState, FrameDataRaw
from src.arc_core.contracts import Observation
from src.arc_agent.legality_adapter import LegalityAdapter
from src.arc_agent.perception.layered_perception import LayeredPerception, FrameAnalysis
from src.arc_agent.world_model.belief_state import BeliefStateWorldModel
from src.arc_agent.planning.epistemic_policy import EpistemicPolicy
from src.arc_agent.memory.scoped_memory import ScopedEpisodeMemory

# When running in official starter, `Agent` is imported from `agents.agent`
try:
    from agents.agent import Agent
except ImportError:
    # Base fallback for local testing without the starter framework wrapper
    class Agent:
        def __init__(self, game_id: str = "local_game", *args: Any, **kwargs: Any):
            self.game_id = game_id


class MyAgent(Agent):
    """
    Production-ready Uncertainty-Aware Agent for ARC-AGI-3.
    """
    MAX_ACTIONS = 120

    def __init__(self, game_id: str = "default_game", *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.game_id = getattr(self, "game_id", game_id)
        self.perception = LayeredPerception()
        self.world_model = BeliefStateWorldModel()
        self.policy = EpistemicPolicy()
        self.memory = ScopedEpisodeMemory(game_key=self.game_id)

        self.previous_observation: Optional[Observation] = None
        self.previous_analysis: Optional[FrameAnalysis] = None
        self.previous_action: Optional[str] = None
        self.action_count = 0

    def is_done(self, frames: Any, latest_frame: Any) -> bool:
        """Determines whether agent should stop playing."""
        if self.action_count >= self.MAX_ACTIONS:
            return True
        state = getattr(latest_frame, "state", None)
        if state in ("WIN", GameState.WIN, GameState.WIN.value):
            return True
        return False

    def choose_action(self, frames: Any, latest_frame: Any) -> GameAction:
        """
        Main decision loop for ARC-AGI-3 agent contract.
        Inspects environment state and returns a validated GameAction enum.
        """
        self.action_count += 1

        # Extract frame arrays
        raw_frames = getattr(latest_frame, "frame", [])
        if not raw_frames:
            if isinstance(frames, list) and frames:
                raw_frames = frames
            else:
                raw_frames = [np.zeros((16, 16), dtype=int)]

        grid = raw_frames[0] if isinstance(raw_frames, (list, tuple)) else raw_frames
        if not isinstance(grid, np.ndarray):
            grid = np.array(grid, dtype=int)

        # Extract metadata
        raw_state = getattr(latest_frame, "state", GameState.NOT_FINISHED)
        state_str = raw_state.value if hasattr(raw_state, "value") else str(raw_state)

        raw_avail = getattr(latest_frame, "available_actions", [])
        avail_actions = set()
        for a in raw_avail:
            if isinstance(a, int):
                try:
                    avail_actions.add(GameAction.from_id(a).name)
                except Exception:
                    pass
            elif hasattr(a, "name"):
                avail_actions.add(a.name)
            else:
                try:
                    avail_actions.add(GameAction.from_name(str(a)).name)
                except Exception:
                    avail_actions.add(str(a).split(".")[-1])


        if not avail_actions:
            avail_actions = {"RESET", "ACTION1"}

        current_obs = Observation(
            frames=(grid,),
            state=state_str,
            available_actions=frozenset(avail_actions),
            game_key=self.game_id,
            level=getattr(latest_frame, "levels_completed", 0) + 1,
            action_count=self.action_count,
            guid=getattr(latest_frame, "guid", None),
        )

        # Perception
        prev_grid = self.previous_observation.frames[0] if self.previous_observation else None
        current_analysis = self.perception.analyze(grid, prev_grid)

        # Belief State Update from previous transition
        if (
            self.previous_observation is not None
            and self.previous_analysis is not None
            and self.previous_action is not None
        ):
            self.world_model.update_with_transition(
                prev_obs=self.previous_observation,
                action=self.previous_action,
                curr_obs=current_obs,
                prev_analysis=self.previous_analysis,
                curr_analysis=current_analysis,
            )

            # Record event in scoped memory
            self.memory.record_transition(
                step=self.action_count,
                level=current_obs.level,
                from_frame=prev_grid,
                action=self.previous_action,
                payload={},
                to_state=state_str,
                to_frame=grid,
                planning_mode="ONLINE_DECISION",
                confidence=self.world_model.belief.one_step_accuracy,
            )

        # Decision Policy: Probing vs Goal Planning
        action_name, payload, trace = self.policy.select_action(
            observation=current_obs,
            analysis=current_analysis,
            world_model=self.world_model,
        )

        # Update tracking
        self.previous_observation = current_obs
        self.previous_analysis = current_analysis
        self.previous_action = action_name

        return LegalityAdapter.to_game_action(action_name)
