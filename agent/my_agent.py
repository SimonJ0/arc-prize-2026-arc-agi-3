"""
Uncertainty-Aware Agent for ARC-AGI-3.
Integrates:
1. DRE-Bench 4-Level Cognitive Hierarchy (Attribute, Spatial, Sequential Macro-Planning, Intuitive Physics).
2. OpenAI Reasoning Persistence (Persistent working scratchpad & active multi-step macro-plans).
3. Automatic Context Compaction (Compact semantic delta summaries, zero context rot).
4. Hard legality adapter (strictly conforms to available_actions, handles GAME_OVER -> RESET).
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
from src.arc_agent.perception.cognitive_hierarchy import (
    CognitiveHierarchyPerception,
    CognitiveHierarchyAnalysis,
)
from src.arc_agent.world_model.belief_state import BeliefStateWorldModel
from src.arc_agent.planning.epistemic_policy import EpistemicPolicy
from src.arc_agent.memory.scoped_memory import ScopedEpisodeMemory
from src.arc_agent.memory.reasoning_state import (
    PersistentReasoningState,
    ContextCompactor,
)

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
    MAX_ACTIONS = 1000

    def __init__(self, game_id: str = "default_game", parameters: Optional[Dict[str, Any]] = None, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.game_id = getattr(self, "game_id", game_id)
        self.parameters = parameters or {}
        self.perception = LayeredPerception()
        self.cognitive_perception = CognitiveHierarchyPerception()
        self.world_model = BeliefStateWorldModel()
        self.policy = EpistemicPolicy()
        self.memory = ScopedEpisodeMemory(game_key=self.game_id)
        self.reasoning_state = PersistentReasoningState(game_id=self.game_id)
        self.compactor = ContextCompactor()

        self.previous_observation: Optional[Observation] = None
        self.previous_analysis: Optional[FrameAnalysis] = None
        self.previous_cognitive: Optional[CognitiveHierarchyAnalysis] = None
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

        current_level = getattr(latest_frame, "levels_completed", 0) + 1
        if current_level != self.reasoning_state.current_level:
            self.reasoning_state.reset_level(current_level)

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

        # 1. Perception & DRE-Bench Cognitive Analysis
        prev_grid = self.previous_observation.frames[0] if self.previous_observation else None
        current_analysis = self.perception.analyze(grid, prev_grid)
        cognitive_analysis = self.cognitive_perception.analyze(
            frame=grid,
            prev_frame=prev_grid,
            known_player_color=self.reasoning_state.player_color,
        )

        if cognitive_analysis.player_color is not None:
            if self.reasoning_state.player_color is None or cognitive_analysis.is_stagnant:
                self.reasoning_state.player_color = cognitive_analysis.player_color
            elif cognitive_analysis.player_color != self.reasoning_state.player_color and prev_grid is not None:
                self.reasoning_state.player_color = cognitive_analysis.player_color

        # 2. Update Reasoning Persistence (Causal displacements & deaths)
        if (
            self.previous_action is not None
            and self.previous_cognitive is not None
            and self.previous_cognitive.player_pos is not None
            and cognitive_analysis.player_pos is not None
        ):
            dy = cognitive_analysis.player_pos[0] - self.previous_cognitive.player_pos[0]
            dx = cognitive_analysis.player_pos[1] - self.previous_cognitive.player_pos[1]
            self.reasoning_state.update_action_effect(self.previous_action, dy, dx)

        # Context compaction
        prev_pos = self.previous_cognitive.player_pos if self.previous_cognitive else None
        self.compactor.compact_step(
            step=self.action_count,
            level=current_obs.level,
            action=self.previous_action or "NONE",
            curr_pos=cognitive_analysis.player_pos,
            prev_pos=prev_pos,
            state=state_str,
            levels_completed=getattr(latest_frame, "levels_completed", 0),
        )

        # Belief State Update
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

        # 3. Decision Policy: Macro-Planning vs Epistemic Probing
        action_name, payload, trace = self.policy.select_action(
            observation=current_obs,
            analysis=current_analysis,
            world_model=self.world_model,
            reasoning_state=self.reasoning_state,
            cognitive_analysis=cognitive_analysis,
        )

        # Update tracking
        self.previous_observation = current_obs
        self.previous_analysis = current_analysis
        self.previous_cognitive = cognitive_analysis
        self.previous_action = action_name
        self.last_payload = payload

        return LegalityAdapter.to_game_action(action_name)
