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

from typing import Any

import numpy as np
from arcengine import GameAction, GameState

from src.arc_agent.diagnostics.replay_logger import ReplayLogger
from src.arc_agent.legality_adapter import LegalityAdapter
from src.arc_agent.memory.effect_taxonomy import EffectType, classify_effect
from src.arc_agent.memory.level_transfer import LevelTransferManager
from src.arc_agent.memory.mechanism_memory import EntitySignature
from src.arc_agent.memory.reasoning_state import (
    ContextCompactor,
    PersistentReasoningState,
)
from src.arc_agent.memory.scoped_memory import ScopedEpisodeMemory
from src.arc_agent.memory.structured_belief import StructuredBeliefState
from src.arc_agent.perception.cognitive_hierarchy import (
    CognitiveHierarchyAnalysis,
    CognitiveHierarchyPerception,
)
from src.arc_agent.perception.layered_perception import FrameAnalysis, LayeredPerception
from src.arc_agent.planning.deadlock_taxonomy import DeadlockTaxonomyEngine, DeadlockType
from src.arc_agent.planning.epistemic_policy import EpistemicPolicy
from src.arc_agent.world_model.belief_state import BeliefStateWorldModel
from src.arc_agent.world_model.falsification_engine import (
    FalsificationEngine,
    ForwardPrediction,
)
from src.arc_core.contracts import Observation

# When running in official starter, `Agent` is imported from `agents.agent`
try:
    from agents.agent import Agent
except ImportError:
    # Base fallback for local testing without the starter framework wrapper
    class Agent:  # type: ignore[no-redef]
        def __init__(self, game_id: str = "local_game", *args: Any, **kwargs: Any):
            self.game_id = game_id


class MyAgent(Agent):
    """
    Production-ready Uncertainty-Aware Agent for ARC-AGI-3 (Baseline 2.0: FD-NSA).
    """

    MAX_ACTIONS = 1000

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
        self.perception = LayeredPerception()
        self.cognitive_perception = CognitiveHierarchyPerception()
        self.world_model = BeliefStateWorldModel()
        self.policy = EpistemicPolicy()
        self.memory = ScopedEpisodeMemory(game_key=self.game_id)
        self.reasoning_state = PersistentReasoningState(game_id=self.game_id)
        self.compactor = ContextCompactor()

        # Baseline 2.0: 5-Tier Memory, Falsification Engine, Deadlock Taxonomy, Transfer Manager
        self.structured_belief = StructuredBeliefState(game_id=self.game_id)
        self.falsification_engine = FalsificationEngine()
        self.deadlock_engine = DeadlockTaxonomyEngine()
        self.level_transfer = LevelTransferManager()
        self.replay_logger = ReplayLogger(game_id=self.game_id)
        self.previous_prediction: ForwardPrediction | None = None

        self.previous_observation: Observation | None = None
        self.previous_analysis: FrameAnalysis | None = None
        self.previous_cognitive: CognitiveHierarchyAnalysis | None = None
        self.previous_action: str | None = None
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
        if current_level != self.structured_belief.current_level:
            # 0. Cross-level invariant extraction upon level progression
            self.level_transfer.extract_level_schema(
                level_index=self.structured_belief.current_level,
                kinematic_mappings=self.falsification_engine.kinematic_mappings,
                avatar_size=1,
                actions_used=set(self.falsification_engine.kinematic_mappings.keys()),
                skill_memory=self.structured_belief.skills,
                game_id=self.game_id,
                mechanism_memory=self.structured_belief.mechanisms,
            )
            self.structured_belief.reset_level(current_level)
            self.deadlock_engine.reset_level()
            self.reasoning_state.reset_level(current_level)
            # Apply transferred mechanism priors to the new level
            self.level_transfer.apply_mechanism_priors(self.structured_belief.mechanisms)

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
            level=current_level,
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
            elif (
                cognitive_analysis.player_color != self.reasoning_state.player_color
                and prev_grid is not None
            ):
                self.reasoning_state.player_color = cognitive_analysis.player_color

        # 2. Prediction-Error Evaluation & Bayesian Belief Revision
        curr_pos = cognitive_analysis.player_pos if cognitive_analysis else None
        prev_pos = self.previous_cognitive.player_pos if self.previous_cognitive else None

        if self.previous_action is not None and self.previous_prediction is not None:
            diff_count = (
                int(np.sum(current_analysis.dynamic_diff_mask))
                if current_analysis.dynamic_diff_mask is not None
                else 0
            )

            # Compute decomposed prediction error vector
            pred_error = self.falsification_engine.evaluate_prediction_error(
                prediction=self.previous_prediction,
                actual_pos=curr_pos,
                prev_pos=prev_pos,
                level_advanced=(current_level > self.structured_belief.current_level),
                grid_diff_count=diff_count,
            )

            # Bayesian update and conditional falsification
            self.falsification_engine.update_and_falsify(
                action=self.previous_action,
                prev_pos=prev_pos,
                actual_pos=curr_pos,
                prediction=self.previous_prediction,
                error=pred_error,
                hypotheses=self.structured_belief.hypotheses,
            )

            # Record transition in causal memory
            self.structured_belief.causal.record(
                step=self.action_count - 1,
                action=self.previous_action,
                payload=getattr(self, "last_payload", {}),
                prev_pos=prev_pos,
                curr_pos=curr_pos,
                diff_pixel_count=diff_count,
            )

            # Deadlock classification & recovery check
            tgt_coord = None
            if hasattr(self, "last_payload") and self.last_payload:
                tgt_coord = (self.last_payload.get("y"), self.last_payload.get("x"))

            deadlock = self.deadlock_engine.record_step(
                current_pos=curr_pos,
                action=self.previous_action,
                state=state_str,
                active_goal=self.structured_belief.goals.active_goal_coord,
                goal_reached_without_win=(
                    curr_pos == self.structured_belief.goals.active_goal_coord
                    and state_str not in ("WIN", "GameState.WIN")
                ),
                diff_count=diff_count,
                target_coord=tgt_coord,
            )
            if deadlock and deadlock.deadlock_type == DeadlockType.TYPE_2_WRONG_GOAL:
                if curr_pos:
                    self.structured_belief.goals.falsify(curr_pos)
                    self.reasoning_state.falsify_goal(curr_pos)

            # Coordinate affordance & mechanism ledger registration for ACTION6
            if self.previous_action == "ACTION6" and hasattr(self, "last_payload") and self.last_payload:
                px = self.last_payload.get("x")
                py = self.last_payload.get("y")
                if px is not None and py is not None:
                    is_lethal = state_str in ("GAME_OVER", "GameState.GAME_OVER") or (
                        deadlock is not None and deadlock.deadlock_type == DeadlockType.TYPE_5_LETHAL_TRAP
                    )
                    level_adv = (current_level > self.structured_belief.current_level)
                    self.reasoning_state.register_affordance_result(
                        coord=(py, px),
                        diff_count=diff_count,
                        is_lethal=bool(is_lethal),
                        state_str=state_str,
                        level_advanced=level_adv,
                    )

                    # B3.02: Resolve target entity and record in Causal Mechanism Ledger
                    target_sig = None
                    cand_pool = self.previous_analysis.entities if self.previous_analysis and self.previous_analysis.entities else current_analysis.entities
                    if cand_pool:
                        for ent in cand_pool:
                            if (py, px) in getattr(ent, "cells", ()):
                                target_sig = EntitySignature.from_entity(ent)
                                break
                        if target_sig is None:
                            best_d = float("inf")
                            best_ent = None
                            for ent in cand_pool:
                                d = abs(ent.centroid[0] - py) + abs(ent.centroid[1] - px)
                                if d < best_d and d <= 4.0:
                                    best_d = d
                                    best_ent = ent
                            if best_ent is not None:
                                target_sig = EntitySignature.from_entity(best_ent)

                    # Identify changed entity signatures
                    changed_sigs = []
                    if current_analysis.entities:
                        for ent in current_analysis.entities:
                            if getattr(ent, "is_dynamic", False):
                                changed_sigs.append(EntitySignature.from_entity(ent))

                    eff_type = self.reasoning_state.last_effect_type or classify_effect(
                        diff_count=diff_count,
                        state=state_str,
                        level_advanced=level_adv,
                        is_lethal=bool(is_lethal),
                    )
                    pre_hash = str(hash(prev_grid.tobytes())) if prev_grid is not None else ""
                    post_hash = str(hash(grid.tobytes()))

                    self.structured_belief.mechanisms.record_transition(
                        step=self.action_count - 1,
                        action=self.previous_action,
                        target_coord=(py, px),
                        target_entity=target_sig,
                        effect_type=eff_type,
                        diff_count=diff_count,
                        changed_entity_signatures=changed_sigs,
                        level_advanced=level_adv,
                        is_lethal=bool(is_lethal),
                        pre_frame_hash=pre_hash,
                        post_frame_hash=post_hash,
                    )

                    # B3.05: Update Goal-Variable Hypotheses
                    if eff_type in (
                        EffectType.LOCAL_MUTATION,
                        EffectType.STRUCTURAL_MUTATION,
                        EffectType.GLOBAL_MUTATION,
                        EffectType.LEVEL_ADVANCE,
                    ):
                        var_key = f"var_{target_sig.size_bucket if target_sig else 'unknown'}_{eff_type.value}"
                        self.structured_belief.goals.record_variable_transition(
                            var_name=var_key,
                            progress_occurred=bool(level_adv or eff_type == EffectType.STRUCTURAL_MUTATION),
                        )

            # Log step telemetry to ReplayLogger
            self.replay_logger.log_step(
                step=self.action_count - 1,
                level=self.structured_belief.current_level,
                observation_hash=str(hash(grid.tobytes())),
                action=self.previous_action,
                payload=getattr(self, "last_payload", {}),
                predicted_avatar_pos=self.previous_prediction.predicted_player_pos,
                actual_avatar_pos=curr_pos,
                prediction_error=pred_error,
                active_hypotheses=self.structured_belief.hypotheses.get_distribution(),
                active_goal=self.structured_belief.goals.active_goal_coord,
                deadlock_type=deadlock.deadlock_type.value if deadlock else None,
                state=state_str,
                levels_completed=getattr(latest_frame, "levels_completed", 0),
            )

        # Update perceptual and goal memories
        bg_col = (
            current_analysis.background_hypotheses[0][0]
            if current_analysis.background_hypotheses
            else 0
        )
        self.structured_belief.perceptual.update(
            grid=grid,
            entities=current_analysis.entities,
            player_pos=cognitive_analysis.player_pos,
            player_color=cognitive_analysis.player_color,
            bg_color=bg_col,
            step=self.action_count,
        )
        for goal_coord in cognitive_analysis.candidate_goals:
            self.structured_belief.goals.add_candidate(goal_coord, color=0)

        # Update Reasoning Persistence displacements
        if self.previous_action is not None and prev_pos is not None and curr_pos is not None:
            dy = curr_pos[0] - prev_pos[0]
            dx = curr_pos[1] - prev_pos[1]
            self.reasoning_state.update_action_effect(self.previous_action, dy, dx)

        # Context compaction
        self.compactor.compact_step(
            step=self.action_count,
            level=current_obs.level,
            action=self.previous_action or "NONE",
            curr_pos=curr_pos,
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
            structured_belief=self.structured_belief,
        )

        # 4. Generate 1-Step Forward Prediction for chosen action
        self.previous_prediction = self.falsification_engine.predict_next_state(
            action=action_name,
            current_pos=curr_pos,
            hypotheses=self.structured_belief.hypotheses,
            grid_shape=grid.shape,
            target_pos=self.structured_belief.goals.active_goal_coord,
        )

        # Update tracking
        self.previous_observation = current_obs
        self.previous_analysis = current_analysis
        self.previous_cognitive = cognitive_analysis
        self.previous_action = action_name
        self.last_payload = payload
        return LegalityAdapter.to_game_action(action_name, payload)
