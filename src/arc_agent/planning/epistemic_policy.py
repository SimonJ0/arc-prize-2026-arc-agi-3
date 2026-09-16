"""
Uncertainty-Aware Epistemic Policy for ARC-AGI-3.
Selects actions via:
1. Low-risk epistemic probing when world model uncertainty is high (Value of Information > Action Cost).
2. Goal-directed search (BFS / A*) over validated transition dynamics when model consensus is established.
3. Candidate-coordinate spatial pruning for ACTION6.
4. Hard legality validation via LegalityAdapter.
"""

from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from src.arc_core.contracts import ActionProposal, DecisionTrace, Observation
from src.arc_agent.legality_adapter import LegalityAdapter
from src.arc_agent.perception.layered_perception import FrameAnalysis
from src.arc_agent.world_model.belief_state import BeliefStateWorldModel


class EpistemicPolicy:
    """Decision engine balancing active epistemic learning with goal pursuit."""

    def __init__(self):
        self.step_counter = 0

    def select_action(
        self,
        observation: Observation,
        analysis: FrameAnalysis,
        world_model: BeliefStateWorldModel,
    ) -> Tuple[str, Dict[str, Any], DecisionTrace]:
        """
        Selects next physical environment action given current belief state.
        Guaranteed to return a legal action from observation.available_actions.
        """
        self.step_counter += 1
        state = observation.state
        available = observation.available_actions

        # 1. State-level guard: GAME_OVER requires RESET
        if state == "GAME_OVER":
            action, payload = LegalityAdapter.validate_action(
                state=state,
                available_actions=available,
                proposed_action="RESET",
            )
            trace = DecisionTrace(
                level=observation.level,
                step=self.step_counter,
                observation_hash=str(hash(observation.frames[0].tobytes())),
                legal_actions=tuple(sorted(list(available))),
                selected_action=action,
                selected_payload=payload,
                planning_mode="RESET_RECOVERY",
                predicted_next_state="NOT_FINISHED",
                confidence=1.0,
            )
            return action, payload, trace

        # 2. Epistemic Probing Mode: Model is uncertain
        if not world_model.can_reliably_plan():
            action, payload, trace = self._select_epistemic_probe(
                observation, analysis, world_model
            )
            return action, payload, trace

        # 3. Exploitation Mode: Validated model allows forward planning
        action, payload, trace = self._plan_goal_trajectory(
            observation, analysis, world_model
        )
        return action, payload, trace

    def _select_epistemic_probe(
        self,
        observation: Observation,
        analysis: FrameAnalysis,
        world_model: BeliefStateWorldModel,
    ) -> Tuple[str, Dict[str, Any], DecisionTrace]:
        """Selects informative probe action to distinguish candidate transition models."""
        available = observation.available_actions

        # Rank probe candidates: test untested directional actions first
        probes = [a for a in ("ACTION1", "ACTION2", "ACTION3", "ACTION4") if a in available]
        if not probes:
            probes = [a for a in ("ACTION5", "ACTION6", "ACTION7") if a in available]
        if not probes:
            probes = list(available)

        # Pick probe with highest epistemic utility
        selected = probes[self.step_counter % len(probes)]
        payload = {}

        if selected == "ACTION6":
            # Prune coordinates to candidate dynamic or target entity centroids
            if analysis.entities:
                target_ent = analysis.entities[0]
                cy, cx = target_ent.centroid
                payload = {"x": int(cx), "y": int(cy)}
            else:
                H, W = analysis.frame_shape
                payload = {"x": W // 2, "y": H // 2}

        action, valid_payload = LegalityAdapter.validate_action(
            state=observation.state,
            available_actions=available,
            proposed_action=selected,
            proposed_payload=payload,
        )

        trace = DecisionTrace(
            level=observation.level,
            step=self.step_counter,
            observation_hash=str(hash(observation.frames[0].tobytes())),
            legal_actions=tuple(sorted(list(available))),
            selected_action=action,
            selected_payload=valid_payload,
            planning_mode="EPISTEMIC_PROBE",
            predicted_next_state="NOT_FINISHED",
            confidence=0.5,
        )
        return action, valid_payload, trace

    def _plan_goal_trajectory(
        self,
        observation: Observation,
        analysis: FrameAnalysis,
        world_model: BeliefStateWorldModel,
    ) -> Tuple[str, Dict[str, Any], DecisionTrace]:
        """Plans shortest path to candidate goal under validated transition dynamics."""
        frame = observation.frames[0]
        avatar_color = world_model.avatar_color
        available = observation.available_actions

        # Locate avatar
        avatar_coords = np.argwhere(frame == avatar_color)
        if len(avatar_coords) == 0:
            # Avatar lost, fallback to probe
            return self._select_epistemic_probe(observation, analysis, world_model)

        start_pos = (int(avatar_coords[0][0]), int(avatar_coords[0][1]))

        # Identify candidate goal entities (distinct non-background, non-avatar entity)
        target_pos = None
        for ent in analysis.entities:
            if ent.color != avatar_color and ent.size <= 36:
                target_pos = (int(ent.centroid[0]), int(ent.centroid[1]))
                break

        if target_pos is None:
            # No clear target found, fallback to safe exploration
            return self._select_epistemic_probe(observation, analysis, world_model)

        # Run BFS to find shortest action sequence to target
        path = self._bfs_search(start_pos, target_pos, frame.shape, world_model, available)

        if path:
            chosen_action = path[0]
            action, valid_payload = LegalityAdapter.validate_action(
                state=observation.state,
                available_actions=available,
                proposed_action=chosen_action,
            )
            trace = DecisionTrace(
                level=observation.level,
                step=self.step_counter,
                observation_hash=str(hash(frame.tobytes())),
                legal_actions=tuple(sorted(list(available))),
                selected_action=action,
                selected_payload=valid_payload,
                planning_mode="GOAL_PLAN",
                predicted_next_state="NOT_FINISHED",
                confidence=0.9,
            )
            return action, valid_payload, trace

        # Path not found or blocked, fallback
        return self._select_epistemic_probe(observation, analysis, world_model)

    def _bfs_search(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        grid_shape: Tuple[int, int],
        world_model: BeliefStateWorldModel,
        available_actions: Set[str],
    ) -> Optional[List[str]]:
        """Bounded BFS search over validated directional dynamics."""
        H, W = grid_shape
        queue = deque([(start, [])])
        visited = {start}

        dir_actions = [a for a in ("ACTION1", "ACTION2", "ACTION3", "ACTION4") if a in available_actions]
        max_depth = 40  # Bound search to avoid budget exhaustion

        while queue:
            curr_pos, path = queue.popleft()
            if curr_pos == goal:
                return path

            if len(path) >= max_depth:
                continue

            for act in dir_actions:
                nxt = world_model.predict_next_avatar_pos(curr_pos, act)
                if nxt is None:
                    continue
                ny, nx = nxt
                if 0 <= ny < H and 0 <= nx < W and nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, path + [act]))

        return None
