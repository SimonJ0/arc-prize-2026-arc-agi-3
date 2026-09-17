"""
Uncertainty-Aware Epistemic Policy for ARC-AGI-3.
Integrates:
1. OpenAI Reasoning Persistence: Executes active multi-step macro-plans across turns.
2. DRE-Bench Cognitive Hierarchy: Targets goal candidates and avoids static obstacles / death coordinates.
3. Bayesian Epistemic Probing: Explores untested actions to deduce movement dynamics when uncertain.
4. Hard Legality Validation: Via LegalityAdapter.
"""

from collections import deque
from collections.abc import Collection
from typing import Any

import numpy as np

from src.arc_agent.legality_adapter import LegalityAdapter
from src.arc_agent.memory.reasoning_state import PersistentReasoningState
from src.arc_agent.perception.cognitive_hierarchy import CognitiveHierarchyAnalysis
from src.arc_agent.perception.layered_perception import FrameAnalysis
from src.arc_agent.reasoning.action_algebra import ActionAlgebraEngine
from src.arc_agent.reasoning.model_gate import ModelGate, ModelUseDecision
from src.arc_agent.world_model.belief_state import BeliefStateWorldModel
from src.arc_core.contracts import DecisionTrace, Observation


class EpistemicPolicy:
    """Decision engine balancing active epistemic learning with persistent macro-planning."""

    def __init__(self):
        self.step_counter = 0
        self.algebra = ActionAlgebraEngine()
        self.model_gate = ModelGate()
        self.recent_coords: deque[tuple[int, int]] = deque(maxlen=6)

    def select_action(
        self,
        observation: Observation,
        analysis: FrameAnalysis,
        world_model: BeliefStateWorldModel,
        reasoning_state: PersistentReasoningState | None = None,
        cognitive_analysis: CognitiveHierarchyAnalysis | None = None,
        structured_belief: Any | None = None,
    ) -> tuple[str, dict[str, Any], DecisionTrace]:
        """
        Selects next physical environment action given current belief state.
        Guaranteed to return a legal action from observation.available_actions.
        """
        self.step_counter += 1
        state = observation.state
        available = observation.available_actions
        p_pos = cognitive_analysis.player_pos if cognitive_analysis else None

        # 1. State-level guard: GAME_OVER requires RESET
        if state == "GAME_OVER":
            if reasoning_state is not None:
                # Learn fatal location to avoid it in future runs
                reasoning_state.record_death(p_pos, None)

            action, payload = LegalityAdapter.validate_action(
                state=state,
                available_actions=available,
                proposed_action="RESET",
            )
            trace = DecisionTrace(
                level=observation.level,
                step=self.step_counter,
                observation_hash=str(hash(observation.frames[0].tobytes())),
                legal_actions=tuple(sorted(available)),
                selected_action=action,
                selected_payload=payload,
                planning_mode="RESET_RECOVERY",
                predicted_next_state="NOT_FINISHED",
                confidence=1.0,
            )
            return action, payload, trace

        # Surprise & Deadlock Loop Management
        is_loop = False
        if reasoning_state is not None and p_pos is not None:
            # 1b. Surprise Invalidation: Did last action land where we expected?
            reasoning_state.check_and_handle_surprise(p_pos)

            # 1c. Loop / Deadlock Breaker
            v_count = reasoning_state.record_visitation(p_pos)
            if v_count >= 3:
                is_loop = True

            # 1d. Falsify unrewarded goals: If player reached active goal or looped without level advancing
            if reasoning_state.active_goal_coord is not None:
                g = reasoning_state.active_goal_coord
                if not reasoning_state.has_active_plan() and p_pos == g:
                    reasoning_state.falsify_goal(g)
                elif is_loop:
                    reasoning_state.falsify_goal(g)

        # Model-Use Gate: Determine execution strategy (direct vs simulation vs probe)
        has_plan = bool(reasoning_state and reasoning_state.has_active_plan())
        can_plan = world_model.can_reliably_plan()
        has_exploit = False
        if structured_belief is not None and hasattr(structured_belief, "mechanisms"):
            has_exploit = len(structured_belief.mechanisms.get_exploitable_controllers()) > 0

        model_decision = self.model_gate.evaluate_decision(
            has_active_plan=has_plan,
            can_reliably_plan=can_plan,
            has_exploitable_controller=has_exploit,
            is_loop=is_loop,
        )

        # 2. Reasoning Persistence: Check if there is an active macro-plan in flight (and not in deadlock loop)
        if reasoning_state is not None and reasoning_state.has_active_plan() and not is_loop:
            planned_action = reasoning_state.next_planned_action(available)
            if planned_action is not None:
                action, payload = LegalityAdapter.validate_action(
                    state=state,
                    available_actions=available,
                    proposed_action=planned_action,
                )
                if p_pos is not None:
                    reasoning_state.last_predicted_pos = world_model.predict_next_avatar_pos(
                        p_pos, action
                    )
                trace = DecisionTrace(
                    level=observation.level,
                    step=self.step_counter,
                    observation_hash=str(hash(observation.frames[0].tobytes())),
                    legal_actions=tuple(sorted(available)),
                    selected_action=action,
                    selected_payload=payload,
                    planning_mode="PERSISTENT_MACRO_PLAN",
                    predicted_next_state="NOT_FINISHED",
                    confidence=0.95,
                )
                return action, payload, trace

        # 3. DRE-Bench Sequential Planning: Macro-path to candidate goals
        if cognitive_analysis is not None and p_pos is not None and not is_loop:
            all_goals = cognitive_analysis.candidate_goals
            # Filter out falsified goals for this level
            if reasoning_state is not None and reasoning_state.falsified_goals:
                candidate_goals = [g for g in all_goals if g not in reasoning_state.falsified_goals]
            else:
                candidate_goals = all_goals

            obstacles = set(cognitive_analysis.static_obstacles)
            if reasoning_state is not None:
                obstacles.update(reasoning_state.death_coords)

            frame_shape = observation.frames[0].shape
            for goal in candidate_goals[:3]:  # Evaluate top non-falsified candidate goals
                path = self._astar_search(
                    start=p_pos,
                    goal=goal,
                    grid_shape=frame_shape,
                    obstacles=obstacles,
                    world_model=world_model,
                    available_actions=available,
                    reasoning_state=reasoning_state,
                )
                if path:
                    chosen = path[0]
                    if reasoning_state is not None and len(path) > 1:
                        # Retain remainder of plan in persistent memory
                        reasoning_state.set_macro_plan(path[1:], goal=goal)

                    action, payload = LegalityAdapter.validate_action(
                        state=state,
                        available_actions=available,
                        proposed_action=chosen,
                    )
                    if reasoning_state is not None:
                        reasoning_state.last_predicted_pos = world_model.predict_next_avatar_pos(
                            p_pos, action
                        )
                    trace = DecisionTrace(
                        level=observation.level,
                        step=self.step_counter,
                        observation_hash=str(hash(observation.frames[0].tobytes())),
                        legal_actions=tuple(sorted(available)),
                        selected_action=action,
                        selected_payload=payload,
                        planning_mode="MACRO_GOAL_PLAN",
                        predicted_next_state="NOT_FINISHED",
                        confidence=0.9,
                    )
                    return action, payload, trace

        # 4. Exploitation Mode: Validated model allows forward planning (if not in loop)
        if world_model.can_reliably_plan() and not is_loop:
            action, payload, trace = self._plan_goal_trajectory(observation, analysis, world_model)
            if reasoning_state is not None and p_pos is not None:
                reasoning_state.last_predicted_pos = world_model.predict_next_avatar_pos(
                    p_pos, action
                )
            return action, payload, trace

        # 5. Epistemic Probing Mode / Deadlock Breaker
        action, payload, trace = self._select_epistemic_probe(
            observation, analysis, world_model, reasoning_state, is_loop, structured_belief
        )
        if reasoning_state is not None and p_pos is not None:
            reasoning_state.last_predicted_pos = world_model.predict_next_avatar_pos(p_pos, action)
        return action, payload, trace

    def _select_epistemic_probe(
        self,
        observation: Observation,
        analysis: FrameAnalysis,
        world_model: BeliefStateWorldModel,
        reasoning_state: PersistentReasoningState | None = None,
        is_loop: bool = False,
        structured_belief: Any | None = None,
    ) -> tuple[str, dict[str, Any], DecisionTrace]:
        """Selects informative probe action to distinguish candidate transition models or break deadlocks."""
        available = list(observation.available_actions)

        # Candidate probe actions: evaluate all available non-reset actions
        candidate_probes = [a for a in available if a != "RESET"]
        if not candidate_probes:
            candidate_probes = list(available)

        # Prioritize untested actions in world_model.action_stats
        untested = [
            a
            for a in candidate_probes
            if a not in getattr(world_model, "action_stats", {})
            or world_model.action_stats[a]["attempts"] == 0
        ]
        if untested:
            selected = untested[self.step_counter % len(untested)]
        else:
            # Sort by least attempts
            candidate_probes.sort(
                key=lambda a: getattr(world_model, "action_stats", {}).get(a, {}).get("attempts", 0)
            )
            selected = candidate_probes[0]

        payload = {}
        if selected == "ACTION6":
            # Update action algebra from mechanism transition ledger if available
            if structured_belief is not None and hasattr(structured_belief, "mechanisms"):
                self.algebra.update_from_records(structured_belief.mechanisms.records)

            # Affordance-driven coordinate selection
            last_coord = getattr(reasoning_state, "last_affordance_coord", None) if reasoning_state else None
            last_diff = getattr(reasoning_state, "last_affordance_diff", 0) if reasoning_state else 0
            repeat_count = getattr(reasoning_state, "affordance_repeat_count", 0) if reasoning_state else 0
            lethal_set = getattr(reasoning_state, "lethal_affordances", set()) if reasoning_state else set()
            inert_set = getattr(reasoning_state, "inert_affordances", set()) if reasoning_state else set()
            active_list = getattr(reasoning_state, "active_affordances", []) if reasoning_state else []

            # 1. Filter entities: compact play entities (avoiding huge obstacle blobs and letterbox/UI bounds)
            cands = []
            if analysis.entities:
                for e in analysis.entities:
                    cy, cx = int(round(e.centroid[0])), int(round(e.centroid[1]))
                    # Discard letterbox border pixels and top UI indicator rows (cy <= 1)
                    if not (2 <= cy <= 61 and 2 <= cx <= 61):
                        continue
                    if e.size > 120:  # Massive entities are obstacles or static backgrounds
                        continue
                    if (cy, cx) in lethal_set:
                        continue
                    cands.append(e)

            chosen_coord: tuple[int, int] | None = None

            # 2. Decision logic:
            # Check mechanism confidence vs goal relevance before repeating active controller
            mech_permits_exploit = True
            if structured_belief is not None and hasattr(structured_belief, "mechanisms") and last_coord:
                hyp = structured_belief.mechanisms.get_hypothesis(last_coord)
                if hyp is not None and not hyp.can_reliably_exploit():
                    mech_permits_exploit = False

            if last_coord is not None and last_diff > 2 and repeat_count < 5 and last_coord not in lethal_set and mech_permits_exploit:
                chosen_coord = last_coord
            elif cands:
                # Prioritize candidates not confirmed inert by coordinate
                non_inert = [
                    e
                    for e in cands
                    if (int(round(e.centroid[0])), int(round(e.centroid[1]))) not in inert_set
                ]
                # B3.04: Causal equivalence class filtering (prune entire classes of inert entities)
                if structured_belief is not None and hasattr(structured_belief, "mechanisms"):
                    non_inert = structured_belief.mechanisms.filter_inert_classes(non_inert)

                pool = non_inert if non_inert else cands
                # Filter out oscillating inverse actions
                coords_pool = [(int(round(e.centroid[0])), int(round(e.centroid[1]))) for e in pool]
                pruned_coords = self.algebra.filter_oscillating_actions(coords_pool, list(self.recent_coords))
                cands_to_score = [
                    e
                    for e in pool
                    if (int(round(e.centroid[0])), int(round(e.centroid[1]))) in set(pruned_coords)
                ] or pool

                # B3.11: RHAE-aware utility ranking U(a) = αG + βI + γC - λK - μR
                cand_scores = [
                    (
                        e,
                        self._compute_action_utility(
                            coord=(int(round(e.centroid[0])), int(round(e.centroid[1]))),
                            entity=e,
                            structured_belief=structured_belief,
                            reasoning_state=reasoning_state,
                        ),
                    )
                    for e in cands_to_score
                ]
                max_u = max(s[1] for s in cand_scores)
                top_cands = [s[0] for s in cand_scores if abs(s[1] - max_u) < 0.05]
                best_ent = top_cands[self.step_counter % len(top_cands)]
                chosen_coord = (int(round(best_ent.centroid[0])), int(round(best_ent.centroid[1])))
            elif active_list:
                pruned_active = self.algebra.filter_oscillating_actions(active_list, list(self.recent_coords))
                chosen_coord = (pruned_active or active_list)[self.step_counter % len(pruned_active or active_list)]
            else:
                H, W = analysis.frame_shape
                chosen_coord = (int(np.clip(H // 2, 0, 63)), int(np.clip(W // 2, 0, 63)))

            if chosen_coord is not None:
                self.recent_coords.append(chosen_coord)

            payload = {
                "x": int(np.clip(chosen_coord[1], 0, 63)),
                "y": int(np.clip(chosen_coord[0], 0, 63)),
            }

        action, valid_payload = LegalityAdapter.validate_action(
            state=observation.state,
            available_actions=observation.available_actions,
            proposed_action=selected,
            proposed_payload=payload,
        )

        trace = DecisionTrace(
            level=observation.level,
            step=self.step_counter,
            observation_hash=str(hash(observation.frames[0].tobytes())),
            legal_actions=tuple(sorted(observation.available_actions)),
            selected_action=action,
            selected_payload=valid_payload,
            planning_mode="DEADLOCK_BREAKER" if is_loop else "EPISTEMIC_PROBE",
            predicted_next_state="NOT_FINISHED",
            confidence=0.5,
        )
        return action, valid_payload, trace

    def _compute_action_utility(
        self,
        coord: tuple[int, int],
        entity: Any,
        structured_belief: Any | None,
        reasoning_state: PersistentReasoningState | None,
    ) -> float:
        """
        Computes action utility: U(a) = α*G(a) + β*I(a) + γ*C(a) - λ*K(a) - μ*R(a)
        Maximizes goal progress and information gain while penalizing action cost and risk.
        """
        G = 0.5
        C = 0.5
        I = 1.0
        R = 0.0
        K = 1.0

        if structured_belief is not None and hasattr(structured_belief, "mechanisms"):
            hyp = structured_belief.mechanisms.get_hypothesis(coord)
            if hyp is not None:
                G = hyp.goal_relevance
                C = hyp.causal_confidence
                I = 1.0 / (hyp.observations + 1.0)
                if hyp.is_lethal:
                    R = 1.0
            else:
                I = 1.0

        if reasoning_state is not None and coord in reasoning_state.death_coords:
            R = 1.0

        # Weights: α=3.5 (goal progress), β=1.0 (information gain), γ=1.0 (causal confidence), λ=0.5 (action cost), μ=5.0 (risk)
        return 3.5 * G + 1.0 * I + 1.0 * C - 0.5 * K - 5.0 * R

    def _plan_goal_trajectory(
        self,
        observation: Observation,
        analysis: FrameAnalysis,
        world_model: BeliefStateWorldModel,
    ) -> tuple[str, dict[str, Any], DecisionTrace]:
        """Plans shortest path to candidate goal under validated transition dynamics."""
        frame = observation.frames[0]
        avatar_color = world_model.avatar_color
        available = observation.available_actions

        avatar_coords = np.argwhere(frame == avatar_color)
        if len(avatar_coords) == 0:
            return self._select_epistemic_probe(observation, analysis, world_model)

        start_pos = (int(avatar_coords[0][0]), int(avatar_coords[0][1]))

        target_pos = None
        for ent in analysis.entities:
            if ent.color != avatar_color and ent.size <= 36:
                target_pos = (int(ent.centroid[0]), int(ent.centroid[1]))
                break

        if target_pos is None:
            return self._select_epistemic_probe(observation, analysis, world_model)

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
                legal_actions=tuple(sorted(available)),
                selected_action=action,
                selected_payload=valid_payload,
                planning_mode="GOAL_PLAN",
                predicted_next_state="NOT_FINISHED",
                confidence=0.9,
            )
            return action, valid_payload, trace

        return self._select_epistemic_probe(observation, analysis, world_model)

    def _astar_search(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        grid_shape: tuple[int, int],
        obstacles: set[tuple[int, int]],
        world_model: BeliefStateWorldModel,
        available_actions: Collection[str],
        reasoning_state: PersistentReasoningState | None = None,
    ) -> list[str] | None:
        """A* search towards goal avoiding static obstacles and lethal death coordinates."""
        H, W = grid_shape
        import heapq

        default_deltas: dict[str, tuple[int, int]] = {
            "ACTION1": (-1, 0),  # UP
            "ACTION2": (1, 0),  # DOWN
            "ACTION3": (0, -1),  # LEFT
            "ACTION4": (0, 1),  # RIGHT
        }

        # Build action displacement mapping from empirical learning and world model
        action_deltas: dict[str, tuple[int, int]] = {}
        known_step_sizes = []
        for act in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"):
            if act in available_actions:
                disp = None
                if reasoning_state is not None and act in reasoning_state.action_effects:
                    disp = reasoning_state.action_effects[act]
                elif world_model is not None:
                    disp = world_model.get_action_displacement(act)
                if disp is not None and disp != (0, 0):
                    action_deltas[act] = disp
                    known_step_sizes.append(max(abs(disp[0]), abs(disp[1])))

        base_step = int(np.median(known_step_sizes)) if known_step_sizes else 1

        # Populate cardinal fallback per available action so search space never collapses to 1D
        for act, (dy, dx) in default_deltas.items():
            if act in available_actions and act not in action_deltas:
                action_deltas[act] = (dy * base_step, dx * base_step)

        valid_actions = [act for act in action_deltas if act in available_actions]
        if not valid_actions:
            return None

        # Ensure start and goal coordinates are not blocked by obstacle mask
        nav_obstacles = set(obstacles) - {start, goal}

        # Priority queue stores (f_score, cost, current_pos, path)
        def h(pos: tuple[int, int]) -> int:
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

        heap: list[tuple[int, int, tuple[int, int], list[str]]] = [(h(start), 0, start, [])]
        visited = {start: 0}
        max_expansions = 200  # Bound computation
        goal_tolerance = max(1, base_step)

        best_path = None
        best_dist = float("inf")

        while heap and max_expansions > 0:
            max_expansions -= 1
            f, cost, curr, path = heapq.heappop(heap)

            if curr == goal:
                return path

            dist = h(curr)
            if path and dist < best_dist:
                best_dist = dist
                best_path = path

            for act in valid_actions:
                dy, dx = action_deltas[act]
                ny, nx = curr[0] + dy, curr[1] + dx

                if not (0 <= ny < H and 0 <= nx < W):
                    continue
                nxt = (ny, nx)
                if nxt in nav_obstacles:
                    continue

                # Add penalty for highly-visited positions to discourage looping
                extra_cost = 0
                if reasoning_state and nxt in reasoning_state.visitation_counts:
                    extra_cost = reasoning_state.visitation_counts[nxt] * 2

                # Prefer verified actions
                if world_model and not world_model.is_action_verified(act):
                    extra_cost += 1

                new_cost = cost + 1 + extra_cost
                if nxt not in visited or new_cost < visited[nxt]:
                    visited[nxt] = new_cost
                    heapq.heappush(heap, (new_cost + h(nxt), new_cost, nxt, path + [act]))

        if best_path is not None and best_dist <= goal_tolerance:
            return best_path

        return None

    def _bfs_search(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        grid_shape: tuple[int, int],
        world_model: BeliefStateWorldModel,
        available_actions: Collection[str],
    ) -> list[str] | None:
        """Bounded BFS search over validated directional dynamics."""
        H, W = grid_shape
        queue: deque[tuple[tuple[int, int], list[str]]] = deque([(start, [])])
        visited = {start}

        dir_actions = [
            a for a in ("ACTION1", "ACTION2", "ACTION3", "ACTION4") if a in available_actions
        ]
        max_depth = 40

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
