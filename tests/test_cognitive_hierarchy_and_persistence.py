"""
Unit tests for DRE-Bench Cognitive Hierarchy and OpenAI Reasoning Persistence & Context Compaction.
"""

from collections import deque
import numpy as np
import pytest

from src.arc_agent.perception.cognitive_hierarchy import (
    CognitiveHierarchyPerception,
    CognitiveHierarchyAnalysis,
)
from src.arc_agent.memory.reasoning_state import (
    PersistentReasoningState,
    ContextCompactor,
    StepSummary,
)
from src.arc_agent.planning.epistemic_policy import EpistemicPolicy
from src.arc_core.contracts import Observation
from src.arc_agent.perception.layered_perception import LayeredPerception
from src.arc_agent.world_model.belief_state import BeliefStateWorldModel


def test_dre_bench_attribute_and_spatial_levels():
    """Verify Level 1 Attribute and Level 2 Spatial analysis."""
    # Create 16x16 grid with background=0, horizontal wall=1, player=2, goal=3
    grid = np.zeros((16, 16), dtype=int)
    grid[5, 2:14] = 1   # Wall (size 12)
    grid[2, 2] = 2      # Player singleton (size 1)
    grid[10, 10] = 3    # Goal singleton (size 1)

    perception = CognitiveHierarchyPerception()
    analysis = perception.analyze(grid)

    assert analysis.dominant_color == 0
    assert len(analysis.attributes) == 3
    # Check singleton detection
    singletons = [p for p in analysis.singleton_entities]
    assert len(singletons) >= 2

    # Check symmetry scores exist
    assert 0.0 <= analysis.symmetry.horizontal <= 1.0
    assert 0.0 <= analysis.symmetry.vertical <= 1.0


def test_dre_bench_sequential_macro_planning():
    """Verify Level 3 Sequential planning paths around obstacles."""
    policy = EpistemicPolicy()
    world_model = BeliefStateWorldModel()
    reasoning_state = PersistentReasoningState(game_id="test_game")

    start = (2, 2)
    goal = (4, 2)
    # Put a barrier directly between start and goal at (3, 2)
    obstacles = {(3, 2)}
    available = {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}

    path = policy._astar_search(
        start=start,
        goal=goal,
        grid_shape=(10, 10),
        obstacles=obstacles,
        world_model=world_model,
        available_actions=available,
        reasoning_state=reasoning_state,
    )

    assert path is not None
    assert len(path) >= 3, "Path must route around the obstacle at (3, 2)"
    # Path should not just step straight down (ACTION2) directly into the obstacle
    assert not (len(path) == 2 and path == ["ACTION2", "ACTION2"])


def test_openai_reasoning_persistence_execution():
    """Verify multi-step macro-plans persist and execute across turns."""
    policy = EpistemicPolicy()
    world_model = BeliefStateWorldModel()
    reasoning_state = PersistentReasoningState(game_id="test_game")

    # Queue an active macro-plan
    reasoning_state.set_macro_plan(["ACTION1", "ACTION4", "ACTION4"], goal=(0, 5))
    assert reasoning_state.has_active_plan()

    obs = Observation(
        frames=(np.zeros((10, 10), dtype=int),),
        state="NOT_FINISHED",
        available_actions=frozenset(["ACTION1", "ACTION2", "ACTION3", "ACTION4"]),
        game_key="test_game",
        level=1,
        action_count=1,
    )
    analysis = LayeredPerception().analyze(obs.frames[0])

    # Turn 1: Should pop ACTION1 and report PERSISTENT_MACRO_PLAN
    act1, _, trace1 = policy.select_action(
        observation=obs,
        analysis=analysis,
        world_model=world_model,
        reasoning_state=reasoning_state,
    )
    assert act1 == "ACTION1"
    assert trace1.planning_mode == "PERSISTENT_MACRO_PLAN"
    assert len(reasoning_state.active_macro_plan) == 2

    # Turn 2: Should pop ACTION4
    act2, _, trace2 = policy.select_action(
        observation=obs,
        analysis=analysis,
        world_model=world_model,
        reasoning_state=reasoning_state,
    )
    assert act2 == "ACTION4"
    assert trace2.planning_mode == "PERSISTENT_MACRO_PLAN"
    assert len(reasoning_state.active_macro_plan) == 1


def test_openai_death_avoidance_learning():
    """Verify GAME_OVER triggers lethal coordinate learning and avoidance."""
    policy = EpistemicPolicy()
    world_model = BeliefStateWorldModel()
    reasoning_state = PersistentReasoningState(game_id="test_game")

    fatal_coord = (3, 3)
    reasoning_state.record_death(fatal_coord, None)
    assert fatal_coord in reasoning_state.death_coords

    # Path from (2, 3) to (4, 3) with fatal_coord (3, 3) in obstacles
    path = policy._astar_search(
        start=(2, 3),
        goal=(4, 3),
        grid_shape=(10, 10),
        obstacles=reasoning_state.death_coords,
        world_model=world_model,
        available_actions={"ACTION1", "ACTION2", "ACTION3", "ACTION4"},
        reasoning_state=reasoning_state,
    )
    assert path is not None
    # Path must avoid (3, 3)
    curr = (2, 3)
    deltas = {"ACTION1": (-1, 0), "ACTION2": (1, 0), "ACTION3": (0, -1), "ACTION4": (0, 1)}
    for act in path:
        dy, dx = deltas[act]
        curr = (curr[0] + dy, curr[1] + dx)
        assert curr != fatal_coord, "Planner stepped into learned lethal coordinate!"


def test_context_compactor():
    """Verify ContextCompactor creates dense summaries without large frame arrays."""
    compactor = ContextCompactor(max_rolling_steps=5)

    for step in range(1, 8):
        summary = compactor.compact_step(
            step=step,
            level=1,
            action="ACTION1",
            curr_pos=(step, step),
            prev_pos=(step - 1, step - 1),
            state="NOT_FINISHED",
            levels_completed=0,
        )
        assert isinstance(summary, StepSummary)
        assert summary.delta_pos == (1, 1)

    traj = compactor.get_trajectory_summary()
    assert traj["total_steps"] == 7
    # Rolling history capped at 5
    assert len(traj["recent_trajectory"]) == 5
