"""
Unit tests for DRE-Bench Cognitive Hierarchy and OpenAI Reasoning Persistence & Context Compaction.
"""

import numpy as np

from src.arc_agent.memory.reasoning_state import (
    ContextCompactor,
    PersistentReasoningState,
    StepSummary,
)
from src.arc_agent.perception.cognitive_hierarchy import (
    CognitiveHierarchyPerception,
)
from src.arc_agent.perception.layered_perception import LayeredPerception
from src.arc_agent.planning.epistemic_policy import EpistemicPolicy
from src.arc_agent.world_model.belief_state import BeliefStateWorldModel
from src.arc_core.contracts import Observation


def test_dre_bench_attribute_and_spatial_levels():
    """Verify Level 1 Attribute and Level 2 Spatial analysis."""
    # Create 16x16 grid with background=0, horizontal wall=1, player=2, goal=3
    grid = np.zeros((16, 16), dtype=int)
    grid[5, 2:14] = 1  # Wall (size 12)
    grid[2, 2] = 2  # Player singleton (size 1)
    grid[10, 10] = 3  # Goal singleton (size 1)

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


def test_surprise_detection_invalidates_plan():
    """Verify that unexpected position displacement immediately invalidates active macro-plan."""
    reasoning_state = PersistentReasoningState(game_id="surprise_test")
    reasoning_state.set_macro_plan(["ACTION1", "ACTION2"], goal=(5, 5))
    reasoning_state.last_predicted_pos = (2, 3)

    assert reasoning_state.has_active_plan()

    # Actual position lands at (2, 2) instead of (2, 3) -> surprise!
    surprise = reasoning_state.check_and_handle_surprise(actual_pos=(2, 2))
    assert surprise is True
    assert not reasoning_state.has_active_plan()
    assert reasoning_state.last_predicted_pos is None


def test_loop_detection_and_visitation_tracking():
    """Verify visitation counting detects oscillations and triggers loop flag."""
    reasoning_state = PersistentReasoningState(game_id="loop_test")
    pos = (4, 4)

    assert not reasoning_state.is_loop_detected(pos)
    assert reasoning_state.record_visitation(pos) == 1
    assert reasoning_state.record_visitation(pos) == 2
    assert not reasoning_state.is_loop_detected(pos)

    assert reasoning_state.record_visitation(pos) == 3
    assert reasoning_state.is_loop_detected(pos)


def test_falsifiable_goals():
    """Verify goal falsification clears active plan and registers coordinate."""
    reasoning_state = PersistentReasoningState(game_id="goal_test")
    goal = (7, 7)
    reasoning_state.set_macro_plan(["ACTION1"], goal=goal)

    assert goal not in reasoning_state.falsified_goals
    reasoning_state.falsify_goal(goal)
    assert goal in reasoning_state.falsified_goals
    assert not reasoning_state.has_active_plan()


def test_empirical_dynamics_learning():
    """Verify empirical displacement tracking and 1-step verification."""
    world_model = BeliefStateWorldModel()

    # Initially unverified
    assert not world_model.is_action_verified("ACTION1")

    # Record two consistent displacements: dy=-1, dx=0
    world_model._record_empirical_displacement("ACTION1", -1.0, 0.0)
    world_model._record_empirical_displacement("ACTION1", -1.0, 0.0)

    assert world_model.is_action_verified("ACTION1")
    assert world_model.get_action_displacement("ACTION1") == (-1, 0)


def test_multidirectional_astar_preserves_unlearned_actions():
    """Verify A* search does not collapse to 1D when only a single action delta is learned."""
    policy = EpistemicPolicy()
    reasoning_state = PersistentReasoningState(game_id="astar_2d_test")
    # Only ACTION2 is empirically observed
    reasoning_state.update_action_effect("ACTION2", dy=1, dx=0)

    world_model = BeliefStateWorldModel()
    start = (10, 10)
    # Goal requires moving UP (dy=-5) and RIGHT (dx=+5)
    goal = (5, 15)

    path = policy._astar_search(
        start=start,
        goal=goal,
        grid_shape=(30, 30),
        obstacles=set(),
        world_model=world_model,
        available_actions={"ACTION1", "ACTION2", "ACTION3", "ACTION4"},
        reasoning_state=reasoning_state,
    )
    assert path is not None, "A* failed to find 2D path when only 1 action was empirically known!"
    assert "ACTION1" in path or "ACTION4" in path, (
        "A* only used learned action, collapsing search to 1D!"
    )


def test_walkable_surface_not_marked_as_static_obstacle():
    """Verify large floor/corridor surfaces (size >= 40) are not classified as obstacles."""
    perception = CognitiveHierarchyPerception()
    frame = np.zeros((30, 30), dtype=int)
    # Background color 0
    # Walkable floor of color 2 occupying 100 pixels in center
    frame[5:15, 5:15] = 2

    analysis = perception.analyze(frame=frame, known_player_color=2)
    assert analysis.player_pos is not None
    # Verify coordinates of color 2 floor are NOT in static_obstacles
    assert (10, 10) not in analysis.static_obstacles
    assert len(analysis.static_obstacles) == 0


def test_dynamic_avatar_detection_from_multi_pixel_motion_diff():
    """Verify multi-pixel moving entity (>4 pixels) is correctly identified as player avatar."""
    perception = CognitiveHierarchyPerception()
    f0 = np.zeros((30, 30), dtype=int)
    # 9-pixel sprite of color 3 at (10:13, 10:13)
    f0[10:13, 10:13] = 3

    f1 = np.zeros((30, 30), dtype=int)
    # Moves to (12:15, 10:13)
    f1[12:15, 10:13] = 3

    analysis = perception.analyze(frame=f1, prev_frame=f0)
    assert analysis.player_color == 3
    assert analysis.player_pos is not None
    # Centroid of (12:15, 10:13) is (13, 11)
    assert abs(analysis.player_pos[0] - 13) <= 1
    assert abs(analysis.player_pos[1] - 11) <= 1
