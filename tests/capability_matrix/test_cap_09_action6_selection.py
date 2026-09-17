"""
Capability Diagnostic 9: Coordinate Selection & Target Pruning (ACTION6).
Verifies that when ACTION6 is chosen, payload coordinates target candidate entities.
"""

import numpy as np
from arcengine import GameAction, GameState
from agent.my_agent import MyAgent
from tests.microworlds.test_coordinate_selection import TargetClickEnv


def test_coordinate_selection_targeting():
    env = TargetClickEnv()
    agent = MyAgent(game_id="cap_action6")

    act = agent.choose_action(env.frame, env)
    assert act == GameAction.ACTION6
    payload = getattr(agent, "last_payload", None)
    assert payload is not None
    assert (payload.get("y"), payload.get("x")) == (4, 8)


def test_inert_affordance_pruning():
    class TwoButtonEnv:
        def __init__(self):
            self.game_id = "two_button"
            self.state = GameState.NOT_FINISHED
            self.levels_completed = 0
            self.available_actions = [6]
            self.guid = "test-two-button"
            self.grid = np.zeros((20, 20), dtype=int)
            self.grid[4, 4] = 3  # Button 1 (inert, clicking changes nothing)
            self.grid[12, 12] = 5  # Button 2 (active, clicking changes pixel)

        @property
        def frame(self):
            return [self.grid]

        def step(self, action, payload=None):
            if payload and payload.get("y") == 12 and payload.get("x") == 12:
                self.grid[12, 12] = 6
            return self

    env = TwoButtonEnv()
    agent = MyAgent(game_id="cap_action6_pruning")

    act1 = agent.choose_action(env.frame, env)
    payload1 = getattr(agent, "last_payload", None)
    assert act1 == GameAction.ACTION6
    assert payload1 is not None

    env.step(act1, payload1)

    act2 = agent.choose_action(env.frame, env)
    payload2 = getattr(agent, "last_payload", None)
    assert act2 == GameAction.ACTION6
    assert payload2 is not None

    first_coord = (payload1.get("y"), payload1.get("x"))
    second_coord = (payload2.get("y"), payload2.get("x"))
    assert first_coord != second_coord, "Agent should not repeat inert click on step 2"


def test_massive_obstacle_filtering():
    class ObstacleClickEnv:
        def __init__(self):
            self.game_id = "obstacle_env"
            self.state = GameState.NOT_FINISHED
            self.levels_completed = 0
            self.available_actions = [6]
            self.guid = "test-obs-click"
            self.grid = np.zeros((30, 30), dtype=int)
            # Create a massive obstacle entity (15x15 = 225 pixels > 120)
            self.grid[5:20, 5:20] = 7
            # Small interactive button (2x2 = 4 pixels)
            self.grid[25:27, 25:27] = 2

        @property
        def frame(self):
            return [self.grid]

        def step(self, action, payload=None):
            return self

    env = ObstacleClickEnv()
    agent = MyAgent(game_id="cap_obs_filter")

    act = agent.choose_action(env.frame, env)
    assert act == GameAction.ACTION6
    payload = getattr(agent, "last_payload", None)
    assert payload is not None

    py, px = payload.get("y"), payload.get("x")
    assert (py, px) in [(25, 25), (25, 26), (26, 25), (26, 26)]
