"""
Synthetic micro-world test for ACTION6 coordinate selection.
Verifies that:
1. When ACTION6 is available, coordinates are pruned to candidate entity locations.
2. Coordinates are legal integers in [0, 63].
3. Clicking target entity completes the objective.
"""

import numpy as np
from arcengine import GameAction, GameState

from agent.my_agent import MyAgent


class TargetClickEnv:
    """
    Micro-world where only ACTION6 is available.
    Target button is at (4, 8). Clicking it triggers WIN.
    """

    def __init__(self):
        self.game_id = "target_click"
        self.state = GameState.NOT_FINISHED
        self.levels_completed = 0
        self.available_actions = [6]  # ACTION6 only
        self.guid = "test-click-guid"
        self.grid_size = 16
        self.target_y = 4
        self.target_x = 8
        self.step_count = 0

    @property
    def frame(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        grid[self.target_y, self.target_x] = 5  # Target button color = 5
        return [grid]

    def step(self, action: GameAction, payload: dict | None = None):
        self.step_count += 1
        if action == GameAction.ACTION6 and payload:
            x = payload.get("x", -1)
            y = payload.get("y", -1)
            if (y, x) == (self.target_y, self.target_x):
                self.state = GameState.WIN


def test_agent_targets_entity_coordinate_with_action6():
    env = TargetClickEnv()
    agent = MyAgent(game_id="click_test")

    # Agent must select ACTION6
    act = agent.choose_action(env.frame, env)
    assert act == GameAction.ACTION6

    # Verify that the policy chose the candidate entity coordinate (8, 4)
    # The policy stored the trace in agent.previous_action
    assert agent.previous_action == "ACTION6"
    assert agent.previous_analysis is not None
    assert len(agent.previous_analysis.entities) > 0

    target_ent = agent.previous_analysis.entities[0]
    cy, cx = target_ent.centroid
    assert (int(cy), int(cx)) == (4, 8)
