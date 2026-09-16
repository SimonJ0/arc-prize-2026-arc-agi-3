"""
Synthetic micro-world test for movement induction and goal pursuit.
Tests whether the agent can perceive its avatar, induce action mappings,
and reach a goal in an interactive 2D grid environment.
"""

import numpy as np
import pytest
from arcengine import GameAction, GameState

from agent.my_agent import MyAgent


class GridWorldEnv:
    """A deterministic 15x15 micro-world with cardinal movement and a target."""

    def __init__(self, avatar_start=(7, 7), goal_pos=(7, 11)):
        self.game_id = "microworld_nav"
        self.state = GameState.NOT_FINISHED
        self.levels_completed = 0
        self.available_actions = [0, 1, 2, 3, 4]
        self.guid = "test-micro-guid"
        self.avatar_pos = list(avatar_start)
        self.goal_pos = list(goal_pos)
        self.grid_size = 15
        self.step_count = 0

    @property
    def frame(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        grid[self.avatar_pos[0], self.avatar_pos[1]] = 3  # Avatar = color 3
        grid[self.goal_pos[0], self.goal_pos[1]] = 8      # Goal = color 8
        return [grid]

    def step(self, action: GameAction):
        self.step_count += 1
        y, x = self.avatar_pos

        # Semantics: ACTION1=UP, ACTION2=DOWN, ACTION3=LEFT, ACTION4=RIGHT
        if action == GameAction.ACTION1 and y > 0:
            y -= 1
        elif action == GameAction.ACTION2 and y < self.grid_size - 1:
            y += 1
        elif action == GameAction.ACTION3 and x > 0:
            x -= 1
        elif action == GameAction.ACTION4 and x < self.grid_size - 1:
            x += 1

        self.avatar_pos = [y, x]
        if self.avatar_pos == self.goal_pos:
            self.state = GameState.WIN


def test_agent_navigates_and_completes_microworld():
    env = GridWorldEnv(avatar_start=(7, 7), goal_pos=(7, 11))
    agent = MyAgent(game_id="microworld_test")

    max_steps = 30
    for step in range(1, max_steps + 1):
        if agent.is_done(env.frame, env):
            break
        act = agent.choose_action(env.frame, env)
        assert isinstance(act, GameAction)
        assert act.value in env.available_actions
        env.step(act)
        if env.state == GameState.WIN:
            break

    assert env.state == GameState.WIN
    assert env.step_count <= max_steps
