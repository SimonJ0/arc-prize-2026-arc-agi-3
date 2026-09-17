"""
Synthetic micro-world test for sequential / delayed effects (Key-Door mechanics).
Verifies that:
1. Agent explores and picks up / triggers the key object.
2. Door opens upon trigger, permitting access to the terminal goal.
"""

import numpy as np
from arcengine import GameAction, GameState

from agent.my_agent import MyAgent


class KeyDoorEnv:
    """
    Micro-world with key and door:
    Avatar = color 2 at (4, 2)
    Key    = color 7 at (4, 5)
    Door   = color 6 at (4, 8) [Blocks path until key collected]
    Goal   = color 9 at (4, 11)
    """

    def __init__(self):
        self.game_id = "key_door"
        self.state = GameState.NOT_FINISHED
        self.levels_completed = 0
        self.available_actions = [0, 1, 2, 3, 4]  # RESET, A1-A4
        self.guid = "test-keydoor-guid"
        self.grid_size = 14
        self.avatar_pos = [4, 2]
        self.has_key = False
        self.step_count = 0

    @property
    def frame(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        grid[self.avatar_pos[0], self.avatar_pos[1]] = 2  # Avatar
        if not self.has_key:
            grid[4, 5] = 7  # Key
            grid[4, 8] = 6  # Locked door
        grid[4, 11] = 9  # Goal
        return [grid]

    def step(self, action: GameAction):
        self.step_count += 1
        y, x = self.avatar_pos

        # Semantics: A1=UP, A2=DOWN, A3=LEFT, A4=RIGHT
        if action == GameAction.ACTION4 and x < self.grid_size - 1:
            next_x = x + 1
            # Door block check
            if next_x == 8 and not self.has_key:
                return  # Blocked by door
            x = next_x
            if x == 5:
                self.has_key = True  # Pick up key
            elif x == 11:
                self.state = GameState.WIN
        elif action == GameAction.ACTION3 and x > 0:
            x -= 1
        elif action == GameAction.ACTION1 and y > 0:
            y -= 1
        elif action == GameAction.ACTION2 and y < self.grid_size - 1:
            y += 1

        self.avatar_pos = [y, x]


def test_agent_completes_delayed_key_door_puzzle():
    env = KeyDoorEnv()
    agent = MyAgent(game_id="keydoor_test")

    max_steps = 35
    for _step in range(1, max_steps + 1):
        if agent.is_done(env.frame, env):
            break
        act = agent.choose_action(env.frame, env)
        env.step(act)
        if env.state == GameState.WIN:
            break

    assert env.state == GameState.WIN
    assert env.has_key is True
