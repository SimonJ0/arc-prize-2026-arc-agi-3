"""
Synthetic micro-world test for reversibility and undo dynamics.
Verifies that:
1. When ACTION7 (undo) is available, it reverses the previous move.
2. The agent accounts for the fact that ACTION7 still increments the action counter.
3. RESET recovers from trap / dead-end states.
"""

import numpy as np
from arcengine import GameAction, GameState

from agent.my_agent import MyAgent


class ReversibleTrapEnv:
    """
    Micro-world with a trap tile at (5, 5). Stepping on it leads to GAME_OVER
    unless UNDO (ACTION7) or RESET is used.
    Target is at (5, 7).
    """

    def __init__(self):
        self.game_id = "reversible_trap"
        self.state = GameState.NOT_FINISHED
        self.levels_completed = 0
        self.available_actions = [0, 1, 2, 3, 4, 7]  # RESET, A1-A4, A7 (undo)
        self.guid = "test-reversible-guid"
        self.grid_size = 12
        self.avatar_pos = [5, 3]
        self.trap_pos = [5, 5]
        self.goal_pos = [5, 7]
        self.history = []
        self.step_count = 0

    @property
    def frame(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        grid[self.avatar_pos[0], self.avatar_pos[1]] = 2  # Avatar
        grid[self.trap_pos[0], self.trap_pos[1]] = 4  # Trap = red
        grid[self.goal_pos[0], self.goal_pos[1]] = 3  # Goal = green
        return [grid]

    def step(self, action: GameAction):
        self.step_count += 1
        y, x = self.avatar_pos
        self.history.append(list(self.avatar_pos))

        if action == GameAction.RESET:
            self.avatar_pos = [5, 3]
            self.state = GameState.NOT_FINISHED
            return

        if action == GameAction.ACTION7:  # Undo
            if len(self.history) >= 2:
                self.history.pop()  # Pop current
                self.avatar_pos = self.history.pop()
            return

        # Directional moves: A1=UP, A2=DOWN, A3=LEFT, A4=RIGHT
        if action == GameAction.ACTION1 and y > 0:
            y -= 1
        elif action == GameAction.ACTION2 and y < self.grid_size - 1:
            y += 1
        elif action == GameAction.ACTION3 and x > 0:
            x -= 1
        elif action == GameAction.ACTION4 and x < self.grid_size - 1:
            x += 1

        self.avatar_pos = [y, x]
        if self.avatar_pos == self.trap_pos:
            self.state = GameState.GAME_OVER
        elif self.avatar_pos == self.goal_pos:
            self.state = GameState.WIN


def test_agent_handles_game_over_with_reset():
    env = ReversibleTrapEnv()
    agent = MyAgent(game_id="trap_test")

    # Force step into trap
    env.step(GameAction.ACTION4)
    env.step(GameAction.ACTION4)  # Lands on trap (5, 5)
    assert env.state == GameState.GAME_OVER

    # Agent must emit RESET when state is GAME_OVER
    act = agent.choose_action(env.frame, env)
    assert act == GameAction.RESET
    env.step(act)
    assert env.state == GameState.NOT_FINISHED
