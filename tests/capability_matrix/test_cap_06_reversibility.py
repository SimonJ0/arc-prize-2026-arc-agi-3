"""
Capability Diagnostic 6: Reversibility & Dead-End Recovery.
Verifies that the agent recovers from GAME_OVER traps using RESET.
"""

from arcengine import GameAction, GameState
from agent.my_agent import MyAgent
from tests.microworlds.test_reversibility import ReversibleTrapEnv


def test_reversibility_and_reset_recovery():
    env = ReversibleTrapEnv()
    agent = MyAgent(game_id="cap_reversibility")

    # Step into trap
    env.step(GameAction.ACTION4)
    env.step(GameAction.ACTION4)
    assert env.state == GameState.GAME_OVER

    # Agent must emit RESET
    act = agent.choose_action(env.frame, env)
    assert act == GameAction.RESET
