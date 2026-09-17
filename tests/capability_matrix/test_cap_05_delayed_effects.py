"""
Capability Diagnostic 5: Sequential & Delayed Effects (Key-Door Logic).
Verifies sequential prerequisite satisfaction to complete multi-stage goals.
"""

from agent.my_agent import MyAgent
from tests.microworlds.test_delayed_effects import KeyDoorEnv


def test_delayed_effects_sequential_unlock():
    env = KeyDoorEnv()
    agent = MyAgent(game_id="cap_delayed_effects")

    for _ in range(35):
        if agent.is_done(env.frame, env):
            break
        act = agent.choose_action(env.frame, env)
        env.step(act)
        if env.state.name == "WIN":
            break

    assert env.has_key is True
    assert env.state.name == "WIN"
