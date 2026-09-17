"""
Unit tests verifying contract compliance and legality for all 4 cognitive hypotheses:
Hypothesis A: LatentWorldModelAgent
Hypothesis B: ObjectDslAgent
Hypothesis C: EpistemicCuriosityAgent
Hypothesis D: HdNsaAgent
"""

import numpy as np
import pytest
from arcengine import GameAction, GameState

from src.arc_agent.models.latent_world_model_agent import LatentWorldModelAgent
from src.arc_agent.models.object_dsl_agent import ObjectDslAgent
from src.arc_agent.models.epistemic_curiosity_agent import EpistemicCuriosityAgent
from src.arc_agent.models.hd_nsa_agent import HdNsaAgent


class DummyFrameData:
    def __init__(self, frame, available_actions, state=GameState.NOT_FINISHED):
        self.frame = [frame]
        self.available_actions = available_actions
        self.state = state
        self.levels_completed = 0


@pytest.mark.parametrize("AgentCls", [
    LatentWorldModelAgent,
    ObjectDslAgent,
    EpistemicCuriosityAgent,
    HdNsaAgent,
])
def test_agent_contract_and_legality(AgentCls):
    agent = AgentCls(game_id="unit_test_game")
    assert agent.game_id == "unit_test_game"

    grid = np.zeros((16, 16), dtype=int)
    grid[5, 5] = 2  # Avatar
    grid[5, 10] = 3 # Target

    available = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]
    frame = DummyFrameData(grid, available)

    # Initial action
    act = agent.choose_action(frame.frame, frame)
    assert isinstance(act, GameAction)
    assert act in available

    # Transition with movement
    grid2 = grid.copy()
    grid2[5, 5] = 0
    grid2[5, 6] = 2
    frame2 = DummyFrameData(grid2, available)
    act2 = agent.choose_action(frame2.frame, frame2)
    assert isinstance(act2, GameAction)
    assert act2 in available

    # GAME_OVER must force RESET
    frame_game_over = DummyFrameData(grid2, [GameAction.RESET, GameAction.ACTION1], state=GameState.GAME_OVER)
    act3 = agent.choose_action(frame_game_over.frame, frame_game_over)
    assert act3 == GameAction.RESET