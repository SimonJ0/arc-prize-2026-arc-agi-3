"""
Capability Diagnostic 1: Object Identification under Background Noise.
Verifies that the agent correctly identifies entities and background in noisy grid contexts.
"""

import numpy as np
from src.arc_agent.perception.layered_perception import LayeredPerception


def test_object_identification_under_noise():
    perception = LayeredPerception()
    grid = np.zeros((20, 20), dtype=int)
    # Background is color 0
    # Add an avatar object (color 2, 2x2)
    grid[5:7, 5:7] = 2
    # Add a target object (color 5, 3x3)
    grid[12:15, 12:15] = 5
    # Add a single-pixel distractor (color 8)
    grid[2, 18] = 8

    analysis = perception.analyze(grid)
    bg_color = analysis.background_hypotheses[0][0]
    assert bg_color == 0
    assert len(analysis.entities) >= 3

    colors = {e.color for e in analysis.entities}
    assert 2 in colors
    assert 5 in colors
    assert 8 in colors
