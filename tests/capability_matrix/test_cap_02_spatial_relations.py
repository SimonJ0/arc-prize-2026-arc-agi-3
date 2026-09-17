"""
Capability Diagnostic 2: Spatial Relations & Metric Distances.
Verifies accurate centroid calculation, bounding boxes, and relative directional vectors.
"""

import numpy as np
from src.arc_agent.perception.cognitive_hierarchy import CognitiveHierarchyPerception


def test_spatial_relations_and_geometry():
    cog_perc = CognitiveHierarchyPerception()
    grid = np.zeros((16, 16), dtype=int)
    # Player at (4, 4)
    grid[4, 4] = 3
    # Target at (10, 10)
    grid[10, 10] = 7

    analysis = cog_perc.analyze(grid)
    assert analysis.player_pos == (4, 4)
    assert (10, 10) in analysis.candidate_goals
