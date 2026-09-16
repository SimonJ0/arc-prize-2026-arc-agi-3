"""
DRE-Bench 4-Level Cognitive Hierarchy for ARC-AGI-3 (arXiv:2506.02648v1).
Decomposes perception and causal reasoning across four cognitive tiers:
  Level 1: Attribute (size, count, color distribution, bounding box solidity)
  Level 2: Spatial (directional vectors, symmetry axes, rotation, canonicalization)
  Level 3: Sequential (multi-step macro-planning overcoming the 2-step depth collapse)
  Level 4: Conceptual / Intuitive Physics (gravity fields, collision barriers, reflection)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from scipy.ndimage import label


@dataclass(frozen=True)
class AttributeProfile:
    """Level 1 Attribute abstraction of an entity."""
    entity_id: int
    color: int
    size: int
    bbox: Tuple[int, int, int, int]  # (min_y, min_x, max_y, max_x)
    centroid: Tuple[float, float]
    aspect_ratio: float
    solidity: float  # size / (bbox_height * bbox_width)
    is_singleton: bool  # size == 1


@dataclass(frozen=True)
class SpatialSymmetry:
    """Level 2 Spatial symmetry detection."""
    horizontal: float
    vertical: float
    diagonal: float


@dataclass
class CognitiveHierarchyAnalysis:
    """Unified 4-level cognitive evaluation of an observation."""
    # Level 1: Attribute
    attributes: List[AttributeProfile]
    color_counts: Dict[int, int]
    singleton_entities: List[AttributeProfile]
    dominant_color: int

    # Level 2: Spatial
    symmetry: SpatialSymmetry
    player_pos: Optional[Tuple[int, int]] = None
    player_color: Optional[int] = None

    # Level 3: Sequential Targets
    candidate_goals: List[Tuple[int, int]] = field(default_factory=list)

    # Level 4: Intuitive Physics
    gravity_detected: bool = False
    gravity_vector: Tuple[int, int] = (0, 0)
    static_obstacles: Set[Tuple[int, int]] = field(default_factory=set)


class CognitiveHierarchyPerception:
    """Analyzes ARC-AGI-3 frames using DRE-Bench's four cognitive levels."""

    def __init__(self):
        self.prev_frame: Optional[np.ndarray] = None
        self.prev_player_pos: Optional[Tuple[int, int]] = None

    def analyze(
        self,
        frame: np.ndarray,
        prev_frame: Optional[np.ndarray] = None,
        known_player_color: Optional[int] = None,
    ) -> CognitiveHierarchyAnalysis:
        """Executes full 4-level cognitive breakdown of the grid."""
        H, W = frame.shape

        # --- LEVEL 1: ATTRIBUTE ANALYSIS ---
        unique_colors, counts = np.unique(frame, return_counts=True)
        color_counts = {int(c): int(cnt) for c, cnt in zip(unique_colors, counts)}
        # Background is typically the color with maximum area
        dominant_color = int(unique_colors[np.argmax(counts)])

        attributes: List[AttributeProfile] = []
        singleton_entities: List[AttributeProfile] = []
        entity_id_seq = 0

        for color in unique_colors:
            color = int(color)
            if color == dominant_color:
                continue

            color_mask = (frame == color)
            labeled_arr, num_feats = label(color_mask)

            for feat_idx in range(1, num_feats + 1):
                coords = np.argwhere(labeled_arr == feat_idx)
                if len(coords) == 0:
                    continue

                size = len(coords)
                min_y, min_x = coords.min(axis=0)
                max_y, max_x = coords.max(axis=0)
                bh = max(1, max_y - min_y + 1)
                bw = max(1, max_x - min_x + 1)
                bbox_area = bh * bw
                solidity = float(size / bbox_area)
                aspect_ratio = float(bw / bh)
                centroid = (float(coords[:, 0].mean()), float(coords[:, 1].mean()))

                profile = AttributeProfile(
                    entity_id=entity_id_seq,
                    color=color,
                    size=size,
                    bbox=(int(min_y), int(min_x), int(max_y), int(max_x)),
                    centroid=centroid,
                    aspect_ratio=aspect_ratio,
                    solidity=solidity,
                    is_singleton=(size == 1),
                )
                attributes.append(profile)
                if size <= 4:
                    singleton_entities.append(profile)
                entity_id_seq += 1

        # --- LEVEL 2: SPATIAL ANALYSIS ---
        h_sym = float(np.mean(frame == np.fliplr(frame)))
        v_sym = float(np.mean(frame == np.flipud(frame)))
        d_sym = float(np.mean(frame == frame.T)) if H == W else 0.0
        symmetry = SpatialSymmetry(horizontal=h_sym, vertical=v_sym, diagonal=d_sym)

        # Infer player location:
        # If known_player_color is specified, find its centroid;
        # otherwise look for mobile singleton or dynamic entity
        player_pos = None
        player_color = known_player_color

        if player_color is not None:
            p_coords = np.argwhere(frame == player_color)
            if len(p_coords) > 0:
                player_pos = (int(p_coords[:, 0].mean()), int(p_coords[:, 1].mean()))

        if player_pos is None and prev_frame is not None and prev_frame.shape == frame.shape:
            # Find moving pixels
            diff = (frame != prev_frame)
            if np.any(diff):
                # Pixels present in current frame but not previous
                curr_diff_colors = frame[diff]
                # Player is usually a small moving entity
                for profile in singleton_entities:
                    cy, cx = int(profile.centroid[0]), int(profile.centroid[1])
                    if diff[cy, cx]:
                        player_pos = (cy, cx)
                        player_color = profile.color
                        break

        # Fallback player: first small singleton entity
        if player_pos is None and singleton_entities:
            target = singleton_entities[0]
            player_pos = (int(target.centroid[0]), int(target.centroid[1]))
            player_color = target.color

        # --- LEVEL 3: CANDIDATE GOALS (Sequential targets) ---
        candidate_goals: List[Tuple[int, int]] = []
        for profile in attributes:
            if player_color is not None and profile.color == player_color:
                continue
            # Goals are typically small unique objects (doors, sockets, stars)
            if profile.size <= 25:
                candidate_goals.append((int(profile.centroid[0]), int(profile.centroid[1])))

        # Sort candidate goals by distance to player
        if player_pos is not None:
            candidate_goals.sort(
                key=lambda g: abs(g[0] - player_pos[0]) + abs(g[1] - player_pos[1])
            )

        # --- LEVEL 4: INTUITIVE PHYSICS (Obstacles & Gravity) ---
        static_obstacles: Set[Tuple[int, int]] = set()
        # Large entities (>60 pixels) or boundary clusters are treated as impassable walls
        for profile in attributes:
            if profile.size >= 40:
                min_y, min_x, max_y, max_x = profile.bbox
                for y in range(min_y, max_y + 1):
                    for x in range(min_x, max_x + 1):
                        if frame[y, x] == profile.color:
                            static_obstacles.add((y, x))

        # Check for gravity (downward vertical displacement across unforced steps)
        gravity_detected = False
        gravity_vector = (0, 0)
        if (
            self.prev_frame is not None
            and prev_frame is not None
            and player_pos is not None
            and self.prev_player_pos is not None
        ):
            dy = player_pos[0] - self.prev_player_pos[0]
            dx = player_pos[1] - self.prev_player_pos[1]
            if dy > 0 and dx == 0:
                gravity_detected = True
                gravity_vector = (1, 0)

        self.prev_frame = frame.copy()
        if player_pos is not None:
            self.prev_player_pos = player_pos

        return CognitiveHierarchyAnalysis(
            attributes=attributes,
            color_counts=color_counts,
            singleton_entities=singleton_entities,
            dominant_color=dominant_color,
            symmetry=symmetry,
            player_pos=player_pos,
            player_color=player_color,
            candidate_goals=candidate_goals,
            gravity_detected=gravity_detected,
            gravity_vector=gravity_vector,
            static_obstacles=static_obstacles,
        )
