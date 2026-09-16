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
    is_stagnant: bool = False


class CognitiveHierarchyPerception:
    """Analyzes ARC-AGI-3 frames using DRE-Bench's four cognitive levels."""

    def __init__(self):
        self.prev_frame: Optional[np.ndarray] = None
        self.prev_player_pos: Optional[Tuple[int, int]] = None
        self.stagnant_steps: int = 0

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
        player_pos = None
        player_color = known_player_color

        # 1. Prioritize dynamic motion diffs across consecutive frames
        if prev_frame is not None and prev_frame.shape == frame.shape:
            diff = (frame != prev_frame)
            if np.any(diff):
                best_entity = None
                best_size = 999999
                for profile in attributes:
                    if profile.color == dominant_color:
                        continue
                    # Ignore UI indicators on extreme boundary rows if size <= 4
                    cy, cx = profile.centroid
                    if (int(cy) <= 1 or int(cy) >= H - 2) and profile.size <= 4:
                        continue
                    min_y, min_x, max_y, max_x = profile.bbox
                    ent_diff = diff[min_y:max_y+1, min_x:max_x+1] & (frame[min_y:max_y+1, min_x:max_x+1] == profile.color)
                    if np.any(ent_diff):
                        if profile.size < best_size:
                            best_entity = profile
                            best_size = profile.size
                if best_entity is not None:
                    player_pos = (int(best_entity.centroid[0]), int(best_entity.centroid[1]))
                    player_color = best_entity.color

        # 2. If no dynamic motion detected or first frame, use known player color
        if player_pos is None and player_color is not None:
            p_coords = np.argwhere(frame == player_color)
            if len(p_coords) > 0:
                player_pos = (int(p_coords[:, 0].mean()), int(p_coords[:, 1].mean()))

        # 3. Fallback player: smallest non-dominant entity
        if player_pos is None and singleton_entities:
            target = singleton_entities[0]
            player_pos = (int(target.centroid[0]), int(target.centroid[1]))
            player_color = target.color
        elif player_pos is None and attributes:
            sorted_candidates = sorted([a for a in attributes if a.color != dominant_color], key=lambda a: a.size)
            if sorted_candidates:
                target = sorted_candidates[0]
                player_pos = (int(target.centroid[0]), int(target.centroid[1]))
                player_color = target.color

        # Track position stagnation (failsafe against false static avatar locks)
        if self.prev_player_pos is not None and player_pos is not None and player_pos == self.prev_player_pos:
            self.stagnant_steps += 1
        else:
            self.stagnant_steps = 0
        is_stagnant = (self.stagnant_steps >= 4)

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
        player_standing_color = frame[player_pos[0], player_pos[1]] if player_pos is not None else None

        for profile in attributes:
            # Walkable surface/floor the player stands on is NEVER an obstacle
            if player_standing_color is not None and profile.color == player_standing_color:
                continue
            if player_color is not None and profile.color == player_color:
                continue
            # Small interactive entities (<35 pixels: keys, doors, stars) are never obstacles
            if profile.size < 35:
                continue

            min_y, min_x, max_y, max_x = profile.bbox
            touches_border = (min_y == 0 or max_y == H - 1 or min_x == 0 or max_x == W - 1)

            # Only entities touching the border with high solidity are treated as static boundary walls
            if touches_border and profile.solidity >= 0.7:
                for y in range(min_y, max_y + 1):
                    for x in range(min_x, max_x + 1):
                        if frame[y, x] == profile.color:
                            static_obstacles.add((y, x))

        if player_pos is not None:
            static_obstacles.discard(player_pos)

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
            is_stagnant=is_stagnant,
        )
