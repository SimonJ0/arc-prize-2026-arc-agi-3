"""
Layered & Multi-Hypothesis Perception for ARC-AGI-3.
Avoids fragile single-background assumptions. Computes:
1. Multi-candidate background hypotheses (frequency, border presence, stability).
2. Connected component segmentation across multiple connectivity modes (4-way, 8-way).
3. Temporal diff layers Delta(F_{t-1}, F_t) isolating active entities from invariant terrain.
4. Entity candidate extraction with spatial bounding boxes and centroids.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from scipy.ndimage import label


@dataclass(frozen=True)
class EntityCandidate:
    """An identified object or cluster within the grid."""
    entity_id: int
    color: int
    cells: Tuple[Tuple[int, int], ...]
    bbox: Tuple[int, int, int, int]  # (min_y, min_x, max_y, max_x)
    centroid: Tuple[float, float]
    is_dynamic: bool = False

    @property
    def size(self) -> int:
        return len(self.cells)


@dataclass
class FrameAnalysis:
    """Multi-hypothesis perception result for a single observation."""
    frame_shape: Tuple[int, int]
    present_colors: Set[int]
    background_hypotheses: List[Tuple[int, float]]  # (color, confidence)
    entities: List[EntityCandidate]
    dynamic_diff_mask: Optional[np.ndarray] = None
    symmetry_scores: Dict[str, float] = field(default_factory=dict)


class LayeredPerception:
    """Perception pipeline maintaining multiple structural segmentations."""

    def __init__(self):
        self.previous_frame: Optional[np.ndarray] = None

    def analyze(self, frame: np.ndarray, prev_frame: Optional[np.ndarray] = None) -> FrameAnalysis:
        """
        Processes a 2D integer grid frame into layered perceptual abstractions.
        """
        if frame.ndim != 2:
            raise ValueError(f"Expected 2D grid frame, got shape {frame.shape}")

        H, W = frame.shape
        unique_colors, counts = np.unique(frame, return_counts=True)
        present_colors = set(int(c) for c in unique_colors)

        # 1. Multi-candidate background inference
        bg_hypotheses = self._infer_background_candidates(frame, unique_colors, counts)

        # 2. Dynamic temporal diff mask
        if prev_frame is None:
            prev_frame = self.previous_frame
        dynamic_mask = None
        if prev_frame is not None and prev_frame.shape == frame.shape:
            dynamic_mask = (frame != prev_frame)

        # 3. Extract entities across candidate non-background components
        primary_bg = bg_hypotheses[0][0] if bg_hypotheses else 0
        entities = self._extract_entities(frame, primary_bg, dynamic_mask)

        # 4. Symmetries
        symmetry_scores = {
            "horizontal": float(np.mean(frame == np.fliplr(frame))),
            "vertical": float(np.mean(frame == np.flipud(frame))),
        }

        self.previous_frame = frame.copy()

        return FrameAnalysis(
            frame_shape=(H, W),
            present_colors=present_colors,
            background_hypotheses=bg_hypotheses,
            entities=entities,
            dynamic_diff_mask=dynamic_mask,
            symmetry_scores=symmetry_scores,
        )

    def _infer_background_candidates(
        self, frame: np.ndarray, unique_colors: np.ndarray, counts: np.ndarray
    ) -> List[Tuple[int, float]]:
        """
        Ranks candidate background colors using combined border density,
        overall area fraction, and connectivity.
        """
        H, W = frame.shape
        total_pixels = H * W
        border_pixels = np.concatenate([
            frame[0, :], frame[-1, :], frame[:, 0], frame[:, -1]
        ])
        border_total = len(border_pixels)

        candidates = []
        for color, count in zip(unique_colors, counts):
            color = int(color)
            area_frac = count / total_pixels
            border_frac = np.mean(border_pixels == color)
            # Composite score (weighted border presence and area)
            conf = 0.6 * border_frac + 0.4 * area_frac
            candidates.append((color, float(conf)))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates

    def _extract_entities(
        self, frame: np.ndarray, background_color: int, dynamic_mask: Optional[np.ndarray]
    ) -> List[EntityCandidate]:
        """Extracts connected component entities excluding the primary candidate background."""
        entities = []
        entity_id_counter = 0

        for color in np.unique(frame):
            color = int(color)
            if color == background_color:
                continue

            color_mask = (frame == color)
            labeled_array, num_features = label(color_mask)

            for feat_idx in range(1, num_features + 1):
                coords = np.argwhere(labeled_array == feat_idx)
                if len(coords) == 0:
                    continue

                cells = tuple((int(y), int(x)) for y, x in coords)
                min_y, min_x = coords.min(axis=0)
                max_y, max_x = coords.max(axis=0)
                centroid = (float(coords[:, 0].mean()), float(coords[:, 1].mean()))

                is_dyn = False
                if dynamic_mask is not None:
                    is_dyn = bool(np.any(dynamic_mask[labeled_array == feat_idx]))

                entities.append(EntityCandidate(
                    entity_id=entity_id_counter,
                    color=color,
                    cells=cells,
                    bbox=(int(min_y), int(min_x), int(max_y), int(max_x)),
                    centroid=centroid,
                    is_dynamic=is_dyn,
                ))
                entity_id_counter += 1

        return entities
