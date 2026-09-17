"""
Cross-Level Skill Transfer & Invariant Schema Extraction for ARC-AGI-3.
Prevents negative transfer by separating relational invariant schemas from concrete tokens:
  1. Preserves verified cardinal kinematic mappings across levels (e.g. ACTION1 = UP).
  2. Preserves avatar morphology priors (e.g. singleton vs multi-pixel entity).
  3. Resets and dynamically re-binds concrete color tokens and coordinates per level.
"""

from __future__ import annotations

from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from typing import Any

from src.arc_agent.memory.mechanism_memory import EntitySignature
from src.arc_agent.memory.structured_belief import SkillMemory


@dataclass
class LevelTransferPackage:
    verified_kinematics: dict[str, tuple[int, int]]  # Action -> (dy, dx)
    avatar_is_singleton: bool
    successful_action_modalities: set[str]  # e.g. {"ACTION1", "ACTION2", "ACTION6"}
    total_levels_completed: int
    transferred_inert_signatures: set[EntitySignature] = field(default_factory=set)
    transferred_active_signatures: set[EntitySignature] = field(default_factory=set)


class LevelTransferManager:
    """Manages cross-level knowledge retention while safeguarding against color/token inversion."""

    def __init__(self):
        self.verified_kinematics: dict[str, tuple[int, int]] = {}
        self.avatar_is_singleton: bool = True
        self.confirmed_modalities: set[str] = set()
        self.completed_level_count: int = 0
        self.inert_signatures: set[EntitySignature] = set()
        self.active_signatures: set[EntitySignature] = set()

    def extract_level_schema(
        self,
        level_index: int,
        kinematic_mappings: dict[str, tuple[int, int]],
        avatar_size: int,
        actions_used: set[str],
        skill_memory: SkillMemory,
        game_id: str,
        mechanism_memory: Any | None = None,
    ) -> LevelTransferPackage:
        """Called immediately upon completing a level to compile reusable invariant schemas."""
        self.completed_level_count += 1
        self.verified_kinematics.update(kinematic_mappings)
        self.avatar_is_singleton = (avatar_size <= 1)
        self.confirmed_modalities.update(actions_used)

        if mechanism_memory is not None and hasattr(mechanism_memory, "get_inert_signatures"):
            self.inert_signatures.update(mechanism_memory.get_inert_signatures())
            self.active_signatures.update(getattr(mechanism_memory, "active_signatures", set()))

        pkg = LevelTransferPackage(
            verified_kinematics=dict(self.verified_kinematics),
            avatar_is_singleton=self.avatar_is_singleton,
            successful_action_modalities=set(self.confirmed_modalities),
            total_levels_completed=self.completed_level_count,
            transferred_inert_signatures=set(self.inert_signatures),
            transferred_active_signatures=set(self.active_signatures),
        )

        skill_memory.save_skill(
            skill_id=f"level_{level_index}_kinematic_transfer",
            origin_game=game_id,
            origin_level=level_index,
            rule_type="kinematic_mapping",
            schema={"kinematics": dict(self.verified_kinematics)},
            confidence=0.95,
        )

        return pkg

    def apply_transfer_priors(
        self,
        available_actions: list[str],
    ) -> dict[str, tuple[int, int]]:
        """Provides verified kinematic priors to the new level without carrying over concrete colors."""
        priors: dict[str, tuple[int, int]] = {}
        for act in available_actions:
            if act in self.verified_kinematics:
                priors[act] = self.verified_kinematics[act]
        return priors

    def apply_mechanism_priors(self, mechanism_memory: Any):
        """Populates new level's mechanism memory with verified abstract entity priors."""
        if hasattr(mechanism_memory, "inert_signatures"):
            mechanism_memory.inert_signatures.update(self.inert_signatures)
        if hasattr(mechanism_memory, "active_signatures"):
            mechanism_memory.active_signatures.update(self.active_signatures)
