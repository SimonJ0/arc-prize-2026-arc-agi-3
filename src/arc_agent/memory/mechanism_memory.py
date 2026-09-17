"""
Causal Mechanism Memory for ARC-AGI-3 (Baseline 3.0 B3.02).
Maintains an indexed ledger of causal interventions, mapping:
  (EntitySignature, Action) -> (EffectType, LatentVariableDelta, ChangedEntities, Reversibility).
Enables the agent to infer causal control systems rather than raw coordinate blacklists.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from src.arc_agent.memory.effect_taxonomy import EffectType


@dataclass(frozen=True)
class EntitySignature:
    """
    Abstract invariant representation of an entity's structural morphology.
    Allows transfer and equivalence grouping regardless of transient pixel coordinates.
    """

    color: int
    size_bucket: str          # "tiny" (<10), "small" (10-50), "medium" (50-120), "large" (>120)
    aspect_ratio_bucket: str  # "square" (0.8-1.2), "wide" (>1.2), "tall" (<0.8)
    solidity_bucket: str      # "solid" (>=0.8), "sparse" (<0.8)

    @classmethod
    def from_entity(cls, entity: Any) -> EntitySignature:
        """Constructs an invariant signature from an EntityCandidate or AttributeProfile."""
        color = getattr(entity, "color", 0)
        size = getattr(entity, "size", 1)

        # Size bucket
        if size < 10:
            size_b = "tiny"
        elif size <= 50:
            size_b = "small"
        elif size <= 120:
            size_b = "medium"
        else:
            size_b = "large"

        # Aspect ratio & solidity
        bbox = getattr(entity, "bbox", (0, 0, 1, 1))
        h = max(1, bbox[2] - bbox[0])
        w = max(1, bbox[3] - bbox[1])
        ar = w / h
        if 0.8 <= ar <= 1.2:
            ar_b = "square"
        elif ar > 1.2:
            ar_b = "wide"
        else:
            ar_b = "tall"

        bbox_area = max(1, h * w)
        solidity = size / bbox_area
        sol_b = "solid" if solidity >= 0.8 else "sparse"

        return cls(
            color=color,
            size_bucket=size_b,
            aspect_ratio_bucket=ar_b,
            solidity_bucket=sol_b,
        )


@dataclass
class MechanismRecord:
    """Detailed record of an intervention outcome in the causal ledger."""

    step: int
    action: str
    target_coord: tuple[int, int]
    target_entity: EntitySignature | None
    effect_type: EffectType
    diff_count: int
    changed_entity_signatures: list[EntitySignature] = field(default_factory=list)
    latent_variable_id: str | None = None
    latent_delta: int | None = None
    level_advanced: bool = False
    is_lethal: bool = False
    reversible: bool | None = None
    inverse_action_coord: tuple[int, int] | None = None
    pre_frame_hash: str = ""
    post_frame_hash: str = ""


@dataclass
class MechanismHypothesis:
    """Separates causal mechanism certainty from goal relevance certainty."""

    mechanism_id: str
    target_coord: tuple[int, int]
    entity_signature: EntitySignature | None
    dominant_effect: EffectType
    causal_confidence: float = 0.5  # "I know what this interaction does"
    goal_relevance: float = 0.5     # "I know this interaction predicts winning"
    observations: int = 0
    concordant_effects: int = 0
    progress_correlations: int = 0
    is_lethal: bool = False
    reversible: bool = False

    def update(self, effect: EffectType, level_advanced: bool, is_lethal: bool = False):
        self.observations += 1
        if is_lethal:
            self.is_lethal = True
            self.goal_relevance = 0.0
            self.causal_confidence = 1.0
            return

        if effect == self.dominant_effect:
            self.concordant_effects += 1
        elif self.concordant_effects == 0:
            self.dominant_effect = effect
            self.concordant_effects = 1

        # Causal confidence: how consistently does it produce this effect?
        self.causal_confidence = (self.concordant_effects + 1.0) / (self.observations + 2.0)

        # Goal relevance: does this effect advance the level or transform structural/global board geometry?
        if level_advanced or effect in (
            EffectType.STRUCTURAL_MUTATION,
            EffectType.GLOBAL_MUTATION,
            EffectType.LEVEL_ADVANCE,
        ):
            self.progress_correlations += 1
        self.goal_relevance = (self.progress_correlations + 1.0) / (self.observations + 2.0)

    def can_reliably_exploit(self) -> bool:
        """Only exploit when both mechanism and goal relevance confidence exceed threshold."""
        return (
            not self.is_lethal
            and self.causal_confidence >= 0.60
            and self.goal_relevance >= 0.50
        )


class MechanismMemory:
    """
    Indexed causal transition ledger.
    Indexes interactions by entity signature, effect type, and spatial coordinates.
    """

    def __init__(self, max_records: int = 500):
        self.max_records = max_records
        self.records: list[MechanismRecord] = []
        self.signature_to_effects: dict[EntitySignature, list[EffectType]] = defaultdict(list)
        self.active_controllers: set[tuple[int, int]] = set()
        self.inert_signatures: set[EntitySignature] = set()
        self.active_signatures: set[EntitySignature] = set()
        self.hypotheses: dict[tuple[int, int], MechanismHypothesis] = {}

    def record_transition(
        self,
        step: int,
        action: str,
        target_coord: tuple[int, int],
        target_entity: EntitySignature | None,
        effect_type: EffectType,
        diff_count: int,
        changed_entity_signatures: list[EntitySignature] | None = None,
        latent_variable_id: str | None = None,
        latent_delta: int | None = None,
        level_advanced: bool = False,
        is_lethal: bool = False,
        pre_frame_hash: str = "",
        post_frame_hash: str = "",
    ) -> MechanismRecord:
        """Appends and indexes a causal transition in the ledger."""
        rec = MechanismRecord(
            step=step,
            action=action,
            target_coord=target_coord,
            target_entity=target_entity,
            effect_type=effect_type,
            diff_count=diff_count,
            changed_entity_signatures=changed_entity_signatures or [],
            latent_variable_id=latent_variable_id,
            latent_delta=latent_delta,
            level_advanced=level_advanced,
            is_lethal=is_lethal,
            pre_frame_hash=pre_frame_hash,
            post_frame_hash=post_frame_hash,
        )
        self.records.append(rec)
        if len(self.records) > self.max_records:
            self.records.pop(0)

        # Indexing
        if target_entity is not None:
            self.signature_to_effects[target_entity].append(effect_type)

            if effect_type in (EffectType.NONE, EffectType.UI_NOISE):
                self.inert_signatures.add(target_entity)
            elif effect_type in (
                EffectType.LOCAL_MUTATION,
                EffectType.STRUCTURAL_MUTATION,
                EffectType.GLOBAL_MUTATION,
                EffectType.LEVEL_ADVANCE,
            ):
                self.active_signatures.add(target_entity)
                self.active_controllers.add(target_coord)
                if target_entity in self.inert_signatures:
                    self.inert_signatures.remove(target_entity)

        # Update hypothesis for this coordinate
        if target_coord not in self.hypotheses:
            self.hypotheses[target_coord] = MechanismHypothesis(
                mechanism_id=f"mech_{target_coord[0]}_{target_coord[1]}",
                target_coord=target_coord,
                entity_signature=target_entity,
                dominant_effect=effect_type,
            )
        self.hypotheses[target_coord].update(
            effect=effect_type,
            level_advanced=level_advanced,
            is_lethal=is_lethal,
        )

        return rec

    def get_hypothesis(self, coord: tuple[int, int]) -> MechanismHypothesis | None:
        """Returns the MechanismHypothesis for this coordinate if observed."""
        return self.hypotheses.get(coord)

    def get_exploitable_controllers(self) -> list[tuple[int, int]]:
        """Returns coordinates where both causal and goal relevance confidence permit exploitation."""
        return [c for c, hyp in self.hypotheses.items() if hyp.can_reliably_exploit()]

    def get_by_entity_signature(self, sig: EntitySignature) -> list[MechanismRecord]:
        """Returns all historical transitions targeting entities with matching signature."""
        return [r for r in self.records if r.target_entity == sig]

    def get_by_effect_type(self, effect_type: EffectType) -> list[MechanismRecord]:
        """Returns all historical transitions yielding the specified effect type."""
        return [r for r in self.records if r.effect_type == effect_type]

    def get_active_controllers(self) -> list[tuple[int, int]]:
        """Returns verified active coordinate controllers."""
        return list(self.active_controllers)

    def get_inert_signatures(self) -> set[EntitySignature]:
        """Returns entity signatures confirmed to produce no meaningful state change."""
        return set(self.inert_signatures)

    def is_signature_inert(self, sig: EntitySignature) -> bool:
        """Returns True if the entity signature is confirmed inert."""
        return sig in self.inert_signatures

    def is_signature_active(self, sig: EntitySignature) -> bool:
        """Returns True if the entity signature is confirmed active."""
        return sig in self.active_signatures

    def compute_equivalence_classes(self, entities: Sequence[Any]) -> dict[EntitySignature, list[Any]]:
        """Groups candidate entities into morphological equivalence classes."""
        classes: dict[EntitySignature, list[Any]] = defaultdict(list)
        for ent in entities:
            sig = EntitySignature.from_entity(ent)
            classes[sig].append(ent)
        return dict(classes)

    def filter_inert_classes(self, entities: Sequence[Any]) -> list[Any]:
        """
        Prunes all entities belonging to an equivalence class already confirmed inert.
        If all entities would be pruned, returns the original list as fallback.
        """
        if not self.inert_signatures:
            return list(entities)

        viable = []
        for ent in entities:
            sig = EntitySignature.from_entity(ent)
            if sig not in self.inert_signatures:
                viable.append(ent)

        return viable if viable else list(entities)

    def compress_to_symbolic(self) -> str:
        """
        Compresses the causal ledger into compact Astra-style symbolic shorthand.
        Example: 'CTRL=[(32,5):STRU,(32,58):STRU] INERT_SIGS=1 ACTIVE_SIGS=1'
        """
        ctrl_parts = [
            f"({c[0]},{c[1]}):{self.hypotheses[c].dominant_effect.value[:4]}"
            for c in sorted(self.active_controllers)
            if c in self.hypotheses
        ]
        ctrl_str = ",".join(ctrl_parts) if ctrl_parts else "none"
        inert_count = len(self.inert_signatures)
        active_count = len(self.active_signatures)

        return f"CTRL=[{ctrl_str}] INERT_SIGS={inert_count} ACTIVE_SIGS={active_count}"

    def reset_level(self, keep_abstract_signatures: bool = True):
        """
        Resets level-scoped coordinates while optionally preserving abstract entity signatures.
        """
        self.active_controllers.clear()
        self.hypotheses.clear()
        if not keep_abstract_signatures:
            self.records.clear()
            self.signature_to_effects.clear()
            self.inert_signatures.clear()
            self.active_signatures.clear()
