"""Candidate Requirement Specification (CRS) — bridges biology evidence to chemistry, per
docs/overview.md layer 7 and docs/design-principles.md's "LLM must not invent requirements"
rule: the LLM (if provided) only SELECTS and WEIGHTS from this fixed property ontology; any
property name it returns outside the ontology is dropped, never passed through.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

PROPERTY_ONTOLOGY = [
    "target_interaction", "selectivity", "physicochemical", "stability",
    "solubility", "permeability", "developability", "toxicity_filters", "diversity",
]

# Deterministic fallback / prior used when no LLM is available, and as the reference the LLM
# is shown when asked to select/weight — keyed by the RBT's heuristic mechanism_hint.
MECHANISM_PROPERTY_PRIORS: dict[str, dict[str, float]] = {
    "efflux_regulator": {"target_interaction": 0.8, "permeability": 0.5, "selectivity": 0.6},
    "porin_permeability": {"permeability": 0.9, "physicochemical": 0.6},
    "beta_lactamase": {"target_interaction": 0.9, "selectivity": 0.7, "stability": 0.5},
}

_id_counter = itertools.count(1)


@dataclass
class Requirement:
    requirement_id: str
    property_name: str
    weight: float
    rationale: str
    evidence_ids: list[str] = field(default_factory=list)


class CandidateRequirementSpecification:
    def __init__(self, llm_select_fn=None):
        """llm_select_fn(mechanism_hint, top_hits, ontology, priors) -> dict[str, float],
        or None to use MECHANISM_PROPERTY_PRIORS directly (deterministic fallback)."""
        self.llm_select_fn = llm_select_fn

    def build_requirements(self, rbt_records: list[dict]) -> list[Requirement]:
        requirements = []
        for rec in rbt_records:
            if rec.get("evidence_gap"):
                continue
            mechanism = rec["mechanism_hint"]
            priors = MECHANISM_PROPERTY_PRIORS.get(mechanism, {})

            if self.llm_select_fn is not None:
                try:
                    selected = self.llm_select_fn(mechanism, rec.get("top_hits", []),
                                                    PROPERTY_ONTOLOGY, priors)
                except Exception:
                    selected = priors
            else:
                selected = priors

            for prop, weight in selected.items():
                if prop not in PROPERTY_ONTOLOGY:
                    continue  # reject anything outside the fixed ontology — never invented
                try:
                    weight = float(weight)
                except (TypeError, ValueError):
                    continue
                weight = max(0.0, min(1.0, weight))
                requirements.append(Requirement(
                    requirement_id=f"R{next(_id_counter)}",
                    property_name=prop,
                    weight=weight,
                    rationale=f"mechanism hint '{mechanism}' (cluster {rec['cluster_id']})",
                    evidence_ids=list(rec.get("evidence_ids", [])),
                ))
        return requirements


def build_static_requirements() -> list[Requirement]:
    """Condition 1 ('static chemistry', docs/experiments.md): no AMR-MoEGA, no ProteinRAG, no
    evidence at all — a uniform weight across every ontology property, with no provenance.
    This is the floor the other conditions need to beat: if evolution/biology-informed
    requirements don't outperform this, they aren't adding value."""
    return [
        Requirement(requirement_id=f"R{next(_id_counter)}", property_name=prop, weight=0.5,
                    rationale="static baseline: uniform weight, no evidence used",
                    evidence_ids=[])
        for prop in PROPERTY_ONTOLOGY
    ]
