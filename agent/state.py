"""ScientificState, per docs/overview.md architecture layer 8."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from biology.evidence_graph import EvidenceGraph
from biology.hypotheses import Hypothesis
from biology.requirements import Requirement
from candidates.pool import Candidate


@dataclass
class TrajectoryRecord:
    iteration: int
    action: str
    action_args: dict[str, Any]
    observation: dict[str, Any]
    rationale: str | None = None
    hypothesis_update: str | None = None
    confidence: float | None = None


@dataclass
class ScientificState:
    question: str
    antibiotic: str
    genome_row: int
    budget_max: int = 20

    erp: dict | None = None
    evidence_graph: EvidenceGraph = field(default_factory=EvidenceGraph)
    hypotheses: dict[str, Hypothesis] = field(default_factory=dict)
    rbt_records: list[dict] = field(default_factory=list)
    requirements: list[Requirement] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    evaluations: dict[str, dict] = field(default_factory=dict)
    trajectory: list[TrajectoryRecord] = field(default_factory=list)
    uncertainty_flags: list[str] = field(default_factory=list)
    refinement_deltas: list[float] = field(default_factory=list)
    below_risk_threshold: bool = False

    budget_used: int = 0
    status: str = "running"  # running | stopped | human_review
    stopping_reason: str | None = None
    best_candidate_id: str | None = None

    def next_unmet_precondition(self) -> str | None:
        """Explicitly computed 'what's still blocking progress' — added after observing that
        the LLM policy would retrieve evidence, take several unrelated successful actions
        (VERIFY_HYPOTHESIS, more evidence retrieval), and then jump straight to REFINE_CANDIDATE
        without ever retrying BUILD_REQUIREMENTS, even though its precondition had long since
        been satisfied. A retry nudge keyed to 'your immediately-preceding action's error'
        doesn't fire once other successful actions happen in between — surfacing the next
        blocking step explicitly, every turn, doesn't depend on the model inferring it from
        trajectory history at all."""
        if self.erp is None:
            return "RUN_AMR_ANALYSIS (no ERP yet)"
        if self.below_risk_threshold:
            return ("STOP (resistance_risk is below the retrieval threshold for this genome — "
                     "no further evidence retrieval will change this; retrying "
                     "RETRIEVE_PROTEIN_EVIDENCE again will not help)")
        if not self.rbt_records:
            return "RETRIEVE_PROTEIN_EVIDENCE (no biological evidence yet)"
        if not self.requirements:
            return "BUILD_REQUIREMENTS (evidence is available — this is very likely your next action)"
        if not self.evaluations:
            return "EVALUATE_CANDIDATE (requirements are available — evaluate a candidate before refining)"
        return None

    def summary(self) -> dict:
        """Compact snapshot for the LLM policy prompt — NOT the full evidence graph dump."""
        return {
            "question": self.question,
            "antibiotic": self.antibiotic,
            "budget_used": self.budget_used,
            "budget_max": self.budget_max,
            "next_unmet_precondition": self.next_unmet_precondition(),
            "has_erp": self.erp is not None,
            "resistance_risk": self.erp.get("resistance_risk") if self.erp else None,
            "n_mechanism_clusters": len(self.erp["mechanism_clusters"]) if self.erp else 0,
            "n_rbt_records": len(self.rbt_records),
            "n_evidence_gaps": sum(1 for r in self.rbt_records if r.get("evidence_gap")),
            "n_requirements": len(self.requirements),
            "n_candidates": len(self.candidates),
            "n_evaluated": len(self.evaluations),
            "best_candidate_id": self.best_candidate_id,
            "best_score": self.evaluations.get(self.best_candidate_id, {}).get("overall_score")
                          if self.best_candidate_id else None,
            "hypotheses": [{"id": h.hypothesis_id, "statement": h.statement, "status": h.status,
                              "confidence": h.confidence} for h in self.hypotheses.values()],
            "uncertainty_flags": self.uncertainty_flags,
            "status": self.status,
        }
