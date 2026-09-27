"""Finite action space, per docs/overview.md architecture layer 9. Each handler takes the
ScientificState + an ActionContext (tool instances) and returns an observation dict, mutating
state in place — this is what the LLM ReAct policy dispatches into."""
from __future__ import annotations

from dataclasses import dataclass

from biology.hypotheses import Hypothesis, new_hypothesis
from biology.risk_translation import RiskToBiologyTranslator
from biology.requirements import CandidateRequirementSpecification
from candidates import refiner
from candidates.evaluator import evaluate_candidate as evaluate_candidate_fn
from candidates.pool import Candidate
from agent.state import ScientificState

ACTIONS = [
    "RUN_AMR_ANALYSIS", "RETRIEVE_PROTEIN_EVIDENCE", "EXPAND_PROTEIN_FAMILY",
    "VERIFY_HYPOTHESIS", "BUILD_REQUIREMENTS", "EVALUATE_CANDIDATE",
    "REFINE_CANDIDATE", "RETRIEVE_ADDITIONAL_EVIDENCE", "REASSESS_HYPOTHESIS",
    "HUMAN_REVIEW", "STOP",
]

PRECONDITIONS = {
    "RUN_AMR_ANALYSIS": "only if no ERP exists yet for this genome — it's the most expensive "
                        "action (tens of minutes); once state.has_erp is true, do NOT call "
                        "this again unless you deliberately need to force a re-analysis",
    "RETRIEVE_PROTEIN_EVIDENCE": "requires ERP (run RUN_AMR_ANALYSIS first)",
    "EXPAND_PROTEIN_FAMILY": "requires prior RETRIEVE_PROTEIN_EVIDENCE",
    "VERIFY_HYPOTHESIS": "requires prior RETRIEVE_PROTEIN_EVIDENCE",
    "BUILD_REQUIREMENTS": "requires prior RETRIEVE_PROTEIN_EVIDENCE",
    "EVALUATE_CANDIDATE": "requires requirements (run BUILD_REQUIREMENTS first)",
    "REFINE_CANDIDATE": "requires an evaluated candidate and a transform_name",
    "RETRIEVE_ADDITIONAL_EVIDENCE": "requires prior RETRIEVE_PROTEIN_EVIDENCE",
    "REASSESS_HYPOTHESIS": "requires an existing hypothesis_id",
    "HUMAN_REVIEW": "always available",
    "STOP": "always available",
}


@dataclass
class ActionContext:
    config: dict
    amr_tool: object
    rbt: RiskToBiologyTranslator
    crs: CandidateRequirementSpecification
    transforms: list[dict]
    seed_pool: list[Candidate]


def _find_candidate(state: ScientificState, candidate_id: str | None) -> Candidate | None:
    if candidate_id is None:
        return None
    return next((c for c in state.candidates if c.candidate_id == candidate_id), None)


def _best_candidate(state: ScientificState) -> Candidate | None:
    if state.best_candidate_id is None:
        return state.candidates[0] if state.candidates else None
    return _find_candidate(state, state.best_candidate_id)


def _update_best(state: ScientificState) -> None:
    scored = [(cid, ev["overall_score"]) for cid, ev in state.evaluations.items() if ev.get("valid")]
    if scored:
        state.best_candidate_id = max(scored, key=lambda x: x[1])[0]


def run_amr_analysis(state: ScientificState, ctx: ActionContext, force: bool = False,
                      **kwargs) -> dict:
    # RUN_AMR_ANALYSIS is the single most expensive action (a full GA run, tens of minutes on
    # this hardware) — nothing about the tool itself is idempotent (cache=False by design, see
    # tools/amr_moega_tool.py), so without this guard a policy that re-picks this action (LLM
    # or otherwise) silently burns its entire budget re-deriving the same ERP. `force=True`
    # is the deliberate escape hatch for a real re-analysis request.
    if state.erp is not None and not force:
        return {"skipped": "ERP already computed for this genome_row — pass force=true to "
                            "re-run", "resistance_risk": state.erp["resistance_risk"]}
    result = ctx.amr_tool.run(antibiotic=state.antibiotic, genome_row=state.genome_row)
    if not result.ok:
        state.uncertainty_flags.append(f"amr_analysis_failed: {result.error}")
        return {"error": result.error}
    state.erp = result.output["erp"]
    return {"resistance_risk": state.erp["resistance_risk"],
            "n_mechanism_clusters": len(state.erp["mechanism_clusters"]),
            "held_out_val_auc": state.erp["held_out_val_auc"]}


def retrieve_protein_evidence(state: ScientificState, ctx: ActionContext, **kwargs) -> dict:
    if state.erp is None:
        return {"error": "no ERP yet — run RUN_AMR_ANALYSIS first"}
    records = ctx.rbt.translate(state.erp)
    state.rbt_records = records
    state.evidence_graph = ctx.rbt.graph

    if not records:
        # RiskToBiologyTranslator.translate() intentionally returns [] when resistance_risk is
        # below the retrieval threshold (correct behavior — don't spend a ProteinRAG query on a
        # genome the model doesn't believe is resistant). Confirmed by direct log analysis
        # (experiments/analyze_pilot.py) that WITHOUT this flag, every condition — including
        # the LLM-driven ones — got stuck calling this same action ~18/20 budget steps with no
        # escape, since the empty result never changes. This is a terminal state, not a
        # transient one: no amount of retrying produces evidence for a genuinely low-risk
        # genome, so the policy needs an explicit signal to stop instead of retry.
        state.below_risk_threshold = True
        return {"n_records": 0, "n_evidence_gaps": 0, "risk_below_threshold": True,
                "note": ("resistance_risk is below the retrieval threshold for this genome — "
                         "no further evidence retrieval will change this; recommend STOP.")}

    n_gap = sum(1 for r in records if r.get("evidence_gap"))
    if n_gap:
        state.uncertainty_flags.append(f"evidence_gap: {n_gap}/{len(records)} clusters unmapped")
    return {"n_records": len(records), "n_evidence_gaps": n_gap}


def expand_protein_family(state: ScientificState, ctx: ActionContext, **kwargs) -> dict:
    if not state.rbt_records:
        return {"error": "nothing to expand — run RETRIEVE_PROTEIN_EVIDENCE first"}
    expanded = 0
    for rec in state.rbt_records:
        if rec.get("evidence_gap") or not rec.get("gene"):
            continue
        seq = ctx.rbt.gene_seqs.get(rec["gene"])
        if not seq:
            continue
        wider = ctx.rbt.protein_rag.run(seq=seq, domains=[rec["mechanism_hint"]],
                                          top_k=ctx.config["protein_rag"]["rerank"]["top_k_final"] * 2)
        if not wider.ok:
            continue
        for hit in wider.output["results"]:
            ev = state.evidence_graph.add_evidence(
                source="protein_rag",
                claim=(f"{hit['acc']} (Pfam family {hit['family_id']}) is a wider-search "
                        f"evolutionary relative of query gene '{rec['gene']}'; "
                        f"homology tier={hit['homology_tier']}"),
                confidence=hit["s_final"], confidence_type="retrieval_relevance",
                provenance={"query_gene": rec["gene"], "expanded": True, **hit})
            rec.setdefault("evidence_ids", []).append(ev.evidence_id)
            expanded += 1
    return {"n_new_evidence": expanded}


def build_requirements(state: ScientificState, ctx: ActionContext, **kwargs) -> dict:
    if not state.rbt_records:
        return {"error": "no biology evidence yet — run RETRIEVE_PROTEIN_EVIDENCE first"}
    reqs = ctx.crs.build_requirements(state.rbt_records)
    state.requirements = reqs
    if not state.candidates:
        state.candidates = list(ctx.seed_pool)
    return {"n_requirements": len(reqs),
            "properties": [r.property_name for r in reqs]}


def evaluate_candidate(state: ScientificState, ctx: ActionContext,
                        candidate_id: str | None = None, **kwargs) -> dict:
    if not state.requirements:
        return {"error": "no requirements yet — run BUILD_REQUIREMENTS first"}
    cand = _find_candidate(state, candidate_id)
    if cand is None:
        unevaluated = [c for c in state.candidates if c.candidate_id not in state.evaluations]
        cand = unevaluated[0] if unevaluated else (state.candidates[0] if state.candidates else None)
    if cand is None:
        return {"error": "no candidates available"}
    result = evaluate_candidate_fn(cand.smiles, state.requirements)
    state.evaluations[cand.candidate_id] = result
    _update_best(state)
    return {"candidate_id": cand.candidate_id, "name": cand.name,
            "overall_score": result["overall_score"], "valid": result["valid"]}


def refine_candidate(state: ScientificState, ctx: ActionContext,
                      candidate_id: str | None = None, transform_name: str | None = None,
                      **kwargs) -> dict:
    base = _find_candidate(state, candidate_id) or _best_candidate(state)
    if base is None:
        return {"error": "no candidate to refine"}
    if transform_name is None:
        return {"error": "transform_name required",
                "available_transforms": [t["name"] for t in ctx.transforms]}
    new_cand = refiner.refine(base, transform_name, ctx.transforms)
    if new_cand is None:
        return {"error": f"transform '{transform_name}' produced no valid product from "
                          f"'{base.name}'"}
    state.candidates.append(new_cand)
    result = evaluate_candidate_fn(new_cand.smiles, state.requirements)
    state.evaluations[new_cand.candidate_id] = result
    prev_score = state.evaluations.get(base.candidate_id, {}).get("overall_score", 0.0)
    delta = result["overall_score"] - prev_score
    state.refinement_deltas.append(delta)
    _update_best(state)
    return {"new_candidate_id": new_cand.candidate_id, "parent": base.candidate_id,
            "transform": transform_name, "overall_score": result["overall_score"],
            "delta_vs_parent": delta}


def retrieve_additional_evidence(state: ScientificState, ctx: ActionContext, **kwargs) -> dict:
    return expand_protein_family(state, ctx, **kwargs)


def verify_hypothesis(state: ScientificState, ctx: ActionContext,
                       statement: str | None = None, **kwargs) -> dict:
    if not state.rbt_records:
        return {"error": "no biology evidence yet — run RETRIEVE_PROTEIN_EVIDENCE first"}
    statement = statement or (
        f"Resistance risk is primarily associated with mechanism(s): "
        f"{', '.join(sorted({r['mechanism_hint'] for r in state.rbt_records if not r.get('evidence_gap')}))}")
    hyp = new_hypothesis(statement)
    all_evidence_ids = [eid for r in state.rbt_records for eid in r.get("evidence_ids", [])]
    for eid in all_evidence_ids:
        ev = state.evidence_graph.evidence.get(eid)
        if ev is None:
            continue
        if ev.confidence >= 0.5:
            hyp.supporting_evidence.append(eid)
        else:
            hyp.contradicting_evidence.append(eid)
    hyp.confidence = (len(hyp.supporting_evidence) /
                       max(len(hyp.supporting_evidence) + len(hyp.contradicting_evidence), 1))
    hyp.update_status()
    state.hypotheses[hyp.hypothesis_id] = hyp
    return {"hypothesis_id": hyp.hypothesis_id, "status": hyp.status, "confidence": hyp.confidence}


def reassess_hypothesis(state: ScientificState, ctx: ActionContext,
                         hypothesis_id: str | None = None, **kwargs) -> dict:
    if hypothesis_id is None or hypothesis_id not in state.hypotheses:
        return {"error": "unknown hypothesis_id", "known": list(state.hypotheses)}
    hyp = state.hypotheses[hypothesis_id]
    hyp.update_status()
    return {"hypothesis_id": hyp.hypothesis_id, "status": hyp.status, "confidence": hyp.confidence}


def human_review(state: ScientificState, ctx: ActionContext, reason: str | None = None,
                  **kwargs) -> dict:
    state.status = "human_review"
    state.stopping_reason = reason or "agent_requested_human_review"
    return {"flag": "human_review_requested", "reason": state.stopping_reason}


def stop_action(state: ScientificState, ctx: ActionContext, reason: str | None = None,
                 **kwargs) -> dict:
    state.status = "stopped"
    state.stopping_reason = reason or "agent_stop"
    return {"stopped": True, "reason": state.stopping_reason}


ACTION_HANDLERS = {
    "RUN_AMR_ANALYSIS": run_amr_analysis,
    "RETRIEVE_PROTEIN_EVIDENCE": retrieve_protein_evidence,
    "EXPAND_PROTEIN_FAMILY": expand_protein_family,
    "VERIFY_HYPOTHESIS": verify_hypothesis,
    "BUILD_REQUIREMENTS": build_requirements,
    "EVALUATE_CANDIDATE": evaluate_candidate,
    "REFINE_CANDIDATE": refine_candidate,
    "RETRIEVE_ADDITIONAL_EVIDENCE": retrieve_additional_evidence,
    "REASSESS_HYPOTHESIS": reassess_hypothesis,
    "HUMAN_REVIEW": human_review,
    "STOP": stop_action,
}
