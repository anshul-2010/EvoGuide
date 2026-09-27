"""Final structured report, per docs/experiments.md's 'Artifacts to log': risk profile,
biological evidence, hypotheses w/ status, requirements, candidate + refinement history,
evaluation, uncertainty, evidence provenance, stopping reason — never free-text prose."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from agent.state import ScientificState


def build_report(state: ScientificState) -> dict:
    return {
        "question": state.question,
        "antibiotic": state.antibiotic,
        "genome_row": state.genome_row,
        "status": state.status,
        "stopping_reason": state.stopping_reason,
        "risk_profile": state.erp,
        "biological_evidence": state.evidence_graph.to_dict(),
        "rbt_records": state.rbt_records,
        "hypotheses": [dataclasses.asdict(h) for h in state.hypotheses.values()],
        "requirements": [dataclasses.asdict(r) for r in state.requirements],
        "candidates": [dataclasses.asdict(c) for c in state.candidates],
        "evaluations": state.evaluations,
        "best_candidate_id": state.best_candidate_id,
        "uncertainty_flags": state.uncertainty_flags,
        "refinement_deltas": state.refinement_deltas,
        "budget_used": state.budget_used,
        "budget_max": state.budget_max,
        "trajectory": [dataclasses.asdict(t) for t in state.trajectory],
    }


def save_report(state: ScientificState, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"run_{state.antibiotic}_row{state.genome_row}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(build_report(state), f, indent=2, default=str)
    return path
