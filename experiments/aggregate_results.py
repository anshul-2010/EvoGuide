"""Aggregates every experiments/runs/suite/*.json report into one summary CSV.

    python experiments/aggregate_results.py

Covers a subset of docs/experiments.md's 9 evaluation dimensions: candidate quality (best
score), tool efficiency (budget used, wall-clock), evolutionary awareness (resistance risk,
n_requirements), and a coarse robustness signal (n_evidence_gaps). Calibration (ECE/Brier),
the full failure-mode taxonomy, and perturbation-based robustness testing are NOT computed
here — deferred, not silently dropped (see the plan this harness was built from).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import repo_path  # noqa: E402


def summarize_report(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        r = json.load(f)

    best_id = r.get("best_candidate_id")
    best_score = None
    if best_id and best_id in r.get("evaluations", {}):
        best_score = r["evaluations"][best_id].get("overall_score")

    erp = r.get("risk_profile")
    return {
        "genome_row": r.get("genome_row"),
        "condition": r.get("condition"),
        "antibiotic": r.get("antibiotic"),
        "status": r.get("status"),
        "stopping_reason": r.get("stopping_reason"),
        "resistance_risk": erp.get("resistance_risk") if erp else None,
        "held_out_val_auc": erp.get("held_out_val_auc") if erp else None,
        "n_requirements": len(r.get("requirements", [])),
        "n_candidates": len(r.get("candidates", [])),
        "n_evaluations": len(r.get("evaluations", {})),
        "best_score": best_score,
        "n_evidence": len(r.get("biological_evidence", {}).get("evidence", [])),
        "n_evidence_gaps": sum(1 for rec in r.get("rbt_records", []) if rec.get("evidence_gap")),
        "budget_used": r.get("budget_used"),
        "budget_max": r.get("budget_max"),
        "n_trajectory_steps": len(r.get("trajectory", [])),
        "wall_clock_seconds": r.get("wall_clock_seconds"),
    }


def main() -> None:
    suite_dir = repo_path("experiments", "runs", "suite")
    rows = []
    for path in sorted(suite_dir.glob("*_*.json")):
        if path.name == "suite_manifest.json":
            continue
        rows.append(summarize_report(path))

    if not rows:
        print(f"[aggregate_results] no reports found under {suite_dir}")
        return

    df = pd.DataFrame(rows).sort_values(["genome_row", "condition"])
    out_path = suite_dir / "summary.csv"
    df.to_csv(out_path, index=False)
    print(f"[aggregate_results] {len(df)} rows -> {out_path}")

    print("\nPer-condition means (across genomes):")
    numeric_cols = ["resistance_risk", "n_requirements", "best_score", "n_evidence",
                     "n_evidence_gaps", "budget_used", "wall_clock_seconds"]
    print(df.groupby("condition")[numeric_cols].mean(numeric_only=True).round(3).to_string())


if __name__ == "__main__":
    main()
