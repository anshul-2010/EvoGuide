"""Tier-0 analyses: calibration, failure-mode scan, and efficiency curves computed entirely
from experiments/runs/suite/*.json reports already on disk — no new experiment runs.

    python experiments/analyze_pilot.py

Writes experiments/runs/suite/calibration.csv, failure_modes.csv, efficiency.csv, and prints
a summary. See docs/results.md for how these get folded into the paper write-up.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import repo_path  # noqa: E402

SUITE_DIR = repo_path("experiments", "runs", "suite")


def load_reports() -> list[dict]:
    reports = []
    for path in sorted(SUITE_DIR.glob("*_*.json")):
        if path.name == "suite_manifest.json":
            continue
        with open(path, encoding="utf-8") as f:
            reports.append(json.load(f))
    return reports


# --- Calibration -------------------------------------------------------------------------

def calibration_analysis(reports: list[dict]) -> pd.DataFrame:
    """Brier score + reliability table for AMR-MoEGA's resistance_risk predictions. One row
    per genome (dedup across conditions — the ERP is genome-specific, not condition-specific).
    """
    rows = {}
    for r in reports:
        erp = r.get("risk_profile")
        if not erp:
            continue
        rows[r["genome_row"]] = {
            "genome_row": r["genome_row"],
            "resistance_risk": erp["resistance_risk"],
            "true_label": erp["true_label"],
            "held_out_val_auc": erp["held_out_val_auc"],
        }
    df = pd.DataFrame(rows.values()).sort_values("genome_row")
    df["squared_error"] = (df["resistance_risk"] - df["true_label"]) ** 2
    df["predicted_label"] = (df["resistance_risk"] >= 0.5).astype(int)
    df["correct"] = (df["predicted_label"] == df["true_label"]).astype(int)
    return df


def reliability_bins(df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    bins = np.linspace(0, 1, n_bins + 1)
    df = df.copy()
    df["bin"] = pd.cut(df["resistance_risk"], bins, include_lowest=True)
    return df.groupby("bin", observed=True).agg(
        n=("true_label", "size"),
        mean_predicted=("resistance_risk", "mean"),
        mean_observed=("true_label", "mean"),
    )


# --- Failure-mode scan (programmatically detectable subset only) -------------------------
# docs/experiments.md defines 10 failure categories (F1-F10). Only a subset is cheaply
# detectable from trajectory logs without a semantic/LLM-judge pass:
#   - repeated_identical_call: same (action, args) called consecutively more than once
#   - precondition_violation: an action attempted whose observation contains an unmet-
#     precondition error (i.e. the policy tried something it wasn't ready for)
#   - refine_without_diversifying: REFINE_CANDIDATE repeated with the SAME transform_name
#     after a non-positive delta, instead of trying a different transform (genome 4's case)
# The remaining categories (unsupported biological claim, evidence misinterpretation,
# incorrect risk-to-requirement mapping, contradiction ignored, failure to recognize
# uncertainty) require semantic judgment this script does not attempt — left for a
# manual or LLM-judge pass, not silently claimed as covered.

def failure_mode_scan(reports: list[dict]) -> pd.DataFrame:
    rows = []
    for r in reports:
        traj = r.get("trajectory", [])
        genome, condition = r.get("genome_row"), r.get("condition")
        repeated_identical = 0
        precondition_violations = 0
        refine_same_transform_after_negative = 0
        last_key = None
        last_refine_transform = None
        last_refine_delta = None
        for t in traj:
            key = (t["action"], json.dumps(t["action_args"], sort_keys=True))
            if key == last_key:
                repeated_identical += 1
            last_key = key

            obs = t.get("observation", {})
            if isinstance(obs, dict) and isinstance(obs.get("error"), str):
                if "first" in obs["error"] or "required" in obs["error"]:
                    precondition_violations += 1

            if t["action"] == "REFINE_CANDIDATE":
                transform = t["action_args"].get("transform_name")
                if (transform is not None and transform == last_refine_transform
                        and last_refine_delta is not None and last_refine_delta <= 0):
                    refine_same_transform_after_negative += 1
                last_refine_transform = transform
                last_refine_delta = obs.get("delta_vs_parent") if isinstance(obs, dict) else None

        rows.append({
            "genome_row": genome, "condition": condition, "n_steps": len(traj),
            "repeated_identical_calls": repeated_identical,
            "precondition_violations": precondition_violations,
            "refine_without_diversifying": refine_same_transform_after_negative,
            "final_status": r.get("status"), "stopping_reason": r.get("stopping_reason"),
        })
    return pd.DataFrame(rows).sort_values(["genome_row", "condition"])


# --- Efficiency ----------------------------------------------------------------------------

def efficiency_analysis(reports: list[dict]) -> pd.DataFrame:
    rows = []
    for r in reports:
        best_id = r.get("best_candidate_id")
        best_score = None
        if best_id and best_id in r.get("evaluations", {}):
            best_score = r["evaluations"][best_id].get("overall_score")
        budget_used = r.get("budget_used", 0)
        rows.append({
            "genome_row": r.get("genome_row"), "condition": r.get("condition"),
            "best_score": best_score, "budget_used": budget_used,
            "efficiency_score_per_step": (best_score / budget_used)
                                          if best_score is not None and budget_used else None,
            "wall_clock_seconds": r.get("wall_clock_seconds"),
        })
    return pd.DataFrame(rows).sort_values(["genome_row", "condition"])


def main() -> None:
    reports = load_reports()
    print(f"[analyze_pilot] loaded {len(reports)} reports")

    calib = calibration_analysis(reports)
    calib.to_csv(SUITE_DIR / "calibration.csv", index=False)
    brier = calib["squared_error"].mean()
    accuracy = calib["correct"].mean()
    print(f"\n=== Calibration (n={len(calib)} genomes) ===")
    print(f"Brier score: {brier:.4f}  (0=perfect, 0.25=naive-always-0.5 baseline)")
    print(f"Threshold accuracy (risk>=0.5 vs true label): {accuracy:.3f}")
    print(reliability_bins(calib).round(3).to_string())

    fm = failure_mode_scan(reports)
    fm.to_csv(SUITE_DIR / "failure_modes.csv", index=False)
    print(f"\n=== Failure-mode scan (n={len(fm)} runs) ===")
    print(fm[["repeated_identical_calls", "precondition_violations",
              "refine_without_diversifying"]].sum().to_string())
    flagged = fm[(fm["repeated_identical_calls"] > 0) | (fm["refine_without_diversifying"] > 0)]
    if len(flagged):
        print("\nRuns with at least one flagged behavior:")
        print(flagged.to_string(index=False))

    eff = efficiency_analysis(reports)
    eff.to_csv(SUITE_DIR / "efficiency.csv", index=False)
    print(f"\n=== Efficiency (mean per condition) ===")
    print(eff.groupby("condition")[["best_score", "budget_used", "efficiency_score_per_step",
                                     "wall_clock_seconds"]].mean(numeric_only=True).round(4).to_string())


if __name__ == "__main__":
    main()
