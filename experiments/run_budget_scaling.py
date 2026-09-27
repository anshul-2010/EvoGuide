"""Budget scaling (docs/experiments.md): rerun full_agent at different budget_max_steps values
on the same genomes. The question is NOT "does more budget help" but "does the adaptive policy
use its budget efficiently, or does it need the full 20 steps regardless" — compare against the
existing budget=20 runs already in experiments/runs/suite/.

    python experiments/run_budget_scaling.py --budgets 5,10,40 --genome-rows 1,4,5,6,7

Reuses the cached AMR-MoEGA result and built ProteinRAG index for each genome.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.llm_backend import OllamaBackend  # noqa: E402
from config import load_config, repo_path  # noqa: E402
from evaluation.report import build_report  # noqa: E402
from experiments.conditions import run_full_agent  # noqa: E402
from run import build_context  # noqa: E402


def main(config: dict, budgets: list[int], genome_rows: list[int]) -> None:
    antibiotic = config["amr"]["antibiotic"]
    out_dir = repo_path("experiments", "runs", "budget_scaling")
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = build_context(config)

    backend = OllamaBackend(config)
    if not backend.ping():
        print("[run_budget_scaling] Ollama not reachable — aborting")
        return

    for budget in budgets:
        cfg = copy.deepcopy(config)
        cfg["agent"]["budget_max_steps"] = budget

        for genome_row in genome_rows:
            key = f"{genome_row}_budget{budget}"
            out_path = out_dir / f"{key}.json"
            if out_path.exists():
                continue
            print(f"[run] genome={genome_row} budget={budget}")
            t0 = time.time()
            try:
                state = run_full_agent(genome_row, antibiotic, cfg, ctx, backend)
            except Exception as e:  # noqa: BLE001
                print(f"[error] {key}: {type(e).__name__}: {e}")
                continue
            elapsed = time.time() - t0

            report = build_report(state)
            report["condition"] = "full_agent"
            report["budget_setting"] = budget
            report["wall_clock_seconds"] = elapsed
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)
            print(f"[done] {key} -> {out_path} ({elapsed:.1f}s, "
                  f"budget_used={state.budget_used}, status={state.status})")

    print("[run_budget_scaling] all requested (genome, budget) pairs processed or skipped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--budgets", required=True, help="comma-separated budget_max_steps values")
    parser.add_argument("--genome-rows", required=True, help="comma-separated genome row indices")
    parser.add_argument("--config", default=None)
    cli_args = parser.parse_args()

    cfg = load_config(cli_args.config) if cli_args.config else load_config()
    main(cfg, [int(x) for x in cli_args.budgets.split(",")],
         [int(x) for x in cli_args.genome_rows.split(",")])
