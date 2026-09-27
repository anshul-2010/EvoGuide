"""Resumable multi-genome x multi-condition experiment runner.

    python experiments/run_suite.py --genomes 8 --config configs/default.yaml

Safe to interrupt (session restart, network drop, machine sleep — all have happened during
this project's development) and re-launch as-is: already-completed (genome, condition) pairs
are skipped via experiments/runs/suite/<genome>_<condition>.json, and the expensive AMR-MoEGA
GA is additionally cache-gated per tools/amr_result_cache.py, so even a from-scratch rerun of
a partially-done suite does not repeat finished GA work.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.llm_backend import OllamaBackend  # noqa: E402
from config import load_config, repo_path  # noqa: E402
from evaluation.report import build_report  # noqa: E402
from experiments.conditions import CONDITIONS, NEEDS_LLM  # noqa: E402
from run import build_context  # noqa: E402


def select_genomes(config: dict, n: int) -> list[int]:
    antibiotic = config["amr"]["antibiotic"]
    labels_path = repo_path(config["amr"]["prepared_dir"], f"{antibiotic}_labels.csv")
    df = pd.read_csv(labels_path, index_col=0)
    positives = df[df[antibiotic] == 1]
    negatives = df[df[antibiotic] == 0]
    n_pos = n // 2
    n_neg = n - n_pos
    pos_rows = [df.index.get_loc(i) for i in positives.index[:n_pos]]
    neg_rows = [df.index.get_loc(i) for i in negatives.index[:n_neg]]
    return sorted(pos_rows + neg_rows)


def main(config: dict, n_genomes: int, explicit_genomes: list[int] | None = None) -> None:
    genomes = explicit_genomes if explicit_genomes is not None else select_genomes(config, n_genomes)
    antibiotic = config["amr"]["antibiotic"]
    out_dir = repo_path("experiments", "runs", "suite")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "suite_manifest.json"

    print(f"[run_suite] genomes: {genomes}")
    ctx = build_context(config)
    backend = OllamaBackend(config)
    backend_ok = backend.ping()
    if not backend_ok:
        print("[run_suite] WARNING: Ollama not reachable — generic_react/full_agent will be "
              "skipped until it is (the other 4 conditions do not need it)")

    completed: list[str] = []
    if manifest_path.exists():
        completed = json.loads(manifest_path.read_text(encoding="utf-8")).get("completed", [])

    for genome_row in genomes:
        for cond_name, cond_fn in CONDITIONS.items():
            key = f"{genome_row}_{cond_name}"
            out_path = out_dir / f"{key}.json"
            if out_path.exists():
                continue
            if cond_name in NEEDS_LLM and not backend_ok:
                print(f"[skip] {key}: Ollama not reachable")
                continue

            print(f"[run] genome={genome_row} condition={cond_name}")
            t0 = time.time()
            try:
                state = cond_fn(genome_row, antibiotic, config, ctx, backend)
            except Exception as e:  # noqa: BLE001
                print(f"[error] {key}: {type(e).__name__}: {e}")
                continue
            elapsed = time.time() - t0

            report = build_report(state)
            report["condition"] = cond_name
            report["wall_clock_seconds"] = elapsed
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)

            completed.append(key)
            manifest_path.write_text(
                json.dumps({"completed": completed, "genomes": genomes}, indent=2),
                encoding="utf-8")
            print(f"[done] {key} -> {out_path} ({elapsed:.1f}s, status={state.status})")

    print("[run_suite] all requested (genome, condition) pairs processed or skipped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--genomes", type=int, default=8,
                         help="stratified auto-selection count; ignored if --genome-rows is given")
    parser.add_argument("--genome-rows", default=None,
                         help="comma-separated explicit row indices, e.g. 7,8,10,12 "
                              "(overrides --genomes' stratified auto-selection)")
    parser.add_argument("--config", default=None)
    cli_args = parser.parse_args()

    cfg = load_config(cli_args.config) if cli_args.config else load_config()
    explicit = ([int(x) for x in cli_args.genome_rows.split(",")]
                if cli_args.genome_rows else None)
    main(cfg, cli_args.genomes, explicit)
