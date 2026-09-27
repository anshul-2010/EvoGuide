"""Backbone comparison (docs/experiments.md): rerun full_agent with different local LLM sizes
on the same genomes, holding everything else (tools, evidence, requirements ontology) constant
— tests whether the model's capability, not just the evidence-gap-aware policy design, drives
the observed gains. Motivated directly by this project's own finding that qwen2.5:1.5b-instruct
could not reliably reach a scored candidate while qwen2.5:3b-instruct could (see docs/results.md).

    python experiments/run_backbone_comparison.py --models qwen2.5:1.5b-instruct,qwen2.5:3b-instruct --genome-rows 1,4,5,6,7

Reuses the cached AMR-MoEGA result and built ProteinRAG index for each genome — only the LLM
policy calls themselves are repeated per model, so this is cheap relative to a fresh genome run.
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


def main(config: dict, models: list[str], genome_rows: list[int]) -> None:
    antibiotic = config["amr"]["antibiotic"]
    out_dir = repo_path("experiments", "runs", "backbone")
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = build_context(config)

    for model in models:
        cfg = copy.deepcopy(config)
        cfg["agent"]["ollama"]["model"] = model
        backend = OllamaBackend(cfg)
        if not backend.ping():
            print(f"[run_backbone_comparison] SKIP model={model}: Ollama not reachable")
            continue

        for genome_row in genome_rows:
            key = f"{genome_row}_{model.replace(':', '_').replace('.', 'p')}"
            out_path = out_dir / f"{key}.json"
            if out_path.exists():
                continue
            print(f"[run] genome={genome_row} model={model}")
            t0 = time.time()
            try:
                state = run_full_agent(genome_row, antibiotic, cfg, ctx, backend)
            except Exception as e:  # noqa: BLE001
                print(f"[error] {key}: {type(e).__name__}: {e}")
                continue
            elapsed = time.time() - t0

            report = build_report(state)
            report["condition"] = "full_agent"
            report["model"] = model
            report["wall_clock_seconds"] = elapsed
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)
            print(f"[done] {key} -> {out_path} ({elapsed:.1f}s, status={state.status})")

    print("[run_backbone_comparison] all requested (genome, model) pairs processed or skipped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True, help="comma-separated Ollama model names")
    parser.add_argument("--genome-rows", required=True, help="comma-separated genome row indices")
    parser.add_argument("--config", default=None)
    cli_args = parser.parse_args()

    cfg = load_config(cli_args.config) if cli_args.config else load_config()
    main(cfg, cli_args.models.split(","), [int(x) for x in cli_args.genome_rows.split(",")])
