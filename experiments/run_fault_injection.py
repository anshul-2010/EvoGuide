"""Tool failure injection + contradictory evidence (docs/experiments.md RQ4/reliability):
wraps the real ProteinRAGTool with a configurable fault and checks whether full_agent detects
it (escalates to HUMAN_REVIEW, flags uncertainty) rather than confidently proceeding as if
nothing were wrong.

Three fault modes:
  outage       - ProteinRAG always returns ToolResult(ok=False) — simulates the service being down.
  irrelevant   - ProteinRAG runs for real, but against a deliberately unrelated query sequence,
                 so results are real retrieval output with nothing to do with the actual gene.
  contradictory - ProteinRAG runs for real against the real query, but every hit's s_final
                 (confidence) is inverted (1 - s_final) before being handed to the RBT —
                 simulates systematically unreliable/contradictory evidence quality rather
                 than irrelevant content, a different failure than 'irrelevant'.

    python experiments/run_fault_injection.py --modes outage,irrelevant,contradictory --genome-rows 1,4,5,6,7

Uses a SEPARATE ActionContext per fault mode (not the shared suite context) so the faulty
ProteinRAG wrapper never leaks into other experiments.
"""
from __future__ import annotations

import argparse
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
from tools.base import ToolResult  # noqa: E402


class FaultyProteinRAGTool:
    name = "protein_rag"

    # Deliberately unrelated, generic real-amino-acid filler — used for 'irrelevant' so the
    # tool genuinely runs (real FAISS/BM25/reranker work), just against a query that has
    # nothing to do with the gene actually being investigated.
    _GARBAGE_SEQ = "MAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"

    def __init__(self, real_tool, mode: str):
        self.real_tool = real_tool
        self.mode = mode

    def run(self, seq: str | None = None, domains=None, top_k=None, **kwargs) -> ToolResult:
        if self.mode == "outage":
            return ToolResult(self.name, ok=False,
                               error="Simulated ProteinRAG outage (fault injection)")

        if self.mode == "irrelevant":
            return self.real_tool.run(seq=self._GARBAGE_SEQ, domains=domains, top_k=top_k)

        if self.mode == "contradictory":
            result = self.real_tool.run(seq=seq, domains=domains, top_k=top_k)
            if not result.ok:
                return result
            corrupted = []
            for hit in result.output["results"]:
                hit = dict(hit)
                hit["s_final"] = 1.0 - hit["s_final"]
                corrupted.append(hit)
            return ToolResult(result.tool_name, ok=True, output={"results": corrupted})

        raise ValueError(f"unknown fault mode: {self.mode}")


def main(config: dict, modes: list[str], genome_rows: list[int]) -> None:
    antibiotic = config["amr"]["antibiotic"]
    out_dir = repo_path("experiments", "runs", "fault_injection")
    out_dir.mkdir(parents=True, exist_ok=True)

    backend = OllamaBackend(config)
    if not backend.ping():
        print("[run_fault_injection] Ollama not reachable — aborting")
        return

    for mode in modes:
        # Fresh context per fault mode — the faulty wrapper must never leak into other runs.
        ctx = build_context(config)
        ctx.rbt.protein_rag = FaultyProteinRAGTool(ctx.rbt.protein_rag, mode)

        for genome_row in genome_rows:
            key = f"{genome_row}_{mode}"
            out_path = out_dir / f"{key}.json"
            if out_path.exists():
                continue
            print(f"[run] genome={genome_row} fault_mode={mode}")
            t0 = time.time()
            try:
                state = run_full_agent(genome_row, antibiotic, config, ctx, backend)
            except Exception as e:  # noqa: BLE001
                print(f"[error] {key}: {type(e).__name__}: {e}")
                continue
            elapsed = time.time() - t0

            report = build_report(state)
            report["condition"] = "full_agent"
            report["fault_mode"] = mode
            report["wall_clock_seconds"] = elapsed
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)
            print(f"[done] {key} -> {out_path} ({elapsed:.1f}s, status={state.status}, "
                  f"reason={state.stopping_reason})")

    print("[run_fault_injection] all requested (genome, mode) pairs processed or skipped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", required=True, help="comma-separated: outage,irrelevant,contradictory")
    parser.add_argument("--genome-rows", required=True)
    parser.add_argument("--config", default=None)
    cli_args = parser.parse_args()

    cfg = load_config(cli_args.config) if cli_args.config else load_config()
    main(cfg, cli_args.modes.split(","), [int(x) for x in cli_args.genome_rows.split(",")])
