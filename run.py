"""EvoDesign-Agent CLI entrypoint.

    python run.py --config configs/default.yaml --genome-row 0 --antibiotic CIP --budget 20

Runs one Giessen-dataset row end-to-end through the full loop: AMR-MoEGA risk analysis ->
Risk-to-Biology Translator -> ProteinRAG evidence -> Candidate Requirement Specification ->
candidate evaluation/refinement, orchestrated by the LLM ReAct policy, and writes a
structured JSON report. See docs/overview.md for the architecture and
docs/design-principles.md for the scoping rules this implementation follows.
"""
from __future__ import annotations

import argparse

from agent import stopping
from agent.actions import ACTION_HANDLERS, ActionContext
from agent.llm_backend import OllamaBackend
from agent.policy_llm import LLMPolicy
from agent.state import ScientificState, TrajectoryRecord
from biology.requirements import CandidateRequirementSpecification
from biology.risk_translation import RiskToBiologyTranslator
from candidates.pool import load_seed_pool
from candidates.refiner import load_transforms
from config import load_config, repo_path
from evaluation.report import save_report
from tools.amr_moega_tool import AMRMoEGATool
from tools.protein_rag_tool import ProteinRAGTool


def build_context(config: dict) -> ActionContext:
    amr_tool = AMRMoEGATool(config)
    protein_rag_tool = ProteinRAGTool(config)
    rbt = RiskToBiologyTranslator(config, protein_rag_tool)
    crs = CandidateRequirementSpecification()  # deterministic priors; LLM-select can be wired in later
    transforms = load_transforms(repo_path(config["candidates"]["transform_library"]))
    seed_pool = load_seed_pool(repo_path(config["candidates"]["seed_pool_csv"]))
    return ActionContext(config, amr_tool, rbt, crs, transforms, seed_pool)


def execute_loop(state: ScientificState, ctx: ActionContext, policy, config: dict,
                  verbose: bool = True) -> ScientificState:
    """Shared ReAct-style execution loop used by every policy-driven condition (deterministic,
    fixed-closed-loop, generic-ReAct, full-agent — see experiments/conditions.py). `policy`
    only needs a `choose_action(state) -> {"action","args","rationale"}` method; LLMPolicy and
    agent.policy_deterministic's policies both satisfy this. Conditions 1/2 (static chemistry,
    biology-only) bypass this loop for their setup phase but may still use it afterward.
    """
    iteration = len(state.trajectory)
    while True:
        reason = stopping.check_stopping(state, config["agent"]["convergence_epsilon"])
        if reason:
            state.status = "stopped"
            state.stopping_reason = reason
            if verbose:
                print(f"[stop] {reason}")
            break

        decision = policy.choose_action(state)
        action, args, rationale = decision["action"], decision["args"], decision["rationale"]
        if verbose:
            print(f"[{iteration}] action={action} args={args} :: {rationale}")

        handler = ACTION_HANDLERS[action]
        try:
            observation = handler(state, ctx, **args)
        except Exception as e:  # noqa: BLE001
            observation = {"error": f"handler_exception: {e}"}

        state.budget_used += 1
        state.trajectory.append(TrajectoryRecord(iteration, action, args, observation, rationale))
        iteration += 1

        if state.status in ("stopped", "human_review"):
            break
    return state


def run(config: dict, antibiotic: str, genome_row: int, budget: int) -> tuple[ScientificState, str]:
    ctx = build_context(config)
    state = ScientificState(
        question=(f"What candidate-property refinements are warranted given the evolutionary "
                   f"resistance risk of Giessen-dataset genome row {genome_row} ({antibiotic})?"),
        antibiotic=antibiotic, genome_row=genome_row, budget_max=budget,
    )

    backend = OllamaBackend(config)
    if not backend.ping():
        host = config["agent"]["ollama"]["host"]
        raise RuntimeError(f"Ollama not reachable at {host} — is `ollama serve` running and "
                            f"has `{config['agent']['ollama']['model']}` been pulled?")
    policy = LLMPolicy(config, ctx.transforms, backend)

    execute_loop(state, ctx, policy, config)

    report_path = save_report(state, repo_path(config["run"]["output_dir"]))
    print(f"[done] status={state.status} reason={state.stopping_reason} "
          f"best_candidate={state.best_candidate_id} report={report_path}")
    return state, str(report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--antibiotic", default=None, help="CIP/CTX/CTZ/GEN; defaults to config value")
    parser.add_argument("--genome-row", type=int, default=0)
    parser.add_argument("--budget", type=int, default=None)
    cli_args = parser.parse_args()

    cfg = load_config(cli_args.config) if cli_args.config else load_config()
    ab = cli_args.antibiotic or cfg["amr"]["antibiotic"]
    bud = cli_args.budget or cfg["agent"]["budget_max_steps"]
    run(cfg, ab, cli_args.genome_row, bud)
