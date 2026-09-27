"""The 6 baseline/ablation conditions from docs/experiments.md, each a function
`(genome_row, antibiotic, config, ctx, backend=None) -> ScientificState`. All 6 share the same
ActionContext/action handlers (agent/actions.py) — only the decision-making policy or the
initial evidence-gathering step differs, per docs/design-principles.md's isolation rule.

IMPORTANT: experiments/run_suite.py reuses ONE ActionContext across every (genome, condition)
run to avoid reloading the ProteinRAG index/embedder each time. ctx.rbt.graph is the one piece
of genuinely mutable, run-scoped state on that shared context (every action handler that
retrieves evidence writes into it) — every condition function here MUST reset it via
_fresh_evidence_graph(ctx) before use, or evidence silently accumulates across unrelated runs.
"""
from __future__ import annotations

from agent.actions import ACTION_HANDLERS, ActionContext
from agent.llm_backend import OllamaBackend
from agent.policy_deterministic import BiologyOnlyPolicy, DeterministicPolicy, FixedClosedLoopPolicy
from agent.policy_llm import LLMPolicy, SYSTEM_PROMPT_GENERIC
from agent.state import ScientificState, TrajectoryRecord
from biology.evidence_graph import EvidenceGraph
from biology.requirements import build_static_requirements
from run import execute_loop


def _new_state(genome_row: int, antibiotic: str, config: dict, condition: str) -> ScientificState:
    return ScientificState(
        question=(f"[{condition}] candidate-property refinements for Giessen-dataset genome "
                   f"row {genome_row} ({antibiotic})?"),
        antibiotic=antibiotic, genome_row=genome_row,
        budget_max=config["agent"]["budget_max_steps"],
    )


def _fresh_evidence_graph(ctx: ActionContext) -> None:
    ctx.rbt.graph = EvidenceGraph()


def run_static_chemistry(genome_row: int, antibiotic: str, config: dict,
                          ctx: ActionContext, backend: OllamaBackend | None = None) -> ScientificState:
    """Condition 1: no AMR-MoEGA, no ProteinRAG — a fixed uniform requirement set, one
    evaluation of the seed pool's first candidate, no refinement."""
    state = _new_state(genome_row, antibiotic, config, "static_chemistry")
    state.requirements = build_static_requirements()
    state.candidates = list(ctx.seed_pool)
    observation = ACTION_HANDLERS["EVALUATE_CANDIDATE"](state, ctx)
    state.budget_used = 1
    state.trajectory.append(TrajectoryRecord(0, "EVALUATE_CANDIDATE", {}, observation,
                                              "static baseline: uniform requirements, no evidence"))
    state.status = "stopped"
    state.stopping_reason = "static_chemistry_complete"
    return state


def run_biology_only(genome_row: int, antibiotic: str, config: dict,
                      ctx: ActionContext, backend: OllamaBackend | None = None) -> ScientificState:
    """Condition 2: ProteinRAG evidence with no risk gating (queries every resolved gene
    fixture unconditionally), then the deterministic policy from BUILD_REQUIREMENTS onward."""
    _fresh_evidence_graph(ctx)
    state = _new_state(genome_row, antibiotic, config, "biology_only")
    records = ctx.rbt.translate_generic()
    state.rbt_records = records
    state.evidence_graph = ctx.rbt.graph
    n_gap = sum(1 for r in records if r.get("evidence_gap"))
    observation = {"n_records": len(records), "n_evidence_gaps": n_gap}
    state.trajectory.append(TrajectoryRecord(
        0, "RETRIEVE_PROTEIN_EVIDENCE", {}, observation,
        "biology-only: generic evidence retrieval, no evolutionary risk gating"))
    state.budget_used = 1
    return execute_loop(state, ctx, BiologyOnlyPolicy(), config, verbose=False)


def run_evolution_fixed_retrieval(genome_row: int, antibiotic: str, config: dict,
                                   ctx: ActionContext, backend: OllamaBackend | None = None) -> ScientificState:
    """Condition 3: real AMR-MoEGA -> RBT -> ProteinRAG -> requirements -> one evaluation, via
    the fixed deterministic policy — no adaptive replanning, no refinement loop."""
    _fresh_evidence_graph(ctx)
    state = _new_state(genome_row, antibiotic, config, "evolution_fixed_retrieval")
    return execute_loop(state, ctx, DeterministicPolicy(), config, verbose=False)


def run_no_protein_rag(genome_row: int, antibiotic: str, config: dict,
                        ctx: ActionContext, backend: OllamaBackend | None = None) -> ScientificState:
    """Extra ablation (docs/experiments.md's 'removing ProteinRAG' standalone experiment,
    previously missing from the condition set): real AMR-MoEGA + real risk gating + heuristic
    mechanism mapping, but ProteinRAG is never queried — isolates whether real retrieved
    protein evidence adds anything beyond a risk score and a guessed mechanism label."""
    _fresh_evidence_graph(ctx)
    state = _new_state(genome_row, antibiotic, config, "no_protein_rag")

    obs0 = ACTION_HANDLERS["RUN_AMR_ANALYSIS"](state, ctx)
    state.trajectory.append(TrajectoryRecord(
        0, "RUN_AMR_ANALYSIS", {}, obs0,
        "no_protein_rag ablation: real AMR-MoEGA, ProteinRAG intentionally skipped"))
    state.budget_used = 1
    if state.erp is None:
        state.status, state.stopping_reason = "stopped", "amr_analysis_failed"
        return state

    records = ctx.rbt.translate_no_retrieval(state.erp)
    state.rbt_records = records
    state.evidence_graph = ctx.rbt.graph
    if not records:
        state.below_risk_threshold = True
    obs1 = {"n_records": len(records), "n_evidence_gaps": 0,
            "note": "ProteinRAG intentionally skipped for this ablation"}
    state.trajectory.append(TrajectoryRecord(
        1, "RETRIEVE_PROTEIN_EVIDENCE", {}, obs1,
        "no_protein_rag ablation: heuristic mechanism mapping only, no real evidence"))
    state.budget_used = 2

    return execute_loop(state, ctx, DeterministicPolicy(), config, verbose=False)


def run_fixed_closed_loop(genome_row: int, antibiotic: str, config: dict,
                           ctx: ActionContext, backend: OllamaBackend | None = None) -> ScientificState:
    """Condition 4: same as condition 3, plus a fixed refinement pass applying every transform
    once, in file order, regardless of observed improvement."""
    _fresh_evidence_graph(ctx)
    state = _new_state(genome_row, antibiotic, config, "fixed_closed_loop")
    return execute_loop(state, ctx, FixedClosedLoopPolicy(ctx.transforms), config, verbose=False)


def run_generic_react(genome_row: int, antibiotic: str, config: dict,
                       ctx: ActionContext, backend: OllamaBackend) -> ScientificState:
    """Condition 5: LLM has the full action space but a minimal prompt (no evidence-gap
    guidance, no ordering hints, no retry nudge) — tests whether ReAct alone explains gains."""
    _fresh_evidence_graph(ctx)
    state = _new_state(genome_row, antibiotic, config, "generic_react")
    policy = LLMPolicy(config, ctx.transforms, backend, system_prompt_template=SYSTEM_PROMPT_GENERIC)
    return execute_loop(state, ctx, policy, config, verbose=False)


def run_full_agent(genome_row: int, antibiotic: str, config: dict,
                    ctx: ActionContext, backend: OllamaBackend) -> ScientificState:
    """Condition 6: EvoDesign-Agent — the full evidence-gap-aware LLM policy."""
    _fresh_evidence_graph(ctx)
    state = _new_state(genome_row, antibiotic, config, "full_agent")
    policy = LLMPolicy(config, ctx.transforms, backend)
    return execute_loop(state, ctx, policy, config, verbose=False)


CONDITIONS = {
    "static_chemistry": run_static_chemistry,
    "biology_only": run_biology_only,
    "evolution_fixed_retrieval": run_evolution_fixed_retrieval,
    "no_protein_rag": run_no_protein_rag,
    "fixed_closed_loop": run_fixed_closed_loop,
    "generic_react": run_generic_react,
    "full_agent": run_full_agent,
}

# Conditions 5 and 6 need a live Ollama backend; the other 4 are pure Python, no LLM involved.
NEEDS_LLM = {"generic_react", "full_agent"}
