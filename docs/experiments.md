# Experimental Design for EvoDesign-Agent

Research questions, baselines, ablations, and evaluation dimensions for [overview.md](overview.md), scoped per [design-principles.md](design-principles.md).

## Four central research questions (structure the paper around these)

- **RQ1 (evolutionary grounding):** Does incorporating evolutionary AMR risk improve identification/prioritization of biologically relevant targets or candidate requirements?
- **RQ2 (tool-augmented reasoning):** Can the agent dynamically select between evolutionary analysis, biological retrieval, and candidate evaluation based on unresolved evidence gaps?
- **RQ3 (closed-loop refinement):** Does adaptive evidence-driven refinement improve candidate quality more efficiently than fixed/non-agentic refinement?
- **RQ4 (reliability):** Can the agent recognize uncertainty, conflicting evidence, and tool failures while maintaining traceable reasoning?

## Six baselines

1. **Static chemistry** — candidate generation → evaluation only, no AMR-MoEGA.
2. **Biology-only** — ProteinRAG → requirements → evaluation, no evolutionary info.
3. **Evolution + fixed retrieval** — AMR-MoEGA → ProteinRAG → requirements → evaluation, no agent.
4. **Fixed closed loop** — same pipeline + fixed N refinement iterations regardless of state.
5. **Generic ReAct** — LLM has all tools but without the evidence-gap policy (tests whether ReAct alone explains gains).
6. **EvoDesign-Agent (full method)** — adaptive bidirectional loop: AMR-MoEGA ↔ Agent ↔ ProteinRAG ↔ Requirements ↔ Refinement ↔ Evaluation.

Main ablation table cross-references these on 4 axes: Evolution / ProteinRAG / Adaptive planning / Closed-loop.

## Key standalone experiments

- **Removing the agent** (fixed pipeline vs. adaptive agent) — central, not an appendix ablation.
- **Removing evolution** (ProteinRAG→candidate vs. AMR→ProteinRAG→candidate) — most scientifically important single question: does evolutionary info actually change downstream design decisions?
- **Removing ProteinRAG** (AMR→candidate vs. AMR→ProteinRAG→candidate) — tests whether evolutionary risk alone is insufficient.
- **Counterfactual evidence** — same genome, perturb/remove strongest evolutionary signal, or add contradictory evidence; check if agent adapts retrieval strategy and requirements accordingly.
- **Tool failure injection** — ProteinRAG unavailable, AMR uncertainty spiked, evaluator unavailable, irrelevant retrieval results; measure failure detection, recovery rate, hallucination rate, unjustified continuation.
- **Budget scaling** — fixed budgets of 5/10/20/40 tool calls; question is NOT "does more compute help" but "does adaptive allocation beat fixed allocation under equal budget."
- **Backbone comparison** — general LLM vs. biomedical/clinical LLM vs. smaller local model vs. stronger API model, to test whether backbone specialization matters once structured tools/evidence exist.
- **Human-in-the-loop checkpoint** — agent recommendation → human approve/reject/request-more-evidence → agent continues; measure reliability improvement from intervention.

## Evaluation dimensions (9 total)

1. **Evolutionary awareness** — Risk Coverage (risks incorporated / risks identified), risk-to-requirement consistency, risk-aware prioritization.
2. **Retrieval quality** — standard ProteinRAG metrics (Recall@K, MRR, MAP, nDCG), stratified by close/intermediate/remote homology tier per [references/proteinrag.md](../references/proteinrag.md), plus agent-specific: Retrieval Usefulness (retrieved items actually used / retrieved), evidence grounding rate, evidence redundancy. ProteinRAG's own reported numbers are specific to its indexed corpus (Pfam-36.0, 84,250 proteins) — re-measure rather than reuse if EvoDesign-Agent indexes a different/larger corpus.
3. **Scientific reasoning** — hypothesis support accuracy vs. expert label, evidence consistency, contradiction detection, Unsupported-Claim Rate (claims w/o evidence / total claims).
4. **Candidate quality** — property/constraint satisfaction, validity, diversity, novelty, developability scores, iterations to convergence, ΔS per iteration, Efficiency = ΔS / tool_calls.
5. **Closed-loop improvement** — candidate quality vs. iteration count, compared across fixed pipeline / generic ReAct / EvoDesign-Agent.
6. **Calibration** — ECE, Brier score, reliability diagrams over risk/hypothesis/requirement confidences; key question: "when the agent claims sufficient evidence, does it actually have sufficient evidence?"
7. **Tool efficiency** — tool/retrieval/evaluation/refinement call counts, tokens, latency, cost; scientific utility vs. tool budget curve.
8. **Robustness** — perturbed/removed/injected-irrelevant/contradictory evidence, reduced AMR confidence — does the agent flag uncertainty/conflict rather than arbitrarily picking a side?
9. **Failure-mode taxonomy** (F1–F10): unsupported biological claim, irrelevant retrieval, evidence misinterpretation, incorrect risk→requirement mapping, refinement without evidence, repeated tool calls, premature stopping, unnecessary tool calls, failure to recognize uncertainty, contradiction ignored — trajectories get classified against this taxonomy.

## Artifacts to log

- Every case produces a **structured final report** (risk profile, biological evidence, hypotheses w/ status, requirements, candidate + refinement history, evaluation, uncertainty, evidence provenance, stopping reason) — not free-text prose.
- Every agent step logged as a trajectory record: `{iteration, state, action, observation, hypothesis_update, confidence}` — enables trajectory visualization (evidence-graph-style figures) and the failure-mode analysis above.

## Possible standalone contribution

A reusable **Evolution-to-Design Agent Benchmark** with 7 sub-tasks (risk identification, risk→protein retrieval, evidence synthesis, requirement generation, candidate selection, closed-loop refinement, full trajectory evaluation) — built from existing ground truth per case (AMR-MoEGA labels, known protein families/homologs, existing molecular property benchmarks), not artificially constructed.
