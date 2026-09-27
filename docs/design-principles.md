# Design Principles for EvoDesign-Agent

Standing scoping decisions for [overview.md](overview.md) — treat these as durable constraints when designing, implementing, or writing up EvoDesign-Agent, not just one-off suggestions.

## Don't overclaim beyond what components actually do

- Only implement/claim AMR-MoEGA fields (mutation rates, evolutionary trajectories, time-to-AMR) if the *existing* AMR-MoEGA methodology actually estimates them.
  **Resolved** (see [references/amr-moega.md](../references/amr-moega.md), read from arXiv:2511.12223): AMR-MoEGA produces a per-genome resistance probability, GA-generation-indexed population fitness/diversity trajectories, SnpEff-annotated mutation clusters tied to real gene mechanisms (efflux regulators, porins, β-lactamases), and a discrete Markov transition matrix *between those mechanism clusters* — but explicitly does **not** produce continuous/calendar time-to-AMR predictions, and "mutation rate" is a GA hyperparameter, not an estimated biological quantity. Any temporal/hazard-rate framing beyond GA-generation units would be an overclaim per the paper's own stated limitations.
- Don't claim "the agent discovers new antibiotics" — say "computational candidate prioritization" / "closed-loop computational candidate refinement" unless experimentally validated.
- Don't claim "ProteinRAG discovers drug targets" — say "retrieves and contextualizes relevant protein-family evidence."
  **Resolved** (see [references/proteinrag.md](../references/proteinrag.md)): ProteinRAG is a 5-stage hybrid retrieval pipeline (EASR sparse + ESM-2 dense + score fusion + ColBERT-style late interaction + Gradient Boosting rerank) that returns a ranked homolog list with decomposable evidence scores and a close/intermediate/remote homology tier — ground truth is Pfam family/clan co-membership, not experimental function/structure/druggability. It does not predict structure or function, never generates sequences, and its validated numbers are specific to its indexed Pfam-36.0 corpus (84,250 proteins) — don't imply broader coverage (e.g. full UniProt) or functional/causal confidence without separate justification.
- Don't claim "ReAct performs drug design" — ReAct is just the reasoning/orchestration mechanism, and ReAct itself is not the novelty (it already exists in the literature). The novelty to emphasize is the **scientific state + evidence-gap-driven action policy**.

**Why:** reviewers will discount or penalize claims the methodology doesn't support; precise scoping protects credibility.

## Keep the LLM constrained, not free-form

The LLM must not freely invent chemical/biological requirements from evidence text. Pipeline should be: evidence → structured biological facts → predefined property ontology → LLM selects/weights relevant properties → Requirement Specification. Every requirement needs a provenance chain back to specific evidence IDs.

**Why:** keeps the system auditable/traceable, which is central to the AgenticLS track's faithfulness/traceability emphasis, and prevents hallucinated requirements from silently driving candidate design.

## Benchmark construction must not be self-serving

Use existing ground truth wherever possible (AMR-MoEGA's own validated labels, known protein families/homolog relationships, existing molecular property benchmarks) rather than hand-building cases that flatter the proposed agent.

**Why:** avoids the reviewer criticism "the benchmark was constructed to make the proposed agent look good."

## Isolate the agentic contribution

- Candidates should come from an existing pool/dataset/benchmark/off-the-shelf generator, not a newly trained molecular generator, so the paper's contribution reads as "agentic scientific reasoning and refinement," not "we trained another molecular generator."
- Chemistry/refinement should run on a non-pathogenic/synthetic computational benchmark for v1, while biological reasoning is evaluated on real AMR cases — avoids turning the paper into an operational system for engineering resistance mechanisms (biosafety framing) and keeps evaluation tractable.
- The "removing the agent" ablation (fixed pipeline vs. adaptive agent) is treated as a *central* experiment, not an appendix ablation — if the agent doesn't beat the fixed pipeline, the agentic framing isn't justified.
- Build a deterministic/rule-based expert-policy agent as a non-LLM oracle baseline (fixed if/else rules over the same action space) to test whether LLM reasoning actually adds value over a well-defined workflow.

**Why:** these choices exist specifically so gains can be attributed to the orchestration/reasoning contribution rather than to a bigger model, more compute, or a better generator.

## Reliability/safety must be architectural, not an afterthought

- Agent must be able to represent "uncertain," "insufficient evidence," "conflicting evidence," "outside tool capability," "unsupported inference" as first-class states, and have an explicit `HUMAN_REVIEW`/`STOP` action (e.g. triggered when evidence_conflict exceeds a threshold).
- Experts, when used for evaluation, should be asked "is the computational reasoning adequately supported by the evidence?" — never "is this a good drug?" (unmeasurable / overclaims capability).

**Why:** matches the workshop's emphasis on reliability and keeps the human-evaluation protocol scientifically defensible.

## How to apply

Reference these principles whenever drafting the paper's claims, designing experiments/ablations, writing the method section, or reviewing implementation choices for scope creep. When in doubt about whether a claim or feature is in-scope, default to the more conservative/narrower framing described here.
