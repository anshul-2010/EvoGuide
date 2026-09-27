# EvoDesign-Agent — Overview

Working title: **EvoDesign-Agent: Evolution-Aware Agentic Design for Closed-Loop Antimicrobial Discovery**
Target: NeurIPS AgenticLS Workshop 2026. Alt names considered: EvoAgent, EvoLoop, AMR2Design, EvoRAG-Design, EvoChem Agent — EvoDesign-Agent is the current internal name.

## Core framing

Reformulated from "three papers glued together" into one coherent agentic scientific system:

> An evolution-aware scientific agent that starts from an antimicrobial genome, characterizes potential evolutionary resistance risks, retrieves relevant protein-level biological evidence, translates that evidence into candidate requirements, and iteratively refines candidate properties through a closed computational design loop.

The contribution is the **reasoning/orchestration layer** connecting evolution → biological evidence → candidate requirements → iterative refinement — not merely stitching together the three existing components. See [design-principles.md](design-principles.md) for what NOT to claim.

## The three existing components → four engines

| Engine | Role | Grounded in |
|---|---|---|
| **AMR-MoEGA** | Evolutionary-risk engine — "what resistance-associated patterns/trajectories should we worry about?" | [references/amr-moega.md](../references/amr-moega.md) |
| **ProteinRAG** | Biological evidence engine — "what proteins/families/homologs/domains are relevant?" | [references/proteinrag.md](../references/proteinrag.md) |
| **Candidate refinement** (chemistry/ReAct-based) | Candidate optimization engine | existing benchmark pool, not a trained generator |
| **Scientific agent** | Orchestration/reasoning engine — decides next tool, updates hypotheses, decides when to stop | novel contribution |

Core pipeline:

```
Genome → Evolutionary Risk → Biological Evidence → Candidate Requirements
→ Candidate → Evaluation → Refinement → Re-evaluation → (loop back to Agent)
```

This must be genuinely closed-loop / agentic (adaptive next-action selection based on evidence gaps), not a fixed `AMR-MoEGA → ProteinRAG → ReAct` pipeline. See [experiments.md](experiments.md) for how this claim gets tested against fixed-pipeline baselines.

## Key architectural objects (6-layer system)

1. **Scientific Agent** (planner, memory, hypothesis state, tool routing) sits above AMR-MoEGA, ProteinRAG, Candidate Engine.
2. **Evolutionary Risk Profile (ERP)** — structured output of AMR-MoEGA analysis (risk, features, associated proteins, evidence, uncertainty, provenance).
3. **Risk-to-Biology Translator (RBT)** — converts ERP into explicit ProteinRAG retrieval requests (biological questions + retrieval constraints per risk signal). Arguably the most important new component. It should expect back retrieval-relevance evidence (ranked homologs + `S_sparse/S_dense/S_late/F_MEM/F_motif/F_domain` + a close/intermediate/remote homology tier), **not** a function/target-validity judgment — see [references/proteinrag.md](../references/proteinrag.md). If AMR-MoEGA hands the RBT a freshly implicated gene/sequence with no Pfam annotation yet, an explicit annotation step (Pfam-scan/HMMER-equivalent) must run first — ProteinRAG does not do this internally.
4. **Evidence Graph** — nodes (Genome, Gene, Variant, Protein, Family, Domain, Trajectory, Phenotype, Requirement, Candidate) with typed edges; can start as a plain Python graph, not Neo4j.
5. **Evidence objects** — every claim carries `evidence_id, source, claim, confidence, provenance, supports, uncertainty`. Confidence sourced from ProteinRAG must be labeled as *retrieval relevance*, never causal/biological confidence.
6. **Hypothesis objects** — `statement, supporting_evidence, contradicting_evidence, confidence, status` (unverified / under_evaluation / supported / weakly_supported / contradicted / insufficient_evidence).
7. **Candidate Requirement Specification (CRS)** — bridges biology → chemistry; every requirement must have evidence provenance (e.g. Requirement R4 ← Risk R2 ← Evidence E11+E13). LLM must NOT freely invent chemical requirements — must select/weight from a predefined **chemical property ontology** (target interaction, selectivity, physicochemical, stability, solubility, permeability, developability, toxicity filters, diversity).
8. **ScientificState** dataclass — question, genome, evolutionary_risks, evidence, hypotheses, requirements, candidates, evaluations, trajectory, uncertainty, budget, status.
9. **Finite action space**: `RUN_AMR_ANALYSIS`, `RETRIEVE_PROTEIN_EVIDENCE`, `EXPAND_PROTEIN_FAMILY`, `VERIFY_HYPOTHESIS`, `BUILD_REQUIREMENTS`, `EVALUATE_CANDIDATE`, `REFINE_CANDIDATE`, `RETRIEVE_ADDITIONAL_EVIDENCE`, `REASSESS_HYPOTHESIS`, `STOP` (also `HUMAN_REVIEW` on evidence conflict — see safety notes in design-principles.md).
10. Explicit **stopping criteria**: evidence coverage threshold + hypothesis confidence threshold + constraint satisfaction + improvement < epsilon, OR budget exhausted, OR uncertainty too high, OR contradictory evidence.

## Candidate representation & generation

- Two representations planned: (A) molecular descriptors (MW, logP, HBD, HBA, TPSA, rotatable bonds...) for v1; (B) molecular graph, later.
- Candidates sourced from existing dataset/benchmark/generative model/pool — NOT a newly trained generator. This deliberately isolates the paper's contribution as *agentic scientific reasoning/refinement*, not molecule generation.
- Chemistry stage scoped to **candidate-property refinement on an established computational benchmark** rather than resistance-specific molecular engineering — gives a natural biosafety story (see design-principles.md).

## Repo structure (proposed, not yet created)

Top-level modules planned: `configs/, data/, amr/, protein_rag/, biology/ (risk_translation, evidence_graph, hypotheses, requirements), candidates/ (representation, generator, evaluator, refiner), agent/ (state, memory, planner, router, policies, stopping), evaluation/, experiments/, visualization/, run.py`. Common `ScientificTool` interface (`name`, `run(input_state)`) for AMR-MoEGA/ProteinRAG/candidate tools so the agent is implementation-agnostic.

## Conference alignment (AgenticLS workshop tracks)

- **Track I** (Building Agentic Systems): agent architecture, tool orchestration, scientific grounding, specialist-vs-generalist backbone comparison, long-horizon reasoning.
- **Track II** (Closed-Loop Discovery): computational (not lab-in-the-loop) closed loop biological state → evolutionary analysis → hypothesis → candidate → evaluation → refinement.
- Evaluation/benchmarking: faithfulness, reproducibility, calibrated uncertainty, biological validity, tool efficiency, evidence grounding, traceability — possibly the paper's strongest section.

Full experimental design (baselines, ablations, RQs, metrics) in [experiments.md](experiments.md).
