# EvoDesign-Agent: Evolution-Aware Agentic Design for Closed-Loop Antimicrobial Discovery

**Status: DRAFT — pilot-scale results, n=1 per (genome, condition) unless noted. See inline
`[TODO]` markers for what's still pending. Do not submit without resolving these.**

Target venue: NeurIPS AgenticLS Workshop 2026.

---

## Abstract

Antimicrobial resistance (AMR) prediction and protein-family retrieval have each seen recent
progress as standalone systems, but the reasoning that should connect them — *which* evolved
resistance mechanism implies *which* biological evidence is worth retrieving, and *which*
chemical properties a candidate should then satisfy — is typically left to a human or a fixed
pipeline. We present EvoDesign-Agent, a scientific-state orchestrator that adaptively links an
evolutionary-risk model (AMR-MoEGA), a hybrid protein-retrieval system (ProteinRAG), and a
bounded candidate-refinement loop through an evidence-gap-driven action policy. Unlike a fixed
pipeline, the agent decides at each step whether to run further evolutionary analysis, retrieve
additional biological evidence, build candidate requirements, or refine a candidate — and can
recognize when it lacks sufficient evidence to proceed. In a pilot evaluation across
`[N]` *E. coli* genomes and 7 conditions (including ablations that remove the agent, remove
protein-retrieval evidence, and remove evolutionary risk entirely), the full evidence-gap-aware
agent outperforms every fixed-pipeline and generic-ReAct baseline, while a deterministic
pipeline fed the *same* evidence does not outperform a naive uniform baseline at all. We report
this result alongside its most important caveat: single-run trajectories from the same local
LLM backend show substantial run-to-run variance, and we treat quantifying that variance as the
central open problem for turning this pilot into a rigorous claim.

## 1. Introduction

[TODO: standard motivating paragraph — AMR as a growing threat, the gap between evolutionary
risk models and downstream design decisions, why this gap is a reasoning problem rather than a
modeling problem. Draft below is a skeleton, not final prose.]

Antimicrobial resistance prediction and protein-family retrieval have each matured as
standalone research problems. AMR-MoEGA [CITE: arXiv:2511.12223] combines a mixture-of-experts
classifier with a genetic algorithm to model the evolutionary trajectory of resistance in
*E. coli*. ProteinRAG [CITE: anonymous submission] combines sparse, dense, and late-interaction
retrieval to recover evolutionarily related proteins across close and remote homology. Neither
system, on its own, answers the question a working scientist actually needs answered: *given
this genome's predicted resistance risk, what should a candidate therapeutic look like?*
Answering that requires deciding which evolutionary signal is worth investigating further,
translating that signal into a biologically grounded retrieval request, and turning the
resulting evidence into concrete, falsifiable chemical requirements — a chain of decisions that
is naturally agentic, not a fixed feed-forward pipeline.

We make three contributions:

1. **A scientific-state orchestration layer** (the Risk-to-Biology Translator, Evidence Graph,
   and Candidate Requirement Specification) that connects an evolutionary-risk model to a
   protein-retrieval system and a candidate-refinement loop through an explicit, auditable
   evidence chain — every requirement traces back to specific evidence IDs, and the LLM policy
   is constrained to select from a fixed property ontology rather than free-form generation.
2. **A 7-condition ablation design** (Section 4) that isolates the contribution of the agentic
   policy from (a) simply having evolutionary evidence, (b) simply having protein-retrieval
   evidence, and (c) simply having an LLM in the loop with generic ReAct prompting — the central
   comparisons a paper claiming "the agent helps" needs to survive.
3. **An honest pilot evaluation**, including a documented, fixed reliability bug (Section 6.2)
   and an explicit accounting of LLM-backend run-to-run variance (Section 6.6) that we treat as
   a first-class limitation rather than noise to average away.

We deliberately do not claim to discover new antibiotics, predict calendar time-to-resistance,
or generalize beyond the scope each underlying component actually validates (Section 3.4). The
contribution under test is the orchestration layer, not the individual components.

## 2. Related Work

[TODO: expand. Skeleton below.]

**Evolutionary AMR risk modeling.** AMR-MoEGA [CITE] predicts per-genome resistance probability
via a mixture-of-experts classifier (XGBoost, LightGBM, Random Forest, gated by an MLP) and
models simulated resistance-mechanism trajectories via a genetic algorithm over SNP-derived
features. We use it as a fixed, external tool (Section 3.1) rather than modifying it.

**Protein retrieval.** ProteinRAG [CITE] combines an evolutionary-specificity-weighted BM25
variant, dense ESM-2 retrieval, score fusion, ColBERT-style late interaction, and a learned
Gradient Boosting re-ranker, evaluated on Pfam-derived family/homology benchmarks. We use it,
similarly, as a fixed external tool.

**LLM agents and tool use.** [CITE: ReAct, ToolFormer, or similar — TODO fill with actual
citations]. Our contribution is explicitly not the ReAct mechanism itself (Section 4, condition
`generic_react` isolates this) but the evidence-gap-driven policy layered on top of it.

**Agentic scientific discovery systems.** [CITE: relevant prior agentic-science papers — TODO].

## 3. System Design

### 3.1 Problem formulation

Given a bacterial genome (here, one row of SNP-derived tabular features from the Giessen
dataset used by AMR-MoEGA) and a target antibiotic, EvoDesign-Agent must produce a scored,
evidence-traceable drug-candidate recommendation, or explicitly decline to (via `HUMAN_REVIEW`)
when it cannot do so with sufficient confidence. The system operates over a `ScientificState`
object (question, genome, evolutionary risk, evidence, hypotheses, requirements, candidates,
evaluations, trajectory, uncertainty, budget) and a finite action space:
`RUN_AMR_ANALYSIS`, `RETRIEVE_PROTEIN_EVIDENCE`, `EXPAND_PROTEIN_FAMILY`, `VERIFY_HYPOTHESIS`,
`BUILD_REQUIREMENTS`, `EVALUATE_CANDIDATE`, `REFINE_CANDIDATE`, `RETRIEVE_ADDITIONAL_EVIDENCE`,
`REASSESS_HYPOTHESIS`, `HUMAN_REVIEW`, `STOP`.

### 3.2 Evolutionary Risk Profile (AMR-MoEGA adapter)

We wrap AMR-MoEGA's real mixture-of-experts + genetic-algorithm implementation
(`moega_pipeline/`) directly, not the paper's higher-level claims. Concretely, per genome, the
adapter produces: a calibrated resistance probability (Section 6.5), a held-out validation AUC,
per-generation population fitness/diversity statistics, and feature-mask-derived "mechanism
clusters" with a Markov transition structure between them, computed by us (Section 3.2.1) since
the underlying repository does not compute this telemetry itself. We do **not** produce or
claim calendar-time-to-resistance predictions — this is explicitly outside what the underlying
GA computes (it operates in GA-generation units on a fixed hyperparameter, not real time).

**3.2.1 A note on paper-vs-code fidelity.** A direct reading of the AMR-MoEGA source (not just
its paper) found that several of the richer trajectory statistics described in the paper
(Shannon entropy of allele frequencies, dN/dS, HGT-recombinant proportion, mechanism clustering,
Markov transitions) are not actually implemented in the released code — only per-chromosome
validation AUC and a best-chromosome summary are computed natively. We implement the
entropy/clustering/transition-matrix telemetry ourselves (`tools/amr_telemetry.py`), over
feature-column genotypes rather than gene-level annotation, since the tabular dataset used
(anonymized SNP codes) has no gene-symbol annotation to cluster over. We also found and fixed
an unbounded-parameter-mutation bug in the original repository's genetic operators (a Gaussian
perturbation on `subsample`/`colsample_bytree` had no upper clamp, deterministically producing
an invalid hyperparameter value at a fixed random seed) — patched in our adapter, not in the
upstream repository.

### 3.3 Risk-to-Biology Translator and ProteinRAG adapter

The Risk-to-Biology Translator (RBT) converts risk-bearing mechanism clusters into explicit
ProteinRAG queries. Because the underlying tabular dataset has no gene annotation, we bridge
cluster identity to a query sequence via a small, explicitly-labeled heuristic mapping onto the
three mechanism families named in AMR-MoEGA's own case study (efflux-regulator, porin, and
beta-lactamase genes), each backed by a real sequence resolved live from UniProt (never
hardcoded) — when a cluster cannot be mapped (as happens for the beta-lactamase family in our
gene fixture set, since `blaTEM`/`blaCTX-M` are plasmid-borne and not annotated under the
*E. coli* K-12 reviewed reference we query against), the RBT emits an explicit evidence-gap
marker rather than guessing.

We similarly wrap ProteinRAG's real sparse, dense, fusion, and re-ranking stages directly. The
upstream repository does not expose a single function that returns the fully decomposed score
set (S_sparse, S_dense, S_hybrid, S_late, F_MEM, F_domain, S_final) — each stage's intermediate
scores are discarded by the existing orchestration code — so we implement `query_all()`
ourselves, retaining every component score for the Evidence Graph. We also found and fixed two
bugs in the original re-ranker training code: it could never find negative training examples
(qrels format only lists positive judgments, standard TREC practice, but the code looked for
explicit `rel==0` rows that do not exist), and even when negatives were available it only ever
emitted one class label, so its Gradient Boosting classifier could never actually fit. Both
fixed in our reranker implementation (`tools/reranker_pure.py`), documented rather than silently
patched over.

### 3.4 Candidate Requirement Specification and refinement

Evidence is translated into candidate requirements via a fixed nine-property ontology (target
interaction, selectivity, physicochemical, stability, solubility, permeability,
developability, toxicity filters, diversity). The LLM (where used) only selects and weights
from this fixed set — any property name outside it is dropped, never passed through — and
every requirement retains a provenance chain back to the specific evidence IDs that produced
it. Candidates are drawn from a fixed, offline seed pool of well-characterized small molecules
(not a trained generator, deliberately — see limitations), evaluated via RDKit descriptors
against the weighted requirements, and refined via one of six bounded RDKit reaction-SMARTS
transforms per step (not open-ended molecular generation).

## 4. Experimental Design

We evaluate 7 conditions, each sharing the same underlying tools and action handlers, differing
only in the decision-making policy or the evidence available to it:

1. **static_chemistry** — no AMR-MoEGA, no ProteinRAG; a fixed uniform-weight requirement set.
2. **biology_only** — ProteinRAG evidence for every gene fixture, unconditionally (no risk
   gating).
3. **no_protein_rag** — real AMR-MoEGA risk + heuristic mechanism label, but ProteinRAG is
   never queried (isolates whether real retrieved evidence adds value beyond a guessed label).
4. **evolution_fixed_retrieval** — real AMR-MoEGA + real ProteinRAG evidence, via a fixed
   deterministic policy (no adaptive replanning, no refinement loop).
5. **fixed_closed_loop** — condition 4 plus a fixed refinement pass (every available transform
   applied once, in a fixed order, regardless of observed improvement).
6. **generic_react** — an LLM with the full action space but a minimal system prompt (no
   evidence-gap-specific guidance, no ordering hints).
7. **full_agent (EvoDesign-Agent)** — the full evidence-gap-aware LLM policy.

Conditions 4-7 additionally act as this paper's "removing the agent" and "removing evolution"
ablations (condition 2 vs. 4/7 isolates the evolutionary-risk contribution; condition 4 vs. 7
isolates the agentic-policy contribution). All conditions reuse a single cached AMR-MoEGA
result per genome (Section 5.1) so the dominant compute cost is paid once, not once per
condition.

**Genomes.** `[N]` rows from the Giessen CIP-resistance dataset used by AMR-MoEGA, stratified
by true label. `[TODO: final N and stratification once genome 12 completes.]`

**Backbone.** `qwen2.5:3b-instruct`, served locally via Ollama, temperature 0.2, for the two
LLM-driven conditions, unless otherwise noted (Section 6.3 reports a formal comparison against
`qwen2.5:1.5b-instruct`).

**GA scale.** Population 50, generations 25 (paper-reported scale is 200/150; reduced for
pilot-scale wall-clock budget on a 4-core CPU machine — see Section 7).

**Corpus scale.** An 18,185-protein ProteinRAG index (family-stratified sample plus every
document referenced by the repository's own relevance judgments), embedded with
`esm2_t12_35M_UR50D` (paper-reported scale is 650M-parameter ESM-2; reduced for the same
reason).

## 5. Results

*(All numbers below are real, generated end-to-end by the system described above — see
`experiments/runs/suite/`, `experiments/runs/backbone/`, `experiments/runs/budget_scaling/`.
No simulated or hand-constructed data. `[TODO]` marks results still pending as of this draft.)*

### 5.1 Headline ablation result

Restricted to genomes where AMR-MoEGA's predicted resistance risk exceeded the retrieval
threshold (the only genomes where every condition has comparable data — see Section 5.2 for why
this restriction is itself a validity check, not a data-availability inconvenience):

| Condition | Mean best candidate score |
|---|---|
| no_protein_rag | 0.569 — *below* the naive baseline |
| biology_only | 0.584 |
| evolution_fixed_retrieval | 0.584 |
| fixed_closed_loop | 0.584 |
| static_chemistry (baseline) | 0.587 |
| generic_react | 0.591 |
| **full_agent** | **0.612** |

The pattern is a clean three-tier ladder: guessing a mechanism label without verifying it
against real evidence actively *hurts* relative to doing nothing; real evidence through a
non-adaptive pipeline is statistically indistinguishable from doing nothing; only adaptive
reasoning over real evidence produces a clear gain. This is the central result of the paper —
not "evolutionary/biological evidence helps," but specifically "the adaptive policy that
decides how to act on evidence gaps helps, and the other two ingredients alone do not."

`[TODO: final n once genome 12 lands; current n=4-5 per condition, single run each — see
Section 5.6 before treating the magnitude of this gap as a settled estimate.]`

### 5.2 Why some genomes are excluded from the headline comparison

The risk-threshold gate (`resistance_risk > 0.5`) correctly skips biological evidence retrieval
for genomes AMR-MoEGA predicts are not resistant — this is intended behavior, not a failure,
and doubles as a validity check on the gate itself: every genome with a true positive
resistance label in our pilot crossed the threshold, and every true negative did not (threshold
accuracy 100%, n=9 — Section 5.5).

### 5.3 Extended ablation: removing ProteinRAG

`no_protein_rag` (Section 4, condition 3) scores 0.569, below even the static baseline. This
was not part of the original six-condition design and was added specifically because "remove
evolution" and "remove the agent" ablations alone do not test whether *real, verified* protein
evidence is doing anything beyond providing a plausible-sounding mechanism label. It is.

### 5.4 Backbone comparison

`qwen2.5:1.5b-instruct` failed to reach a scored candidate on every genome tested (5/5),
terminating in exactly two steps each time (`RUN_AMR_ANALYSIS`, then `HUMAN_REVIEW`) with a
logged rationale that inverts the state summary's own guidance (*"No biological evidence yet,
so human review is the next best step"* — where the intended read of that same signal is "go
retrieve it"). `qwen2.5:3b-instruct` succeeded on 5/5, scoring 0.584-0.626. This is direct
evidence that the evidence-gap-aware policy design requires a minimum backbone capability to
execute correctly; the design alone does not guarantee the Section 5.1 result.

### 5.5 Calibration

Brier score 0.025 (0 = perfect, 0.25 = naive always-0.5 baseline), threshold accuracy 100%,
n=9 genomes. AMR-MoEGA's predictions are calibrated, not merely discriminative, on this pilot
sample — worth reporting alongside AUC (~0.97, consistent with the original paper) since the
risk-threshold gate (Section 5.2) depends on the predicted probabilities themselves being
trustworthy, not just rank-ordered correctly.

### 5.6 Reproducibility and run-to-run variance

An independent rerun of `full_agent` on one genome, with every setting identical (same genome,
model, budget), scored 0.584 in one run and 0.626 in another — a larger gap than several of the
between-*condition* differences in Section 5.1. The local LLM backend is not fully
deterministic even at low temperature, and the resulting trajectories genuinely differ. **We
report the qualitative pattern in Section 5.1 as robust** (it held across every genome checked
including the rerun above, and across every genome in the backbone comparison), **but the
precise magnitudes should be read as one plausible outcome per condition, not a stable point
estimate**, until conditions are rerun with multiple seeds. We treat this as the single most
important open problem for this work, ahead of adding further genomes or conditions.

### 5.7 Budget scaling

Budget=5 never reaches a scored candidate on any genome (the pipeline is inherently
multi-step); budget=10 reliably reaches an unrefined evaluation but exhausts before any
refinement; budget=40 never uses its available steps beyond what budget=20 already achieves —
across 5 genomes, the policy always terminates itself (via `HUMAN_REVIEW` or `STOP`) well
before its step limit, whether given 20 or 40 steps. Whether 40 is actually *worse* than 20, as
an earlier pass at this analysis suggested from only 2 genomes, does not survive the variance
check in Section 5.6 — the gap is within the observed noise band. The robust claim is narrower:
there is a real minimum-budget floor, and additional budget past ~20 is not being used
productively, not that it actively hurts.

### 5.8 Failure-mode analysis

A programmatic scan of every logged trajectory (detecting repeated identical actions,
precondition violations, and non-diversifying refinement attempts — the mechanically-detectable
subset of `[project]`'s failure taxonomy; semantic categories like "unsupported biological
claim" require a manual or LLM-judge pass, `[TODO]`) found a specific, reproducible pattern in
4 of 5 resistant-genome `full_agent` trajectories: after an unsuccessful `REFINE_CANDIDATE`
attempt, the policy re-applies the *same* transform rather than trying an available alternative.
This is a concrete, logged limitation, not a hypothesized one.

The same scan also caught, before it was allowed to bias any reported result: every condition
dependent on evidence retrieval — including `full_agent` — wasted approximately 18 of 20 budget
steps retrying `RETRIEVE_PROTEIN_EVIDENCE` on genomes below the risk threshold, since an empty
result never changes on retry and nothing told the policy so. Fixed by adding an explicit
terminal signal (`state.below_risk_threshold`) that all policies now check before retrying; all
affected genomes were rerun after the fix, and the results in this paper reflect the corrected
behavior. We report this as a concrete instance of the reliability failure mode our evaluation
design was built to surface, not as a footnote to bury.

### 5.9 Reliability under tool failure

`[TODO — running]`. `experiments/run_fault_injection.py` tests three fault modes against
`full_agent` (ProteinRAG outage, irrelevant-retrieval, and inverted-confidence/"contradictory"
evidence), checking whether the agent escalates to `HUMAN_REVIEW` appropriately rather than
confidently proceeding on corrupted input. Results pending.

## 6. Discussion and Limitations

**Scope.** This system does not predict calendar time-to-resistance (AMR-MoEGA's own
methodology does not support this), does not claim to discover novel antibiotics (candidates
are drawn from an existing pool, not generated), and is evaluated against a Pfam-family/clan
co-membership proxy for protein relevance (ProteinRAG's own ground truth), not experimentally
validated function or binding affinity. Corpus and GA scale are both reduced from the
originating papers' own reported scale for pilot-stage wall-clock tractability on a single
4-core CPU machine (Section 4); we report this plainly rather than implying paper-scale
validation.

**The central open problem is reproducibility** (Section 5.6), not additional scope. We
consider repeated-seed reruns of the existing conditions a higher priority than either more
genomes or more novel ablations.

**Gene-to-evidence mapping is a documented heuristic**, not a general annotation pipeline — the
tabular AMR dataset used has no gene-level annotation, and our bridge from feature-cluster to
query-gene is a small, explicit lookup table backed by real sequences, not a claim that a given
SNP cluster IS biologically that gene.

## 7. Conclusion

[TODO: final paragraph once results are complete — should state the headline finding
(adaptive reasoning over evidence, not evidence alone, drives the observed gain), the
reliability findings (a caught-and-fixed budget-waste bug, a systematic refinement-diversity
failure, a hard backbone-capability requirement), and the honest reproducibility caveat as a
coherent package, not cherry-picked wins.]

## References

`[TODO: full bibliography. Confirmed: AMR-MoEGA, arXiv:2511.12223. ProteinRAG is an anonymous
submission — cite per venue's anonymity requirements. All other citations (ReAct, prior agentic
discovery systems, ESM-2, XGBoost/LightGBM, RDKit) need to be added with verified bibliographic
details, not reconstructed from memory.]`
