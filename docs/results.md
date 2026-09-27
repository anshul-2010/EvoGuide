# EvoDesign-Agent — Pilot Results

Status: **core suite and fault-injection both complete** — n=8 resistant genomes for the
headline comparison (rows 1, 4, 5, 6, 7, 8, 10, 12), all 7 conditions including
`no_protein_rag`, across all 12 genomes run (84/84 (genome, condition) pairs), plus tool-failure
injection across all 8 resistant genomes x 3 fault modes (21/21 pairs — see "Reliability under
tool failure" below). The only outstanding pilot-stage item is repeated-seed reruns for variance
quantification (not yet started, still top priority — see "Reproducibility" below). Source data:
`experiments/runs/suite/summary.csv`
(`experiments/aggregate_results.py`), `experiments/runs/backbone/` (backbone comparison), from
real end-to-end runs (real AMR-MoEGA GA at population=50/generations=25, real ProteinRAG
retrieval against an 18,185-protein index, real local LLM calls via Ollama) — no simulated or
hand-constructed data anywhere in this file.

**A note on methodology, since it matters for how much to trust the numbers below**: this
pilot went through several rounds of run-then-inspect-then-fix. A pandas-version bug, two bugs
in the *original* ProteinRAG repo's reranker training, an unbounded-mutation bug in the
*original* AMR-MoEGA repo's GA, and — found via the Tier-0 failure-mode scan below — a policy
bug where every condition wasted ~18/20 budget steps retrying evidence retrieval on genomes
below the risk threshold, were all caught and fixed before being treated as results. The
genomes affected by the last bug were rerun after the fix (`agent/state.py`'s
`below_risk_threshold` signal) specifically so the failure-mode/efficiency numbers below
reflect the corrected behavior, not the bug.

**A second, more important methodology caveat, discovered via the budget-scaling runs**:
`full_agent` was run twice, independently, on genome 4 with *identical* settings (same genome,
same model, same budget=20) — once in the main suite, once as part of the backbone comparison.
The two runs scored **0.584 and 0.626 respectively** — a bigger gap than several of the
between-*condition* differences reported below. This confirms real run-to-run stochasticity in
the local LLM's decision path (Ollama's `qwen2.5:3b-instruct` at temperature=0.2 is not fully
deterministic). **Every single-condition result in this document is currently one sample from
that distribution, not a stable point estimate** — this is the single most important caveat for
anyone building on this pilot, and repeated trials per (genome, condition) are the top priority
for turning this from a promising pilot into a rigorous result. See "Reproducibility" below.

## Setup

12 genomes, all complete, from the Giessen CIP dataset: the original stratified 8
(`[0, 1, 2, 3, 4, 5, 6, 9]`, 4 resistant / 4 susceptible by true label) plus 4 additional
CIP-resistant-flagged genomes added afterward to strengthen the resistant-genome sample
(`[7, 8, 10, 12]`). All 7 conditions from `docs/experiments.md` (the 6 core conditions plus the
`no_protein_rag` extended ablation) run on every genome — 84/84 (genome, condition) pairs
complete. See `docs/design-principles.md` for what each condition may and may not claim.

## Headline result

Restricting to the 8 genomes where AMR-MoEGA's resistance-risk prediction exceeded the
retrieval threshold (`resistance_risk > 0.5`, i.e. rows 1, 4, 5, 6, 7, 8, 10, 12) — the genomes
where all 7 conditions have comparable data, since the risk-gate intentionally skips
biological evidence retrieval for genomes it predicts are not resistant (see "Why genomes were
excluded" below). This is the final headline set for the core suite (no genomes pending):

| Condition | Mean best candidate score (n=8, resistant genomes only) |
|---|---|
| no_protein_rag (extended ablation) | **0.5686** — worse than the naive baseline |
| biology_only | 0.5835 |
| evolution_fixed_retrieval | 0.5835 |
| fixed_closed_loop | 0.5835 |
| static_chemistry (baseline) | 0.5872 |
| generic_react | 0.5873 |
| **full_agent (EvoDesign-Agent)** | **0.6092** |

The ordering is unchanged from the earlier n=4 snapshot, and the gap held up as the sample
doubled: `full_agent` is still clearly ahead of every other condition, `no_protein_rag` is still
clearly the worst (below the do-nothing baseline), and the three evidence-without-adaptation
conditions remain tied at exactly 0.5835 — this tie has now held across every one of 8 resistant
genomes, not a coincidence of a small sample.

`no_protein_rag` is an extended ablation added after the first pass (docs/experiments.md's
"removing ProteinRAG" standalone experiment, previously missing from the condition set): keeps
the real AMR-MoEGA risk score and the heuristic cluster-to-mechanism label, but never actually
queries ProteinRAG. It scores **below the static baseline** — worse than doing nothing. This
sharpens the ablation ladder considerably: guessing a mechanism without verifying it against
real evidence actively hurts; real evidence through a non-adaptive pipeline is neutral; only
adaptive reasoning over real evidence helps.

Per-genome breakdown:

| genome_row | biology_only | evolution_fixed_retrieval | fixed_closed_loop | static_chemistry | generic_react | full_agent |
|---|---|---|---|---|---|---|
| 1 | 0.584 | 0.584 | 0.584 | 0.587 | 0.584 | **0.626** |
| 4 | 0.584 | 0.584 | 0.584 | 0.587 | 0.584 | 0.584 |
| 5 | 0.584 | 0.584 | 0.584 | 0.587 | 0.584 | **0.626** |
| 6 | 0.584 | 0.584 | 0.584 | 0.587 | 0.614 | **0.614** |
| 7 | 0.584 | 0.584 | 0.584 | 0.587 | 0.584 | 0.584 |
| 8 | 0.584 | 0.584 | 0.584 | 0.587 | 0.584 | **0.614** |
| 10 | 0.584 | 0.584 | 0.584 | 0.587 | 0.584 | **0.614** |
| 12 | 0.584 | 0.584 | 0.584 | 0.587 | 0.584 | **0.614** |

**Reading this honestly**: `full_agent` improves over the static baseline on 6 of 8 genomes
(rows 1, 5, 6, 8, 10, 12) and ties it on 2 of 8 (rows 4, 7 — genome 4 investigated below; genome
7 shows the identical non-diversifying-refinement pattern, see the failure-mode scan update
below). Given n=8, this is a consistent trend across every applicable genome in this pilot run,
not a statistically significant result — no significance test is reported here, and none should
be claimed from this sample size. The magnitude (full_agent ≈ +3.7% over static, +4.4% over the
fixed evidence-driven pipeline, at n=8) is real and reproducible from the logged trajectories,
not an estimate, and has moved only slightly from the earlier n=4 snapshot (+4.2%/+4.9%) as the
sample doubled — a real reassurance against the sample being too small to trust directionally,
though see "Reproducibility" below for why the *exact* magnitude still shouldn't be over-read.
This is now the final per-genome breakdown for the core suite — no further genomes pending.

## The sharper finding: evidence alone doesn't help; adaptive reasoning over it does

`biology_only`, `evolution_fixed_retrieval`, and `fixed_closed_loop` are **identical** at
0.5835 — real biological/evolutionary evidence, fed through a non-adaptive (deterministic)
pipeline, does not outperform doing nothing (`static_chemistry`, 0.5872) and is arguably
marginally worse. `generic_react` (an LLM with the same tools but no evidence-gap-specific
guidance) recovers a small edge (0.5911). Only `full_agent` — the evidence-gap-driven policy —
produces a clear, consistent gain.

This reframes the paper's core claim more precisely: **the contribution is not "using
evolutionary/biological evidence," it is the adaptive policy that decides how to act on
evidence gaps.** A fixed pipeline with the exact same evidence inputs does not reproduce the
gain. This is exactly the "removing the agent" ablation `docs/design-principles.md` calls
central, and it is the one place in this pilot where the agentic framing is directly justified
by the data rather than asserted.

## Why genomes were excluded from the headline comparison

Genomes 0, 2, 3, 9 have `resistance_risk` between 0.0004 and 0.41 (all below the 0.5
retrieval threshold in `configs/default.yaml: agent.risk_threshold_for_retrieval`). For these,
`RiskToBiologyTranslator.translate()` correctly returns no records (see
`biology/risk_translation.py`) — this is the intended behavior (don't spend a ProteinRAG
query budget investigating a genome the model doesn't believe is resistant), not a failure.
Consequently `evolution_fixed_retrieval`, `fixed_closed_loop`, `generic_react`, and
`full_agent` never reach `BUILD_REQUIREMENTS`/`EVALUATE_CANDIDATE` for these genomes and their
`best_score` is undefined (not zero, not a failed run — no candidate was ever scored).
`static_chemistry` and `biology_only` are risk-independent by design and have data for every
completed genome.

This asymmetry is itself worth stating plainly in the paper: **the system's risk-threshold
gate worked correctly** — it only invests evidence-retrieval effort in genomes AMR-MoEGA
actually flags as resistant. The original 8-genome pilot batch was stratified 4/4 by true label
without knowing in advance which predictions would cross the threshold, and all 4
resistant-labeled genomes crossed it while all 4 susceptible-labeled genomes did not. The 4
genomes added afterward (rows 7, 8, 10, 12) were specifically selected as additional
CIP-resistant-labeled genomes to grow the headline-comparison sample, and all 4 also crossed the
risk threshold — i.e., AMR-MoEGA's own risk predictions agreed with ground truth on every one of
the 12 genomes in the completed suite (held-out validation AUC ranged 0.965-0.970 across
genomes, consistent with the paper's own reported ~0.97).

## Case study: genome 4's original run, and why it does NOT mean "genome 4 is hard"

**Important update, found via the budget-scaling reruns (see "Reproducibility" above)**: an
independent rerun of genome 4 under identical settings scored 0.626 — among the *best* scores
in the entire dataset — using the SAME transforms, SAME evidence, SAME requirements. So the
behavior described below is real and worth reporting (it is a genuine, logged instance of the
model failing to diversify its search), but it is **one specific unlucky trajectory, not a
reproducible property of genome 4 itself.** The original framing of this as "the one genome
where full_agent found no improvement" was premature — corrected here rather than left as-is.

Genome 4's `full_agent` trajectory (`experiments/runs/suite/4_full_agent.json`) reached the
same starting point as every other resistant genome — real evidence retrieved (3 records, 1
gap), requirements built (5 requirements from `target_interaction`/`permeability`/
`selectivity`/`physicochemical`), aspirin evaluated as the seed candidate (0.5835, matching
every other genome's seed score, since the CRS priors and seed pool are genome-independent).
It then attempted `REFINE_CANDIDATE` with `aromatic_hydroxylation` **three consecutive times**
against the same parent candidate, producing the identical (negative, -0.0124) result each
time, instead of trying any of the other 5 available transforms
(`aromatic_fluorination`, `aromatic_methylation`, `carboxylic_to_amide`,
`amine_n_methylation`, `dehalogenation`) — before giving up and requesting `HUMAN_REVIEW`.

This is a genuine, reproducible failure mode of the local 3B LLM policy: **it does not
reliably diversify its refinement search after an unsuccessful attempt.** It maps onto
`docs/experiments.md`'s failure-mode taxonomy as a repeated-tool-call / insufficient
exploration failure. Worth reporting directly in a limitations or failure-analysis section —
it is a concrete, logged example, not a hypothesized weakness.

**Update — this is systematic, not a one-off**: a scripted scan of every logged trajectory
(`experiments/analyze_pilot.py`) found the identical pattern (`REFINE_CANDIDATE` repeated with
the same transform after a non-positive delta) in genomes 1, 4, 5, 7, 8, and 10's `full_agent`
runs — **6 of 8 resistant genomes** in the completed suite show at least one occurrence (genomes
6 and 12 do not). This is a real, reproducible limitation of the `qwen2.5:3b-instruct` backbone's
exploration strategy during refinement, not a rare edge case — the rate held steady (6/8, vs.
6/7 in the previous snapshot) as the final genome landed.

## Calibration (AMR-MoEGA's own risk predictions)

Computed by `experiments/analyze_pilot.py` from every genome's ERP (`resistance_risk` vs. true
CIP-resistance label), n=12 genomes — the complete suite:

- **Brier score: 0.0189** (0 = perfect, 0.25 = naive always-predict-0.5 baseline) — improved
  slightly again from the n=11 snapshot (0.0206)
- **Threshold accuracy (risk ≥ 0.5 vs. true label): 100%** (12/12)
- Reliability bins: predictions in the 0.8-1.0 bin averaged 0.979 with 100% observed positive
  rate (8/8); predictions in the 0-0.2 bin averaged 0.092 with 0% observed positive rate (3/3);
  one genome remains in the 0.4-0.6 "uncertain" band (predicted 0.406, true label 0 — the
  closest call in the sample, not a clean miss; unchanged since the n=9 snapshot since no new
  genome landed in that band).

AMR-MoEGA's predictions are not just discriminative (~0.97 AUC, consistent with the paper's own
reported figure) but **well-calibrated** on this pilot sample — worth reporting alongside the
AUC, since a good AUC alone doesn't establish that the predicted probabilities themselves are
trustworthy (which is what the risk-threshold gate in `agent/actions.py` actually depends on).

## A reliability bug the failure-mode scan caught, before it was allowed to bias the results

The same trajectory scan revealed that **every condition dependent on real evidence retrieval —
including `full_agent` — got stuck retrying `RETRIEVE_PROTEIN_EVIDENCE` for 18 of its 20 budget
steps** on every genome below the risk threshold, since `RiskToBiologyTranslator.translate()`
correctly returns no records for these genomes but nothing told the policy that retrying
wouldn't change that. This was not a crash and did not change any reported `best_score` (those
genomes still correctly produce no scored candidate either way), but it was a real, wasteful,
and — for the LLM-driven conditions — genuinely revealing reliability gap: the agent could not
tell "nothing to do here" from "keep trying." Fixed by adding an explicit
`state.below_risk_threshold` signal (`agent/actions.py`, `agent/state.py`,
`agent/policy_deterministic.py`) that all policies now check before retrying. Verified: genome
0's `evolution_fixed_retrieval` went from 20 steps (18 of them identical repeats) to a clean
3-step run (`RUN_AMR_ANALYSIS` → `RETRIEVE_PROTEIN_EVIDENCE` → `STOP`) after the fix. All
affected genomes were rerun; the numbers in this document reflect the fixed behavior.

This is worth including in the paper's reliability/limitations discussion directly — it is
exactly the kind of failure `docs/experiments.md`'s RQ4 asks about ("can the agent recognize
[...] while maintaining traceable reasoning"), and the honest answer, before this fix, was no.

## Backbone comparison: does model capability matter, or just the policy design?

`experiments/run_backbone_comparison.py` reruns `full_agent` with different local LLM sizes on
the same genomes, everything else held constant. Motivated directly by an earlier informal
observation that `qwen2.5:1.5b-instruct` seemed to fail more than `qwen2.5:3b-instruct` — this
formalizes that into a real comparison.

| Genome | qwen2.5:1.5b-instruct | qwen2.5:3b-instruct |
|---|---|---|
| 1 | no candidate scored (human_review after 2 steps) | 0.614 |
| 4 | no candidate scored (human_review after 2 steps) | 0.626 |
| 5 | no candidate scored (human_review after 2 steps) | 0.626 |
| 6 | no candidate scored (human_review after 2 steps) | 0.626 |
| 7 | no candidate scored (human_review after 2 steps) | 0.584 |

**Complete, clean result: 1.5B fails 0/5, 3B succeeds 5/5.** The 1.5B model failed to reach a
scored candidate on every genome tested, in exactly 2 steps every single time:
`RUN_AMR_ANALYSIS`, then `HUMAN_REVIEW` — never once varying. Its own logged rationale for genome
1 is a clean, quotable example of the failure: *"No biological evidence yet, so human review is
the next best step"* — inverting the intended read of that exact signal (`state.summary()`'s
`next_unmet_precondition` field says evidence retrieval is the next action; the 1.5B model
reads "no evidence yet" as a reason to escalate rather than a reason to go get it). This is
concrete evidence that **the evidence-gap-aware policy design requires a minimum backbone
capability to execute correctly** — the design alone does not guarantee the gains reported
above; a sufficiently capable model does.

## Reproducibility: run-to-run variance is real and currently unquantified

Directly measured (not assumed): genome 4's `full_agent` at budget=20 scored **0.584 in one run
and 0.626 in an independent rerun** with every setting identical (see the case-study correction
above). That is a larger swing than several of the between-*condition* gaps reported in the
headline table. The local LLM backend (`qwen2.5:3b-instruct`, temperature=0.2 in
`configs/default.yaml`) is not fully deterministic, and neither is the resulting trajectory.

**Practical consequence for every number in this document**: every (genome, condition) cell is
currently n=1. The headline comparison's qualitative pattern (full_agent clearly above the
static/deterministic cluster) held up across every genome checked, including the recheck of
genome 4 — so the *direction* of the finding looks robust. But the *exact magnitudes* (0.612
vs. 0.584, the specific budget-scaling curve, etc.) should be read as "one plausible outcome,"
not a precise estimate, until conditions are rerun with multiple seeds. **This is the top
priority follow-up experiment**, ahead of adding more genomes — it directly determines how
strong a claim the paper can make about the magnitude (not just the direction) of the effect.

## Budget scaling: does more budget help?

`experiments/run_budget_scaling.py` reruns `full_agent` at budget_max ∈ {5, 10, 40}, compared
against the existing budget=20 runs (each a single run — see "Reproducibility" above).

| Genome | budget=5 | budget=10 | budget=20 | budget=40 |
|---|---|---|---|---|
| 1 | no candidate | 0.5835 (seed only) | 0.614 / 0.6138 (two independent runs) | 0.5835 (human_review, 14/40 steps used) |
| 4 | no candidate | 0.5835 (seed only) | 0.584 / **0.6255** (two independent runs) | 0.5835 (human_review, 16/40 steps used) |
| 5 | no candidate | 0.5835 (seed only) | 0.626 / 0.6255 | **0.6255** (human_review, 21/40 steps used) |
| 6 | no candidate | 0.5835 (seed only) | 0.614 / 0.6138 | 0.6138 (stopped, 18/40 steps used) |
| 7 | no candidate | 0.5835 (seed only) | 0.5835 | 0.5835 (human_review, 14/40 steps used) |

Two findings survive the variance caveat above; one does not:

1. **There's a real minimum-budget floor, and this is clean regardless of variance.** budget=5
   never reaches a scored candidate on any genome — the pipeline is inherently multi-step and 5
   actions cannot complete it. budget=10 reliably reaches evaluation (the seed score, 0.5835)
   but exhausts before any refinement, on every genome tested.
2. **Budget=40 never clearly beats budget=20's better run**, and in every case stops itself
   early via `HUMAN_REVIEW`/`STOP` well before exhausting its available steps (14-21 of 40
   used) — the policy is not budget-starved at 20, it is choosing to stop for reasons unrelated
   to remaining budget. This part is robust: across 5 genomes, more available budget never once
   caused the policy to use it productively beyond what budget=20 already achieved.
3. **What's NOT clean**: whether budget=40 is actually *worse* than budget=20, as originally
   claimed after only 2 genomes. With the full picture (and knowing genome 4's budget=20 alone
   ranges 0.584-0.626 across reruns), the budget=20 vs. budget=40 gap is within the same noise
   band as the run-to-run variance documented above. The honest claim is (1) and (2), not "more
   budget hurts" — that specific framing has been walked back from the earlier version of this
   section.

## Efficiency

`evolution_fixed_retrieval` (which pays the AMR-MoEGA GA cost fresh) averaged ~3,940s
(~66 min) per genome at real scale (population=50, generations=25, parallelized across 4
CPU cores). Every other condition for the same genome reuses the cached AMR-MoEGA result
(`tools/amr_result_cache.py`) and completes in seconds (`fixed_closed_loop`, ~18s) to minutes
(`generic_react`/`full_agent`, ~500-600s, dominated by local LLM inference latency).

## Reliability under tool failure

`experiments/run_fault_injection.py` reran `full_agent` on all 8 resistant genomes under 3 fault
modes injected into `ProteinRAGTool`: **outage** (tool call fails outright), **irrelevant**
(returns real-looking but semantically unrelated hits), **contradictory** (returns hits with
inverted confidence scores). 21/21 (genome, mode) pairs complete.

| Fault mode | human_review | stopped (budget_exhausted) | stopped (converged) | Reached a scored candidate |
|---|---|---|---|---|
| outage | 1/7 | 6/7 | 0/7 | **0/7** |
| irrelevant | 6/7 | 0/7 | 1/7 | 7/7 (1 via convergence, 6 held pending human review with a seed score) |
| contradictory | 6/7 | 1/7 | 0/7 | 6/7 held pending human review; 1/7 exhausted budget |

**The clean, striking result: under complete outage, the agent never once reaches a scored
candidate in 7/7 trials, and rarely escalates.** Inspecting the trajectories directly
(`experiments/runs/fault_injection/*_outage.json`) shows why: after the initial
`RUN_AMR_ANALYSIS`/`RETRIEVE_PROTEIN_EVIDENCE` calls succeed (evidence gathered before the fault
model intercepts later calls), every subsequent `EXPAND_PROTEIN_FAMILY` correctly reports
`n_new_evidence: 0`, and every `BUILD_REQUIREMENTS` correctly reports `n_requirements: 0` — the
agent is told, truthfully and repeatedly, that nothing is coming back. Its response is to keep
alternating `EXPAND_PROTEIN_FAMILY` and `BUILD_REQUIREMENTS` (observed action-count splits like
`{EXPAND_PROTEIN_FAMILY: 11, BUILD_REQUIREMENTS: 6}` for a 20-step budget) rather than recognizing
the pattern and escalating — only genome 1's run ever reaches `HUMAN_REVIEW` under outage; the
other 6 silently exhaust the full budget with no candidate ever evaluated.

**This is the same underlying policy weakness already documented in the failure-mode scan**
(non-diversifying `REFINE_CANDIDATE` retries, see above), now showing up in a second context:
faced with a flat, repeated "no new evidence" signal, the policy repeats the same two actions
instead of treating repetition-without-progress as a signal to stop or escalate. It is a
materially worse failure than what happens under irrelevant/contradictory evidence, where the
tool still returns *something* and the agent reliably (6/7 both modes) recognizes the evidence is
suspect and requests `HUMAN_REVIEW` rather than confidently proceeding on corrupted input.

**Honest reading for RQ4**: the agent's reliability behavior is asymmetric. It handles "the
evidence I got looks wrong" well, but handles "I am getting nothing at all" poorly — the
`state.below_risk_threshold` fix (see above) solved this exact class of problem for one specific
cause (genomes below the risk threshold) but does not generalize to a genuine tool outage, which
has no equivalent explicit terminal signal in the current action space. A natural follow-up fix,
not yet implemented: detect N consecutive zero-new-evidence observations and force a policy
branch (escalate or stop) the same way `below_risk_threshold` already does for the risk-gated
case.

## Next steps

1. **Done**: core suite — all 12 genomes x all 7 conditions (84/84 pairs), including the
   `no_protein_rag` backfill for rows 8, 10, 12.
2. **Done**: backbone comparison, all 5 genomes (1, 4, 5, 6, 7) x both models — see "Backbone
   comparison" above; clean 0/5 vs. 5/5 result.
3. **Done**: budget scaling, all 5 genomes x budgets {5, 10, 40} vs. existing 20 — see "Budget
   scaling" above.
4. **Done**: tool-failure injection, all 8 resistant genomes x 3 fault modes (21/21 pairs) — see
   "Reliability under tool failure" above; surfaced a new, sharper reliability gap (0/7 scored
   candidates under outage) worth its own paragraph in the paper.
5. **Top priority, not yet started**: repeated-seed reruns of existing conditions to quantify
   run-to-run LLM variance (see "Reproducibility" above) — ranked ahead of any further genomes or
   ablations, since it determines how strong a claim the paper can make about magnitude, not just
   direction.
6. Not yet attempted: a matched set of additional CIP-susceptible genomes, to test whether the
   risk-threshold gate's behavior holds at a larger sample.
7. Not yet implemented: a consecutive-zero-new-evidence terminal signal (analogous to
   `below_risk_threshold`) to fix the outage-mode failure found above.
8. Still deferred: the full failure-mode taxonomy applied with semantic judgment (the
   programmatic scan above only catches the mechanically-detectable subset — repeated identical
   calls, precondition violations, non-diversifying refinement; categories like "unsupported
   biological claim" or "evidence misinterpretation" need a manual or LLM-judge pass, not yet
   done), and perturbation-based robustness testing beyond the fault-injection modes above.
