# EvoGuide (EvoDesign-Agent)

**Evolution-Aware Agentic Design for Closed-Loop Antimicrobial Discovery** — a real, end-to-end
implementation and experimental pilot targeting the NeurIPS AgenticLS Workshop 2026.

EvoGuide is a scientific-state orchestrator that adaptively links an evolutionary-risk model
(AMR-MoEGA), a hybrid protein-retrieval system (ProteinRAG), and a bounded candidate-refinement
loop through an evidence-gap-driven action policy. Unlike a fixed pipeline, it decides at each
step whether to run further evolutionary analysis, retrieve additional biological evidence,
build candidate requirements, or refine a candidate — and can recognize when it lacks sufficient
evidence to proceed. Every result in this repository is from a real run (real AMR-MoEGA GA
training, real ProteinRAG retrieval, real local LLM inference) — nothing is simulated.

## Headline result

Across a 12-genome pilot (8 CIP-resistant, 4 susceptible) and 7 execution conditions designed to
isolate evolutionary information, biological evidence, and adaptive control:

| Condition | Mean best candidate score (n=8 resistant genomes) |
|---|---|
| `no_protein_rag` (real risk + guessed mechanism, no retrieval) | **0.569** — below the naive baseline |
| `biology_only` / `evolution_fixed_retrieval` / `fixed_closed_loop` | 0.584 (identical) |
| `static_chemistry` (naive baseline) | 0.587 |
| `generic_react` | 0.587 |
| **`full_agent` (EvoGuide)** | **0.609** |

The sharper finding: three conditions with *real* biological/evolutionary evidence but no
adaptive control are statistically indistinguishable from doing nothing. Only the
evidence-gap-driven policy produces a gain — the contribution is the adaptive policy, not access
to evidence alone. See [`docs/results.md`](docs/results.md) for the full account, including the
central open caveat (run-to-run LLM variance) and a fault-injection result showing the agent
never recovers a scored candidate under complete tool outage (0/7), while it reliably escalates
to human review under corrupted-but-present evidence (6/7).

## Repository structure

```
agent/            Scientific-state controller: ScientificState, action handlers, LLM and
                   deterministic policies, stopping criteria.
biology/           Evidence graph, hypotheses, Risk-to-Biology Translator, Candidate
                   Requirement Specification (property ontology + mechanism priors).
candidates/        Candidate pool, RDKit-based descriptor computation, evaluator, and the
                   fixed refinement-transform library.
tools/             Adapters wrapping the two external repos as ScientificTools (AMR-MoEGA GA,
                   ProteinRAG hybrid retrieval), plus one-time data/index preparation scripts.
configs/           default.yaml — the single source of truth for every hyperparameter.
data/              Prepared AMR features/labels, gene fixtures, candidate seed pool + transforms.
experiments/       The 7-condition suite runner, aggregation, pilot analysis, backbone/budget/
                   fault-injection experiments, and all logged run outputs (experiments/runs/).
evaluation/        Structured per-run report builder.
docs/              Living project documentation (see below) — read these before the code.
references/        Ground-truth extraction from the two source papers this project builds on.
paper/             The NeurIPS AgenticLS submission itself (LaTeX, official style file, checklist).
run.py             Single-genome, single-condition CLI entry point.
```

## Documentation

Start here, in order:

- [`docs/overview.md`](docs/overview.md) — core framing, the four engines, the 6-layer
  architecture (Scientific Agent, Evolutionary Risk Profile, Risk-to-Biology Translator,
  Evidence Graph, Candidate Requirement Specification, finite action space, stopping criteria).
- [`docs/design-principles.md`](docs/design-principles.md) — standing scoping rules: what each
  component may and may not claim, why the LLM is constrained to a fixed property ontology, why
  candidates come from an existing pool rather than a generator, why "removing the agent" is a
  central ablation rather than an appendix afterthought.
- [`docs/experiments.md`](docs/experiments.md) — the 4 research questions, 7 execution
  conditions, key standalone experiments, and 9 evaluation dimensions the experimental design is
  built around.
- [`docs/results.md`](docs/results.md) — the living results document. Real numbers, updated as
  experiments complete, with every methodology caveat and every bug found along the way
  disclosed rather than smoothed over.
- [`references/amr-moega.md`](references/amr-moega.md) and
  [`references/proteinrag.md`](references/proteinrag.md) — ground-truth extraction from the two
  source papers (`AMR-MoEGA.pdf`, `ProteinRAG.pdf`), including a code-reality-check section
  documenting real gaps between what each paper describes and what its released code does.

## Setup

Requires Python 3.11 (chosen over 3.13 for wheel availability of `rdkit`/`faiss`/`xgboost`), and
local clones of the two external repos this project wraps (never copied into this repo):
AMR-MoEGA and Evolution-Aware-Hybrid-Protein-Retrieval (ProteinRAG). Their paths are set in
`configs/default.yaml: external.*`.

```bash
python -m venv .venv
source .venv/Scripts/activate      # or .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
```

An [Ollama](https://ollama.com) server running locally (`http://localhost:11434`) provides the
LLM backend for the agentic conditions — pull the model set in `configs/default.yaml:
agent.ollama.model` (default `qwen2.5:3b-instruct`; a 1.5B variant is also used for the backbone
comparison and is known to fail the task, see `docs/results.md`).

One-time data preparation (run once, in order):

```bash
python tools/prepare_amr_data.py              # filters the raw 60,936-column Giessen matrix to 500
python tools/prepare_gene_fixtures.py         # resolves AMR gene names to real UniProt sequences
python tools/prepare_protein_rag_index.py     # builds the ProteinRAG corpus subset + FAISS index
```

## Usage

**Single genome, single condition** (the base agent loop):

```bash
python run.py --genome-row 1 --antibiotic CIP --budget 20
```

**The full 7-condition experimental suite**, resumable across interruptions (skips
already-completed (genome, condition) pairs, and the expensive AMR-MoEGA GA is additionally
cached per genome so a from-scratch rerun never repeats finished work):

```bash
python experiments/run_suite.py --genomes 12
# or explicit rows:
python experiments/run_suite.py --genome-rows 0,1,2,3,4,5,6,7,8,9,10,12

python experiments/aggregate_results.py       # -> experiments/runs/suite/summary.csv
python experiments/analyze_pilot.py           # calibration, failure-mode scan, efficiency
```

**Auxiliary experiments:**

```bash
python experiments/run_backbone_comparison.py --genome-rows 1,4,5,6,7
python experiments/run_budget_scaling.py --genome-rows 1,4,5,6,7 --budgets 5,10,40
python experiments/run_fault_injection.py --modes outage,irrelevant,contradictory \
    --genome-rows 1,4,5,6,7,8,10
```

## Honesty and reliability notes

This project surfaced and fixed real bugs — in its own code, and in the *upstream* AMR-MoEGA and
ProteinRAG repositories — rather than silently patching around them. Highlights (full account in
`docs/results.md` and the references):

- AMR-MoEGA's own released `pipeline_cli.py` does not run the real MoE+GA (it routes to stub
  functions); the working implementation lives only in `moega_pipeline/`, called directly here.
- AMR-MoEGA's `genetic_operators.mutate()` produces out-of-bounds hyperparameters that crash
  training; fixed via a clamping wrapper in this project's adapter, not in the upstream repo.
- ProteinRAG's released reranker-training code cannot train at all as shipped (zero negatives
  found, then a single-class fit error); fixed via proper negative sampling in this project.
- A policy bug caused every evidence-dependent condition to waste ~18 of 20 budget steps
  retrying evidence retrieval on genomes below the risk threshold — found via a programmatic
  trajectory scan *before* being allowed to bias any reported result, then fixed and all
  affected genomes rerun.
- Fault-injection testing found a real, disclosed reliability gap: under total tool outage the
  agent never once recovers a scored candidate (0/7) and rarely escalates, versus reliable
  escalation under corrupted-but-present evidence (6/7) — an asymmetry documented, not hidden.

## Paper

The NeurIPS AgenticLS 2026 submission lives in [`paper/`](paper/main.tex), built on the real
official `neurips_2026.sty`. See `paper/checklist.tex` for the completed NeurIPS paper checklist,
including honestly-flagged open items (data/code release decision, final license verification,
a not-yet-written broader-impacts paragraph) rather than a fully green checklist for its own sake.
