"""Deterministic fixed if/else policy over the same action space as LLMPolicy — no LLM.

This is both condition 3 ('evolution + fixed retrieval') from docs/experiments.md AND the
non-LLM oracle baseline docs/design-principles.md calls for building. FixedClosedLoopPolicy
extends it with a fixed refinement pass for condition 4 ('fixed closed loop'). Same
`choose_action(state) -> {"action","args","rationale"}` interface as LLMPolicy, so run.py's
loop can use either interchangeably.
"""
from __future__ import annotations

from agent.state import ScientificState


def _decision(action: str, rationale: str, **args) -> dict:
    return {"action": action, "args": args, "rationale": rationale}


class DeterministicPolicy:
    def __init__(self) -> None:
        # A persistent evidence_gap (e.g. the beta-lactamase cluster with no resolvable gene
        # fixture) never clears via EXPAND_PROTEIN_FAMILY — it skips records with no gene by
        # design (agent/actions.py::expand_protein_family). Without this flag the policy would
        # retry EXPAND_PROTEIN_FAMILY forever instead of moving on.
        self._expanded_once = False

    def choose_action(self, state: ScientificState) -> dict:
        if state.erp is None:
            return _decision("RUN_AMR_ANALYSIS", "deterministic: always start with AMR analysis")
        if not state.rbt_records:
            if state.below_risk_threshold:
                # Confirmed by direct log analysis (experiments/analyze_pilot.py): without this
                # check, every below-threshold genome burned ~18/20 of its budget retrying this
                # exact action, since an empty rbt_records never changes on retry.
                return _decision("STOP", "deterministic: resistance_risk below retrieval "
                                          "threshold — no further evidence retrieval possible",
                                  reason="risk_below_threshold")
            return _decision("RETRIEVE_PROTEIN_EVIDENCE",
                              "deterministic: retrieve biological evidence next")
        has_gap = any(r.get("evidence_gap") for r in state.rbt_records)
        if has_gap and not self._expanded_once:
            self._expanded_once = True
            return _decision("EXPAND_PROTEIN_FAMILY",
                              "deterministic: one expansion pass for evidence gaps")
        if not state.requirements:
            return _decision("BUILD_REQUIREMENTS",
                              "deterministic: build requirements from evidence")
        if not state.evaluations:
            return _decision("EVALUATE_CANDIDATE", "deterministic: evaluate the seed candidate")
        return _decision("STOP", "deterministic: fixed sequence complete",
                          reason="deterministic_sequence_complete")


class BiologyOnlyPolicy(DeterministicPolicy):
    """Condition 2 ('biology-only'): skips the RUN_AMR_ANALYSIS/ERP requirement entirely —
    evidence is seeded externally via RiskToBiologyTranslator.translate_generic() before this
    policy's loop starts (see experiments/conditions.py), so there is no ERP to gate on."""

    def choose_action(self, state: ScientificState) -> dict:
        if not state.rbt_records:
            if state.below_risk_threshold:
                return _decision("STOP", "deterministic: resistance_risk below retrieval "
                                          "threshold — no further evidence retrieval possible",
                                  reason="risk_below_threshold")
            return _decision("RETRIEVE_PROTEIN_EVIDENCE",
                              "biology-only: evidence should already be seeded externally")
        has_gap = any(r.get("evidence_gap") for r in state.rbt_records)
        if has_gap and not self._expanded_once:
            self._expanded_once = True
            return _decision("EXPAND_PROTEIN_FAMILY",
                              "deterministic: one expansion pass for evidence gaps")
        if not state.requirements:
            return _decision("BUILD_REQUIREMENTS",
                              "deterministic: build requirements from evidence")
        if not state.evaluations:
            return _decision("EVALUATE_CANDIDATE", "deterministic: evaluate the seed candidate")
        return _decision("STOP", "deterministic: fixed sequence complete",
                          reason="deterministic_sequence_complete")


class FixedClosedLoopPolicy(DeterministicPolicy):
    """Condition 4: same fixed sequence through the first evaluation, then applies every
    transform in data/candidates/transforms.yaml once, in file order, to the best-so-far
    candidate — no adaptive choice of which transform, no early stop on regression."""

    def __init__(self, transforms: list[dict]) -> None:
        super().__init__()
        self._transforms = transforms
        self._transform_idx = 0

    def choose_action(self, state: ScientificState) -> dict:
        pre_evaluation = (
            state.erp is None or not state.rbt_records
            or (any(r.get("evidence_gap") for r in state.rbt_records) and not self._expanded_once)
            or not state.requirements
        )
        if pre_evaluation:
            return super().choose_action(state)
        if not state.evaluations:
            return _decision("EVALUATE_CANDIDATE",
                              "deterministic: evaluate the seed candidate before fixed refinement")
        if self._transform_idx < len(self._transforms):
            name = self._transforms[self._transform_idx]["name"]
            self._transform_idx += 1
            return _decision(
                "REFINE_CANDIDATE",
                f"deterministic: fixed refinement pass {self._transform_idx}/{len(self._transforms)}",
                transform_name=name)
        return _decision("STOP", "deterministic: fixed refinement sequence complete",
                          reason="deterministic_sequence_complete")
