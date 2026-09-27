"""Stopping criteria, per docs/overview.md architecture layer 10. Budget-exhausted and
Δimprovement<epsilon convergence are AUTHORITATIVE (checked by run.py's loop every iteration,
independent of what the LLM policy would choose) — reliability must be architectural, not
LLM-optional, per docs/design-principles.md. Evidence-conflict/uncertainty are exposed as
flags the LLM sees in state.summary() and can act on (e.g. choose HUMAN_REVIEW), but are not
force-stopped here since resolving them may itself require further agent actions.
"""
from __future__ import annotations

from agent.state import ScientificState


def check_stopping(state: ScientificState, epsilon: float, convergence_window: int = 3) -> str | None:
    if state.status in ("stopped", "human_review"):
        return state.stopping_reason or state.status

    if state.budget_used >= state.budget_max:
        return "budget_exhausted"

    if len(state.refinement_deltas) >= convergence_window:
        recent = state.refinement_deltas[-convergence_window:]
        if all(abs(d) < epsilon for d in recent) and state.best_candidate_id is not None:
            return "converged"

    return None
