"""LLM-driven ReAct policy: each turn, serialize a compact state summary + the fixed action
space into a prompt, ask the local LLM to pick ONE action (+ minimal args), validate the
response against the real action space (never trust the LLM's action name blindly), and
return it for run.py's loop to dispatch. Per docs/design-principles.md, ReAct itself is not
the contribution — the evidence-gap-driven action policy is — so this stays a thin, auditable
layer: every decision + its rationale is recorded in the trajectory log.
"""
from __future__ import annotations

import json

from agent.actions import ACTIONS, PRECONDITIONS
from agent.llm_backend import OllamaBackend
from agent.state import ScientificState

SYSTEM_PROMPT_TEMPLATE = """You are the orchestrating agent in EvoDesign-Agent, a closed-loop \
computational system that goes from an antimicrobial genome's evolutionary-resistance risk \
to a refined drug-candidate. Each turn you choose ONE action from a FIXED action space to \
progress the investigation, based on the current state and what evidence is still missing.

Available actions and preconditions:
{action_list}

Rules:
- Choose EXACTLY one action, using EXACTLY one of the action names listed above.
- Do not call an action whose precondition is not yet met.
- Typical order: RUN_AMR_ANALYSIS -> RETRIEVE_PROTEIN_EVIDENCE -> BUILD_REQUIREMENTS -> \
EVALUATE_CANDIDATE -> REFINE_CANDIDATE (repeat refine/evaluate as useful).
- REFINE_CANDIDATE requires args.transform_name to be one of: {transform_names}.
- If n_evidence_gaps > 0 and you cannot resolve it via EXPAND_PROTEIN_FAMILY or \
RETRIEVE_ADDITIONAL_EVIDENCE, choose HUMAN_REVIEW rather than guessing.
- Call STOP once you have evaluated/refined at least one candidate and further refinement is \
unlikely to help, or the remaining budget is better left unused.
- state.next_unmet_precondition tells you exactly what is still blocking progress toward a \
scored candidate — if it is non-null, strongly prefer the action it names over anything else \
(exploratory actions like VERIFY_HYPOTHESIS or extra evidence retrieval are fine, but do not \
let them substitute for actually taking the named action soon after).

Respond with ONLY a single JSON object, no other text:
{{"action": "<ACTION_NAME>", "args": {{}}, "rationale": "<one short sentence>"}}
"""

# Condition 5 ('generic ReAct', docs/experiments.md): the LLM has the same tools/action space
# but none of the evidence-gap-specific guidance, ordering hints, or retry nudge above — tests
# whether ReAct alone (vs. the evidence-gap-driven policy) explains any observed gains.
SYSTEM_PROMPT_GENERIC = """You are an agent that can call the following actions, one per turn:
{action_list}

REFINE_CANDIDATE requires args.transform_name to be one of: {transform_names}.

Respond with ONLY a single JSON object, no other text:
{{"action": "<ACTION_NAME>", "args": {{}}, "rationale": "<one short sentence>"}}
"""


class LLMPolicy:
    def __init__(self, config: dict, transforms: list[dict], backend: OllamaBackend | None = None,
                 system_prompt_template: str = SYSTEM_PROMPT_TEMPLATE):
        self.backend = backend or OllamaBackend(config)
        self.transforms = transforms
        self.system_prompt_template = system_prompt_template

    def _system_prompt(self) -> str:
        action_list = "\n".join(f"- {a}: {PRECONDITIONS[a]}" for a in ACTIONS)
        transform_names = ", ".join(t["name"] for t in self.transforms)
        return self.system_prompt_template.format(action_list=action_list,
                                                    transform_names=transform_names)

    def choose_action(self, state: ScientificState) -> dict:
        user_payload = {
            "state": state.summary(),
            "recent_trajectory": [
                {"action": t.action, "args": t.action_args, "observation": t.observation}
                for t in state.trajectory[-3:]
            ],
        }
        try:
            decision = self.backend.chat_json(self._system_prompt(), json.dumps(user_payload, default=str))
        except Exception as e:  # noqa: BLE001
            return {"action": "HUMAN_REVIEW",
                    "args": {"reason": f"llm_backend_error: {e}"},
                    "rationale": "fallback — could not reach the local LLM"}

        action = decision.get("action")
        if action not in ACTIONS:
            return {"action": "HUMAN_REVIEW",
                    "args": {"reason": f"llm_returned_invalid_action: {action!r}"},
                    "rationale": "fallback — invalid action name from LLM"}

        args = decision.get("args")
        if not isinstance(args, dict):
            args = {}
        return {"action": action, "args": args, "rationale": str(decision.get("rationale", ""))}
