"""Hypothesis objects, per docs/overview.md architecture layer 6."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Literal

Status = Literal["unverified", "under_evaluation", "supported", "weakly_supported",
                  "contradicted", "insufficient_evidence"]

_id_counter = itertools.count(1)


@dataclass
class Hypothesis:
    hypothesis_id: str
    statement: str
    supporting_evidence: list[str] = field(default_factory=list)
    contradicting_evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0
    status: Status = "unverified"

    def update_status(self) -> None:
        n_sup, n_con = len(self.supporting_evidence), len(self.contradicting_evidence)
        if n_sup == 0 and n_con == 0:
            self.status = "insufficient_evidence"
        elif n_con > n_sup:
            self.status = "contradicted"
        elif n_sup > 0 and n_con == 0:
            self.status = "supported" if self.confidence >= 0.6 else "weakly_supported"
        else:
            self.status = "under_evaluation"


def new_hypothesis(statement: str) -> Hypothesis:
    return Hypothesis(f"H{next(_id_counter)}", statement)
