"""Evidence objects + a plain-Python typed graph, per docs/overview.md's architecture layer 4/5.
No Neo4j — an in-memory graph is enough for a single-run agent loop."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Literal

NodeType = Literal["Genome", "Gene", "Variant", "Protein", "Family", "Domain",
                    "Trajectory", "Phenotype", "Requirement", "Candidate"]

_id_counter = itertools.count(1)


@dataclass
class Evidence:
    """Every claim in the system carries this — confidence is ALWAYS scoped to what actually
    produced it (retrieval relevance, GA-generation trajectory, etc.), never upgraded to a
    causal/biological claim by the field name alone. See docs/design-principles.md."""
    evidence_id: str
    source: str                       # e.g. "amr_moega", "protein_rag"
    claim: str
    confidence: float
    confidence_type: str              # e.g. "retrieval_relevance", "ga_val_auc", "expert_judgment"
    provenance: dict[str, Any] = field(default_factory=dict)
    supports: list[str] = field(default_factory=list)   # node/hypothesis ids this bears on
    uncertainty: str | None = None    # e.g. "evidence_gap", "conflicting", None


def new_evidence_id() -> str:
    return f"E{next(_id_counter)}"


@dataclass
class Node:
    node_id: str
    node_type: NodeType
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    src: str
    dst: str
    relation: str
    data: dict[str, Any] = field(default_factory=dict)


class EvidenceGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.evidence: dict[str, Evidence] = {}

    def add_node(self, node_id: str, node_type: NodeType, **data: Any) -> Node:
        node = Node(node_id, node_type, data)
        self.nodes[node_id] = node
        return node

    def add_edge(self, src: str, dst: str, relation: str, **data: Any) -> Edge:
        edge = Edge(src, dst, relation, data)
        self.edges.append(edge)
        return edge

    def add_evidence(self, source: str, claim: str, confidence: float,
                      confidence_type: str, provenance: dict[str, Any] | None = None,
                      supports: list[str] | None = None,
                      uncertainty: str | None = None) -> Evidence:
        ev = Evidence(new_evidence_id(), source, claim, confidence, confidence_type,
                       provenance or {}, supports or [], uncertainty)
        self.evidence[ev.evidence_id] = ev
        for target in ev.supports:
            self.add_edge(ev.evidence_id, target, "supports")
        return ev

    def neighbors(self, node_id: str, relation: str | None = None) -> list[str]:
        return [e.dst for e in self.edges if e.src == node_id and
                (relation is None or e.relation == relation)]

    def evidence_for(self, node_id: str) -> list[Evidence]:
        supporting_ids = {e.src for e in self.edges if e.dst == node_id and e.relation == "supports"}
        return [self.evidence[eid] for eid in supporting_ids if eid in self.evidence]

    def to_dict(self) -> dict:
        return {
            "nodes": [{"id": n.node_id, "type": n.node_type, "data": n.data} for n in self.nodes.values()],
            "edges": [{"src": e.src, "dst": e.dst, "relation": e.relation, "data": e.data} for e in self.edges],
            "evidence": [vars(e) for e in self.evidence.values()],
        }
