"""Risk-to-Biology Translator (RBT) — converts an AMR-MoEGA Evolutionary Risk Profile (ERP)
into explicit ProteinRAG retrieval requests, per docs/overview.md layer 3.

Scope note (see docs/design-principles.md / references/amr-moega.md): the Giessen dataset's
mechanism_clusters are feature-column clusters (X### SNP codes), not gene-symbol clusters —
there is no gene annotation in this tabular data to translate directly into a query sequence.
This bridges that gap with a small, explicitly-heuristic mapping from cluster ORDER (by
first-appearance generation) onto the three mechanism groups named in the AMR-MoEGA paper's
own case study (efflux-regulator, porin, beta-lactamase), each backed by real UniProt-fetched
gene sequences (tools/prepare_gene_fixtures.py). This is a documented demo bridge, not a
claim that a given feature cluster IS biologically that gene — every evidence record produced
this way says so explicitly in its claim text and is never silently upgraded to a hard fact.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml
from Bio import SeqIO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_config, repo_path  # noqa: E402
from biology.evidence_graph import EvidenceGraph  # noqa: E402
from tools.protein_rag_tool import ProteinRAGTool  # noqa: E402

GENE_GROUPS = [
    {"mechanism": "efflux_regulator", "genes": ["acrR", "marA", "soxS"]},
    {"mechanism": "porin_permeability", "genes": ["ompF", "ompC"]},
    {"mechanism": "beta_lactamase", "genes": ["blaTEM", "blaCTX-M"]},
]


def _load_gene_fixtures(config: dict) -> dict[str, str]:
    fasta_path = repo_path(config["gene_fixtures"]["fasta_path"])
    mapping_path = repo_path(config["gene_fixtures"]["mapping_path"])
    if not fasta_path.exists() or not mapping_path.exists():
        return {}

    with open(mapping_path, encoding="utf-8") as f:
        mapping = yaml.safe_load(f) or {}
    acc_to_gene = {v["accession"]: gene for gene, v in mapping.items() if v.get("accession")}

    seqs: dict[str, str] = {}
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        parts = record.id.split("|")
        accession = parts[1] if len(parts) >= 2 else record.id
        gene = acc_to_gene.get(accession)
        if gene:
            seqs[gene] = str(record.seq)
    return seqs


class RiskToBiologyTranslator:
    def __init__(self, config: dict | None = None,
                 protein_rag_tool: ProteinRAGTool | None = None,
                 graph: EvidenceGraph | None = None):
        self.config = config or load_config()
        self.protein_rag = protein_rag_tool or ProteinRAGTool(self.config)
        self.graph = graph if graph is not None else EvidenceGraph()
        self.gene_seqs = _load_gene_fixtures(self.config)

    def _query_gene(self, gene: str | None, mechanism_hint: str, cluster_id) -> dict:
        """Query ProteinRAG for one gene and wrap the results as Evidence — shared by both
        translate() (risk-conditional, per mechanism-cluster) and translate_generic() (queries
        every resolved fixture unconditionally, for the 'biology-only' baseline condition)."""
        record = {"cluster_id": cluster_id, "mechanism_hint": mechanism_hint,
                   "gene": gene, "heuristic_mapping": True}

        if gene is None:
            self.graph.add_evidence(
                source="risk_translation",
                claim=f"No queryable gene sequence resolved for cluster {cluster_id} "
                      f"(heuristic group '{mechanism_hint}') — run "
                      f"tools/prepare_gene_fixtures.py or extend gene_fixtures config.",
                confidence=0.0, confidence_type="evidence_gap", uncertainty="evidence_gap")
            record["evidence_gap"] = True
            return record

        seq = self.gene_seqs[gene]
        tool_result = self.protein_rag.run(seq=seq, domains=[mechanism_hint])
        if not tool_result.ok:
            self.graph.add_evidence(
                source="risk_translation",
                claim=f"ProteinRAG query failed for gene {gene} (cluster {cluster_id}): "
                      f"{tool_result.error}",
                confidence=0.0, confidence_type="evidence_gap", uncertainty="evidence_gap")
            record["evidence_gap"] = True
            return record

        evidence_ids = []
        for hit in tool_result.output["results"]:
            ev = self.graph.add_evidence(
                source="protein_rag",
                claim=(f"{hit['acc']} (Pfam family {hit['family_id']}) is a retrieved "
                        f"evolutionary relative of query gene '{gene}' "
                        f"(heuristic mechanism hint: {mechanism_hint}); "
                        f"homology tier={hit['homology_tier']}"),
                confidence=hit["s_final"],
                confidence_type="retrieval_relevance",
                provenance={"query_gene": gene, "cluster_id": cluster_id, **hit},
            )
            evidence_ids.append(ev.evidence_id)

        record.update(evidence_gap=False, evidence_ids=evidence_ids,
                       top_hits=tool_result.output["results"][:5])
        return record

    def translate(self, erp: dict) -> list[dict]:
        threshold = self.config["agent"]["risk_threshold_for_retrieval"]
        if erp["resistance_risk"] < threshold:
            return []

        clusters = sorted(erp["mechanism_clusters"],
                           key=lambda c: c["first_appearance_generation"] or 0)
        records = []
        for i, cluster in enumerate(clusters):
            group = GENE_GROUPS[i % len(GENE_GROUPS)]
            gene = next((g for g in group["genes"] if g in self.gene_seqs), None)
            records.append(self._query_gene(gene, group["mechanism"], cluster["cluster_id"]))
        return records

    def translate_no_retrieval(self, erp: dict) -> list[dict]:
        """'Remove ProteinRAG' ablation (docs/experiments.md's 'removing ProteinRAG' standalone
        experiment — previously missing from the condition set). Keeps the REAL AMR-MoEGA risk
        gating and cluster-to-mechanism heuristic mapping, but never actually queries ProteinRAG
        — isolates whether real retrieved protein evidence adds anything beyond just having a
        risk score and a guessed mechanism label. Requirements built from these records will
        have NO evidence_ids (empty provenance chain) — that's the intended, honest signal of
        skipping this stage, not an oversight.
        """
        threshold = self.config["agent"]["risk_threshold_for_retrieval"]
        if erp["resistance_risk"] < threshold:
            return []

        clusters = sorted(erp["mechanism_clusters"],
                           key=lambda c: c["first_appearance_generation"] or 0)
        records = []
        for i, cluster in enumerate(clusters):
            group = GENE_GROUPS[i % len(GENE_GROUPS)]
            gene = next((g for g in group["genes"] if g in self.gene_seqs), None)
            records.append({
                "cluster_id": cluster["cluster_id"], "mechanism_hint": group["mechanism"],
                "gene": gene, "heuristic_mapping": True, "evidence_gap": False,
                "evidence_ids": [], "top_hits": [],
            })
        return records

    def translate_generic(self) -> list[dict]:
        """Condition 2 ('biology-only', docs/experiments.md): no ERP, no risk gating, no
        cluster-conditional gene selection — queries ProteinRAG with every resolved gene
        fixture unconditionally, to test what biological evidence alone (no evolutionary risk
        signal) contributes."""
        records = []
        for group in GENE_GROUPS:
            gene = next((g for g in group["genes"] if g in self.gene_seqs), None)
            records.append(self._query_gene(gene, group["mechanism"], f"generic_{group['mechanism']}"))
        return records
