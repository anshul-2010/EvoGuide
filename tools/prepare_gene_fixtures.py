"""One-time prep: resolve the AMR-relevant gene names named in the AMR-MoEGA paper's case
study (acrR, marA, soxS, ompF, ompC, blaTEM, blaCTX-M) to real UniProt sequences.

Deliberately does NOT hardcode accession numbers anywhere (see docs/design-principles.md /
references/amr-moega.md — sequence identity must be verifiable, not recalled from memory).
Instead resolves each gene name -> reviewed (Swiss-Prot) UniProt accession live via UniProt's
REST search API, then fetches its FASTA sequence, and writes both the sequences and the
resolved gene->accession mapping to disk so the mapping itself is inspectable/auditable.

Usage:
    python tools/prepare_gene_fixtures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_config, repo_path  # noqa: E402

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"


def resolve_gene(gene_name: str, taxid: int, session: requests.Session) -> str | None:
    query = f"gene:{gene_name} AND organism_id:{taxid} AND reviewed:true"
    resp = session.get(UNIPROT_SEARCH_URL, params={"query": query, "format": "fasta", "size": 1},
                        timeout=30)
    resp.raise_for_status()
    text = resp.text.strip()
    return text or None


def main(config: dict) -> None:
    genes = config["gene_fixtures"]["genes"]
    taxid = config["gene_fixtures"]["organism_taxid"]
    fasta_out = repo_path(config["gene_fixtures"]["fasta_path"])
    mapping_out = repo_path(config["gene_fixtures"]["mapping_path"])
    fasta_out.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    records: list[str] = []
    mapping: dict[str, dict] = {}

    for gene in genes:
        fasta_text = resolve_gene(gene, taxid, session)
        if fasta_text is None:
            print(f"[prepare_gene_fixtures] WARNING: no reviewed UniProt entry for "
                  f"gene={gene} taxid={taxid} — skipped, RBT will flag this as an evidence gap")
            continue
        header_line = fasta_text.splitlines()[0]
        accession = header_line.split("|")[1] if "|" in header_line else None
        records.append(fasta_text)
        mapping[gene] = {"accession": accession, "header": header_line, "source": "UniProt REST"}
        print(f"[prepare_gene_fixtures] resolved {gene} -> {accession}")

    if not records:
        raise RuntimeError("No gene fixtures resolved — check network access / gene names.")

    fasta_out.write_text("\n".join(records) + "\n", encoding="utf-8")
    with open(mapping_out, "w", encoding="utf-8") as f:
        yaml.safe_dump(mapping, f, sort_keys=False)

    print(f"[prepare_gene_fixtures] wrote {len(records)} sequences -> {fasta_out}")
    print(f"[prepare_gene_fixtures] wrote mapping -> {mapping_out}")


if __name__ == "__main__":
    main(load_config())
