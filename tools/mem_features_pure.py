"""Pure-Python replacement for ProteinRAG's re_ranker/mem_features.py::get_mem_features_mummer,
which shells out to the MUMmer binary (no native Windows build, and no WSL/conda-forge setup
available on this machine — see docs/design-principles.md's scoping decision on this).

Approximates maximal-exact-matches using difflib.SequenceMatcher's matching-block finder
(stdlib only, no external dependency) instead of a true suffix-array MUM search. This is a
documented approximation, not a claim of MUMmer-equivalent results.

Produces the same feature-dict shape the real mem_features.compute_all_features() does, so it
drops into EvolutionReranker.FEATURE_COLS unchanged: mem_count, mem_total_len, mem_max_len,
mem_coverage, mem_mean_len, domain_overlap, sparse_score, dense_score, len_ratio.
"""
from __future__ import annotations

from difflib import SequenceMatcher


def get_mem_features_pure(query: str, target: str, min_mem_len: int = 5) -> dict[str, float]:
    sm = SequenceMatcher(None, query, target, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size >= min_mem_len]

    if not blocks:
        return {"mem_count": 0, "mem_total_len": 0, "mem_max_len": 0,
                "mem_coverage": 0.0, "mem_mean_len": 0.0}

    lens = [b.size for b in blocks]
    total_len = sum(lens)
    return {
        "mem_count": len(blocks),
        "mem_total_len": total_len,
        "mem_max_len": max(lens),
        "mem_coverage": total_len / max(len(query), 1),
        "mem_mean_len": total_len / len(blocks),
    }


def domain_overlap_score(query_domains: list[str] | None, target_domains: list[str] | None) -> float:
    """Jaccard overlap between two domain-annotation lists. Mirrors ProteinRAG's own
    (pure-Python, no MUMmer dependency) implementation so behavior stays consistent."""
    qd, td = set(query_domains or []), set(target_domains or [])
    if not qd and not td:
        return 0.0
    return len(qd & td) / max(len(qd | td), 1)


def compute_all_features(query_seq: str, target_seq: str,
                          query_domains: list[str] | None = None,
                          target_domains: list[str] | None = None,
                          sparse_score: float = 0.0, dense_score: float = 0.0,
                          min_mem_len: int = 5) -> dict[str, float]:
    feats = get_mem_features_pure(query_seq, target_seq, min_mem_len=min_mem_len)
    feats["domain_overlap"] = domain_overlap_score(query_domains, target_domains)
    feats["sparse_score"] = sparse_score
    feats["dense_score"] = dense_score
    shorter, longer = sorted([len(query_seq), len(target_seq)])
    feats["len_ratio"] = shorter / max(longer, 1)
    return feats
