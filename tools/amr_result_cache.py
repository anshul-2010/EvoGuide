"""Disk cache for AMR-MoEGA ERP results, keyed by (antibiotic, genome_row, GA config).

Every baseline/ablation condition for a given genome reuses the SAME expensive GA result —
this is what makes 'N conditions x M genomes' tractable (the GA, not the other pipeline
stages, is the dominant per-genome cost — see docs/experiments.md and the plan that
introduced this cache), and what makes experiments/run_suite.py resumable across session
restarts: a rerun for an already-cached (antibiotic, genome_row, GA config) triple is a cache
hit, not another multi-hour GA run.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

_GA_KEY_FIELDS = ["population", "generations", "elitism", "tournament_k", "mut_rate_mask",
                   "mut_rate_param", "test_size", "random_state"]


def _cache_key(antibiotic: str, genome_row: int, ga_cfg: dict) -> str:
    payload = {"antibiotic": antibiotic, "genome_row": genome_row,
               **{k: ga_cfg[k] for k in _GA_KEY_FIELDS}}
    s = json.dumps(payload, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def load(cache_dir: Path, antibiotic: str, genome_row: int, ga_cfg: dict) -> dict | None:
    path = cache_dir / f"{_cache_key(antibiotic, genome_row, ga_cfg)}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save(cache_dir: Path, antibiotic: str, genome_row: int, ga_cfg: dict, erp: dict) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{_cache_key(antibiotic, genome_row, ga_cfg)}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(erp, f, indent=2, default=str)
    return path
