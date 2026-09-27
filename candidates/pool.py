"""Candidate sourcing — an existing offline seed pool of real, well-known molecules, NOT a
trained generator (per docs/design-principles.md's 'isolate the agentic contribution' rule).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from rdkit import Chem

_id_counter = itertools.count(1)


def new_candidate_id() -> str:
    return f"C{next(_id_counter)}"


@dataclass
class Candidate:
    candidate_id: str
    name: str
    smiles: str
    parent_id: str | None = None
    transform_applied: str | None = None


def load_seed_pool(csv_path: str | Path) -> list[Candidate]:
    df = pd.read_csv(csv_path)
    candidates = []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            print(f"[pool] WARNING: unparseable SMILES for '{row['name']}', skipping")
            continue
        candidates.append(Candidate(new_candidate_id(), row["name"], Chem.MolToSmiles(mol)))
    return candidates
