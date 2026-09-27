"""ReAct-style candidate refinement — applies ONE of a small, fixed library of RDKit
reaction-SMARTS transforms per step (bounded medicinal-chemistry moves, not open-ended
generation; see docs/design-principles.md's 'candidate-property refinement on an established
computational benchmark' scoping decision). The agent's LLM policy picks WHICH transform to
try next based on unmet CRS requirements — this module only knows how to apply one.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from rdkit import Chem
from rdkit.Chem import AllChem

from candidates.pool import Candidate, new_candidate_id


def load_transforms(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["transforms"]


def apply_transform(smiles: str, transform: dict) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        rxn = AllChem.ReactionFromSmarts(transform["smarts"])
        products = rxn.RunReactants((mol,))
    except Exception:
        return None
    for product_set in products:
        product = product_set[0]
        try:
            Chem.SanitizeMol(product)
        except Exception:
            continue
        return Chem.MolToSmiles(product)
    return None


def refine(candidate: Candidate, transform_name: str, transforms: list[dict]) -> Candidate | None:
    transform = next((t for t in transforms if t["name"] == transform_name), None)
    if transform is None:
        return None
    new_smiles = apply_transform(candidate.smiles, transform)
    if new_smiles is None:
        return None
    return Candidate(
        candidate_id=new_candidate_id(),
        name=f"{candidate.name}+{transform_name}",
        smiles=new_smiles,
        parent_id=candidate.candidate_id,
        transform_applied=transform_name,
    )
