"""Molecular descriptor representation (docs/overview.md's representation A: descriptors for
v1, molecular graph later)."""
from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import QED, Crippen, Descriptors, Lipinski


def compute_descriptors(smiles: str) -> dict[str, float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    return {
        "mw": Descriptors.MolWt(mol),
        "logp": Crippen.MolLogP(mol),
        "hbd": Lipinski.NumHDonors(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "tpsa": Descriptors.TPSA(mol),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "qed": QED.qed(mol),
    }
