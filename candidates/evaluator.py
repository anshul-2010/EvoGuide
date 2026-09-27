"""Scores a candidate's descriptors against CRS requirement weights. Each per-property scoring
function is a documented proxy — several ontology properties (target_interaction, selectivity)
have no real docking/binding-affinity or target-specific model in this build; their scores are
explicitly-labeled crude proxies, not claims of predicted binding/selectivity. toxicity_filters
is the one property backed by a real computation (RDKit's PAINS/BRENK structural alert catalog).
"""
from __future__ import annotations

from rdkit import Chem
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from biology.requirements import Requirement
from candidates.representation import compute_descriptors

_catalog_params = FilterCatalogParams()
_catalog_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
_catalog_params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
_CATALOG = FilterCatalog(_catalog_params)


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def score_property(prop: str, descriptors: dict, mol) -> float:
    if prop == "physicochemical":
        rules = [descriptors["mw"] <= 500, descriptors["logp"] <= 5,
                 descriptors["hbd"] <= 5, descriptors["hba"] <= 10]
        return sum(rules) / len(rules)
    if prop == "permeability":
        return _clip01(1.0 - descriptors["tpsa"] / 140.0)
    if prop == "solubility":
        return _clip01(1.0 - (descriptors["logp"] - 1.0) / 4.0)
    if prop == "stability":
        return _clip01(1.0 - descriptors["rotatable_bonds"] / 15.0)
    if prop == "developability":
        return descriptors["qed"]
    if prop == "target_interaction":
        # crude proxy (H-bond donor/acceptor count) — no docking/binding-affinity model here
        return _clip01((descriptors["hbd"] + descriptors["hba"]) / 10.0)
    if prop == "selectivity":
        # no target-specific data in this build — neutral placeholder, not a real estimate
        return 0.5
    if prop == "toxicity_filters":
        return 0.0 if _CATALOG.HasMatch(mol) else 1.0
    if prop == "diversity":
        return 0.5  # scored relative to the working pool by the caller if needed
    return 0.5


def evaluate_candidate(smiles: str, requirements: list[Requirement]) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"overall_score": 0.0, "per_requirement": {}, "descriptors": {}, "valid": False}

    descriptors = compute_descriptors(smiles)
    per_req = {}
    weighted_sum, weight_total = 0.0, 0.0
    for req in requirements:
        s = score_property(req.property_name, descriptors, mol)
        per_req[req.requirement_id] = {"property": req.property_name, "score": s, "weight": req.weight}
        weighted_sum += s * req.weight
        weight_total += req.weight

    overall = weighted_sum / weight_total if weight_total > 0 else 0.0
    return {"overall_score": overall, "per_requirement": per_req, "descriptors": descriptors, "valid": True}
