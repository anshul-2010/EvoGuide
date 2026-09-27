"""Evolutionary-trajectory telemetry computed OURSELVES from the GA population history
that amr_moega_tool.py captures.

None of this exists in the AMR-MoEGA repo (confirmed by direct code read: ga_engine.py only
logs best-chromosome-per-generation + a final population summary of {hash, fitness}). The
paper describes richer telemetry (entropy, mechanism clusters, Markov transitions) but the
code does not compute it. We compute an honest, documented version of it here, over
feature-mask "genotypes" rather than real gene annotations (see references/amr-moega.md and
docs/design-principles.md for why: the Giessen dataset has anonymized SNP-code columns, not
gene symbols, so "mechanism clusters" here are feature-column clusters, not biological gene
clusters).
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def fitness_trajectory(history: list[dict]) -> list[dict]:
    out = []
    for gen in history:
        aucs = [e["auc"] for e in gen["entries"]]
        out.append({
            "generation": gen["generation"],
            "mean_auc": float(np.mean(aucs)),
            "best_auc": float(np.max(aucs)),
            "std_auc": float(np.std(aucs)),
        })
    return out


def entropy_trajectory(history: list[dict]) -> list[dict]:
    """Shannon entropy (bits) of per-feature 'on' frequency across the population, averaged
    over features, per generation. 0 = fully converged (all genomes agree on every feature),
    1 = maximally diverse (every feature is 50/50 across the population).
    """
    out = []
    for gen in history:
        masks = np.array([e["feature_mask"] for e in gen["entries"]], dtype=float)
        p = masks.mean(axis=0)
        eps = 1e-12
        p = np.clip(p, eps, 1 - eps)
        bit_entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
        out.append({
            "generation": gen["generation"],
            "mean_feature_entropy_bits": float(bit_entropy.mean()),
        })
    return out


def cluster_trajectories(history: list[dict], feature_names: list[str], n_clusters: int = 3,
                          random_state: int = 0) -> dict:
    """KMeans-cluster every (generation, chromosome) instance by its feature_mask genotype.
    Returns cluster summaries + a hash->cluster_id map used to build the transition matrix.
    """
    all_entries = [(gen["generation"], e) for gen in history for e in gen["entries"]]
    if not all_entries:
        return {"clusters": [], "hash_to_cluster": {}}

    X = np.array([e["feature_mask"] for _, e in all_entries], dtype=float)
    k = min(n_clusters, len(all_entries))
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)

    hash_to_cluster: dict[str, int] = {}
    first_seen: dict[int, int] = {}
    members: dict[int, list[tuple[int, str]]] = {c: [] for c in range(k)}
    for (gen_idx, e), label in zip(all_entries, labels):
        hash_to_cluster[e["hash"]] = int(label)
        members[int(label)].append((gen_idx, e["hash"]))
        if int(label) not in first_seen or gen_idx < first_seen[int(label)]:
            first_seen[int(label)] = gen_idx

    clusters = []
    for c in range(k):
        centroid = km.cluster_centers_[c]
        top_idx = np.argsort(-centroid)[:10]
        top_features = [feature_names[i] for i in top_idx if centroid[i] > 0.5]
        clusters.append({
            "cluster_id": c,
            "first_appearance_generation": first_seen.get(c, None),
            "n_members": len(members[c]),
            "representative_features": top_features,
        })

    return {"clusters": clusters, "hash_to_cluster": hash_to_cluster}


def markov_transition_matrix(history: list[dict], hash_to_cluster: dict[str, int], n_clusters: int) -> list[list[float]]:
    """Transition counts between mechanism clusters, built from parent->child lineage
    tracked during crossover/mutation in amr_moega_tool.run_ga(), normalized to a
    row-stochastic matrix. This is a 'simulated transition likelihood between
    resistance-mechanism-style states over GA generations' — NOT a claim about real
    calendar-time evolution (see docs/design-principles.md non-claims).
    """
    counts = np.zeros((n_clusters, n_clusters))
    for gen in history:
        for e in gen["entries"]:
            child_cluster = hash_to_cluster.get(e["hash"])
            if child_cluster is None:
                continue
            for parent_hash in e.get("parents", []):
                parent_cluster = hash_to_cluster.get(parent_hash)
                if parent_cluster is None:
                    continue
                counts[parent_cluster, child_cluster] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        probs = np.where(row_sums > 0, counts / row_sums, 0.0)
    return probs.tolist()
