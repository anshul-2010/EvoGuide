"""Adapter wrapping the real AMR-MoEGA moega_pipeline code as a ScientificTool.

Deliberately does NOT go through pipeline_cli.py / ga_engine.py's CLI (pipeline_cli.py's
`model` command routes to stub files under ml_pipeline/e_modeling/ that `return
random.random()` — confirmed by direct code read) and does NOT import hgt_crossover.py
(runs a toy example with plt.show() at import time and is never wired into the real GA
loop anyway). Instead this re-implements the GA loop in-process, directly against
moega_pipeline.{search_space,genetic_operators,fitness,chromosome,gating,experts}, so we can
retain the FULL population + parent/child lineage per generation (ga_engine.py only logs the
best chromosome per generation), which amr_telemetry.py needs.
"""
from __future__ import annotations

import copy
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import external_path, load_config, repo_path  # noqa: E402
from tools.base import ScientificTool, ToolResult  # noqa: E402
from tools import amr_telemetry, amr_result_cache  # noqa: E402


# genetic_operators.mutate()'s Gaussian perturbation for float params only floors the result
# (`max(1e-6, ...)`) — it never caps the upper bound. Confirmed by hitting this directly: with
# a fixed random seed, mutation deterministically produces subsample=1.19163 at the exact same
# point in the GA every single run, which XGBoost/LightGBM then reject ("value 1.19163 for
# Parameter subsample exceed bound [0,1]"). This is a bug in the external repo's code, not
# something we introduced — clamp here rather than patch moega_pipeline/genetic_operators.py.
_PARAM_BOUNDS = {
    "xgb_params": {"n_estimators": (50, 500), "max_depth": (3, 12), "learning_rate": (0.01, 0.3),
                   "subsample": (0.5, 1.0), "colsample_bytree": (0.4, 1.0)},
    "lgbm_params": {"n_estimators": (50, 500), "num_leaves": (15, 255), "learning_rate": (0.01, 0.3),
                     "subsample": (0.5, 1.0), "colsample_bytree": (0.4, 1.0)},
    "rf_params": {"n_estimators": (50, 1000), "max_depth": (5, 50), "max_features": (0.1, 1.0)},
}


def _clamp_chromosome_params(chrom: dict) -> dict:
    for key, bounds in _PARAM_BOUNDS.items():
        for p, (low, high) in bounds.items():
            if p in chrom.get(key, {}):
                chrom[key][p] = min(high, max(low, chrom[key][p]))
    return chrom


def _inject_amr_moega_path(config: dict) -> None:
    amr_path = str(Path(config["external"]["amr_moega_path"]).resolve())
    if amr_path not in sys.path:
        sys.path.insert(0, amr_path)


def _evaluate_worker(chrom: dict, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray, amr_path: str) -> dict:
    """Top-level (picklable) function run in a separate PROCESS by joblib's default loky
    backend. Windows always spawns fresh processes (no fork), which do NOT inherit the
    parent's in-memory sys.path.insert() — each worker must redo the path injection itself
    before it can import moega_pipeline, or every parallel evaluation fails with
    ModuleNotFoundError. True multiprocessing (not threading) is used deliberately: XGBoost/
    LightGBM/RandomForest only partially release the GIL, so real cross-core speedup for this
    CPU-bound training work needs separate processes, not threads.
    """
    import sys as _sys
    if amr_path not in _sys.path:
        _sys.path.insert(0, amr_path)
    from moega_pipeline.fitness import evaluate_chromosome
    return evaluate_chromosome(chrom, X_train, y_train, X_val, y_val, cache=False)


def run_ga(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
           n_features: int, ga_cfg: dict, amr_path: str, seed: int = 0) -> dict:
    from moega_pipeline import search_space, genetic_operators, chromosome

    random.seed(seed)
    np.random.seed(seed)

    pop_size = ga_cfg["population"]
    n_gens = ga_cfg["generations"]
    elitism = ga_cfg["elitism"]
    k = ga_cfg["tournament_k"]
    mut_mask = ga_cfg["mut_rate_mask"]
    mut_param = ga_cfg["mut_rate_param"]
    n_jobs = ga_cfg.get("n_jobs", -1)

    population = [search_space.sample_chromosome(n_features) for _ in range(pop_size)]
    lineage: dict[str, list[str]] = {}
    history: list[dict] = []
    best_overall: dict | None = None

    for gen in range(n_gens):
        results = Parallel(n_jobs=n_jobs)(
            delayed(_evaluate_worker)(c, X_train, y_train, X_val, y_val, amr_path)
            for c in population)
        fitnesses = [r["auc"] for r in results]

        gen_entries = []
        for chrom, res in zip(population, results):
            h = chromosome.chromosome_hash(chrom)
            gen_entries.append({
                "hash": h,
                "auc": res["auc"],
                "feature_mask": chrom["feature_mask"],
                "parents": lineage.get(h, []),
            })
        history.append({"generation": gen, "entries": gen_entries})

        gen_best_idx = int(np.argmax(fitnesses))
        if best_overall is None or fitnesses[gen_best_idx] > best_overall["auc"]:
            best_overall = {
                "chrom": population[gen_best_idx],
                "auc": fitnesses[gen_best_idx],
                "result": results[gen_best_idx],
                "generation": gen,
            }

        if gen == n_gens - 1:
            break

        order = list(np.argsort(fitnesses)[::-1])
        new_population = [copy.deepcopy(population[i]) for i in order[:elitism]]
        for i in order[:elitism]:
            h = chromosome.chromosome_hash(population[i])
            lineage.setdefault(h, lineage.get(h, []))

        while len(new_population) < pop_size:
            parent_a = genetic_operators.tournament_selection(population, fitnesses, k=k)
            parent_b = genetic_operators.tournament_selection(population, fitnesses, k=k)
            child_a, child_b = genetic_operators.one_point_crossover(parent_a, parent_b)
            child_a = _clamp_chromosome_params(genetic_operators.mutate(child_a, mut_mask, mut_param))
            child_b = _clamp_chromosome_params(genetic_operators.mutate(child_b, mut_mask, mut_param))
            pa_hash = chromosome.chromosome_hash(parent_a)
            pb_hash = chromosome.chromosome_hash(parent_b)
            for child in (child_a, child_b):
                if len(new_population) >= pop_size:
                    break
                ch_hash = chromosome.chromosome_hash(child)
                lineage[ch_hash] = [pa_hash, pb_hash]
                new_population.append(child)
        population = new_population

    return {"best": best_overall, "history": history}


def _predict_resistance_risk(best: dict, x_row: np.ndarray) -> float:
    from moega_pipeline.experts import predict_proba_safe
    from moega_pipeline.gating import gating_weighted_prediction

    mask = np.array(best["chrom"]["feature_mask"], dtype=bool)
    if mask.sum() == 0:
        mask[0] = True
    x_sel = x_row[:, mask]

    models = best["result"]["models"]
    p_xgb = predict_proba_safe(models["xgb"], x_sel)
    p_lgb = predict_proba_safe(models["lgbm"], x_sel)
    p_rf = predict_proba_safe(models["rf"], x_sel)
    expert_probs = np.vstack([p_xgb, p_lgb, p_rf]).T  # (1, 3)

    gating = best["result"]["gating"]
    try:
        gating_weights = gating.predict_proba(x_sel)
        if gating_weights.shape[1] != 3:
            raise ValueError("gating class count mismatch")
    except Exception:
        gating_weights = np.full((1, 3), 1.0 / 3.0)

    return float(gating_weighted_prediction(expert_probs, gating_weights)[0])


class AMRMoEGATool(ScientificTool):
    name = "amr_moega"

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        _inject_amr_moega_path(self.config)

    def run(self, antibiotic: str | None = None, genome_row: int = 0, **kwargs) -> ToolResult:
        from moega_pipeline import trainer

        cfg = self.config
        antibiotic = antibiotic or cfg["amr"]["antibiotic"]
        ga_cfg = cfg["amr"]["ga"]

        features_csv = repo_path(cfg["amr"]["prepared_dir"], f"{antibiotic}_features.csv")
        labels_csv = repo_path(cfg["amr"]["prepared_dir"], f"{antibiotic}_labels.csv")
        if not features_csv.exists() or not labels_csv.exists():
            return ToolResult(self.name, ok=False,
                               error=f"Prepared AMR data not found for {antibiotic} — "
                                     f"run tools/prepare_amr_data.py first.")

        cache_dir = repo_path(cfg["amr"].get("cache_dir", "experiments/amr_cache"))
        cached = amr_result_cache.load(cache_dir, antibiotic, genome_row, ga_cfg)
        if cached is not None:
            return ToolResult(self.name, ok=True, output={"erp": cached})

        feature_names = list(pd.read_csv(features_csv, index_col=0, nrows=0).columns)
        X, y = trainer.load_features_and_labels(str(features_csv), str(labels_csv))

        if not (0 <= genome_row < X.shape[0]):
            return ToolResult(self.name, ok=False,
                               error=f"genome_row {genome_row} out of range [0, {X.shape[0]})")

        target_mask = np.zeros(X.shape[0], dtype=bool)
        target_mask[genome_row] = True
        X_target, y_target = X[target_mask], y[target_mask]
        X_pool, y_pool = X[~target_mask], y[~target_mask]

        X_train, X_val, y_train, y_val = trainer.train_val_split(
            X_pool, y_pool,
            test_size=ga_cfg["test_size"],
            random_state=ga_cfg["random_state"],
        )

        amr_path = str(Path(cfg["external"]["amr_moega_path"]).resolve())
        try:
            ga_result = run_ga(X_train, y_train, X_val, y_val, X.shape[1], ga_cfg, amr_path,
                                seed=ga_cfg["random_state"])
            best = ga_result["best"]
            history = ga_result["history"]
            resistance_risk = _predict_resistance_risk(best, X_target)
        except Exception as e:  # noqa: BLE001
            # A GA run is tens of minutes of work — surface failures as a structured ToolResult
            # (with the genome_row so it's clear WHICH run failed) rather than letting them
            # propagate as a bare exception the caller has to guess the origin of.
            return ToolResult(self.name, ok=False,
                               error=f"AMR-MoEGA GA run failed for genome_row={genome_row}, "
                                     f"antibiotic={antibiotic}: {type(e).__name__}: {e}")

        n_clusters = cfg["amr"]["telemetry"]["n_clusters"]
        cluster_info = amr_telemetry.cluster_trajectories(history, feature_names, n_clusters)
        transition_matrix = amr_telemetry.markov_transition_matrix(
            history, cluster_info["hash_to_cluster"], len(cluster_info["clusters"]))

        erp = {
            "antibiotic": antibiotic,
            "genome_row": genome_row,
            "true_label": int(y_target[0]),
            "resistance_risk": resistance_risk,
            "held_out_val_auc": best["auc"],
            "best_generation": best["generation"],
            "population_fitness_trajectory": amr_telemetry.fitness_trajectory(history),
            "entropy_trajectory": amr_telemetry.entropy_trajectory(history),
            "mechanism_clusters": cluster_info["clusters"],
            "cluster_transition_matrix": transition_matrix,
            "notes": (
                "mechanism_clusters are feature-column clusters (X### SNP codes) from the "
                "Giessen tabular dataset, NOT gene-symbol clusters — this dataset has no gene "
                "annotation. cluster_transition_matrix is a GA-generation-indexed transition "
                "likelihood between these clusters, not a real-time/calendar prediction."
            ),
        }
        amr_result_cache.save(cache_dir, antibiotic, genome_row, ga_cfg, erp)
        return ToolResult(self.name, ok=True, output={"erp": erp})


if __name__ == "__main__":
    tool = AMRMoEGATool()
    result = tool.run(genome_row=0)
    import json
    print(json.dumps(result.output if result.ok else {"error": result.error}, indent=2, default=str)[:3000])
