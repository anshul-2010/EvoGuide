"""One-time prep: turn the AMR-MoEGA repo's Giessen_dataset CSVs into the
features.csv / labels.csv pair that moega_pipeline.trainer.load_features_and_labels
expects (index_col=0 on both, matching row order).

Usage:
    python tools/prepare_amr_data.py --antibiotic CIP
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import external_path, load_config, repo_path  # noqa: E402


def prepare(config: dict, antibiotic: str) -> tuple[Path, Path]:
    features_src = external_path(config, "amr_moega_path", "data", "Giessen_dataset",
                                  "cip_ctx_ctz_gen_multi_data.csv")
    pheno_src = external_path(config, "amr_moega_path", "data", "Giessen_dataset",
                               "cip_ctx_ctz_gen_pheno.csv")

    print(f"[prepare_amr_data] reading {features_src} (this is a ~61k-column CSV, expect "
          f"~1-2 minutes) ...")
    features_df = pd.read_csv(features_src, index_col=0)
    pheno_df = pd.read_csv(pheno_src, index_col=0)

    if antibiotic not in pheno_df.columns:
        raise ValueError(f"Antibiotic '{antibiotic}' not in {list(pheno_df.columns)}")

    labels = pheno_df[antibiotic].dropna()
    common_idx = features_df.index.intersection(labels.index)
    features_df = features_df.loc[common_idx]
    labels = labels.loc[common_idx].astype(int)

    max_features = config["amr"].get("max_features")
    if max_features and features_df.shape[1] > max_features:
        print(f"[prepare_amr_data] pre-filtering {features_df.shape[1]} -> {max_features} "
              f"columns via SelectKBest(f_classif) — see configs/default.yaml amr.max_features "
              f"for why this pre-filter exists (fast-demo scoping, not part of moega_pipeline).")
        selector = SelectKBest(f_classif, k=max_features).fit(features_df.values, labels.values)
        keep_cols = features_df.columns[selector.get_support()]
        features_df = features_df[keep_cols]

    out_dir = repo_path(config["amr"]["prepared_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    features_out = out_dir / f"{antibiotic}_features.csv"
    labels_out = out_dir / f"{antibiotic}_labels.csv"

    features_df.to_csv(features_out)
    labels.to_frame(name=antibiotic).to_csv(labels_out)

    print(f"[prepare_amr_data] {antibiotic}: {features_df.shape[0]} samples, "
          f"{features_df.shape[1]} features, {int(labels.sum())} positive")
    print(f"  -> {features_out}")
    print(f"  -> {labels_out}")
    return features_out, labels_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--antibiotic", default=None, help="CIP/CTX/CTZ/GEN; defaults to config value")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    antibiotic = args.antibiotic or cfg["amr"]["antibiotic"]
    prepare(cfg, antibiotic)
