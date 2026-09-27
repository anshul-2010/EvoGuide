"""One-time prep: build a fast-demo-scale ProteinRAG corpus subset + sparse/dense/multi-vector
indices + a trained re-ranker, all from the real Evolution-Aware-Hybrid-Protein-Retrieval repo
code (sys.path-injected, not copied).

Corpus subset = family-stratified sample (config protein_rag.subset_size) UNION every
accession referenced (capped per query) by the repo's own data/qrels.txt — the union matters:
qrels references ~38.6k distinct doc accessions from the full 207,766-row corpus, and a naive
family-stratified sample of a few thousand would almost certainly miss the specific documents
the 11 real queries have relevance judgments for, leaving the re-ranker with ~0 trainable
pairs. Everything downstream (BM25 index, dense index, multi-vector index, reranker training,
and run.py's query-time corpus lookup) uses this SAME combined subset, so doc_ids returned by
any retrieval stage are always resolvable.

Usage:
    python tools/prepare_protein_rag_index.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import faiss
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_config, repo_path  # noqa: E402
from tools import protein_rag_paths  # noqa: E402
from tools.reranker_pure import EvolutionReranker  # noqa: E402


def _add_synthetic_domains(df: pd.DataFrame) -> pd.DataFrame:
    """No per-domain boundary annotation exists in this dataset (pfam_sequences.parquet has
    family_id/clan_id but no domain start/end coordinates) — approximate 'domains' as a
    single-element list containing the Pfam family_id. Documented limitation: this makes
    domain_overlap a same-family/cross-family signal, not literal InterPro domain matching,
    and multi-vector late interaction (S_late) degenerates to whole-sequence similarity since
    segment_extractor.extract_segments() falls back to one 'full' segment with no domain_hits.
    """
    df = df.copy()
    df["domains"] = df["family_id"].apply(lambda f: [f])
    return df


def build_subset(corpus_df: pd.DataFrame, qrels_df: pd.DataFrame, subset_size: int,
                  seed: int, cap_per_query: int = 20) -> pd.DataFrame:
    frac = min(1.0, subset_size / len(corpus_df))

    # Explicit loop rather than groupby(...).apply(...): pandas 3.x's groupby-apply excludes
    # the grouping column from what's passed to the function by default, which silently
    # dropped 'family_id' from every stratified row (confirmed by direct testing) — the
    # downstream concat with extra_df then filled it with NaN for ~85% of the subset, making
    # family_id-derived diversity/domain features meaningless. A plain loop always keeps every
    # original column, independent of pandas-version apply semantics.
    parts = []
    for _, g in corpus_df.groupby("family_id"):
        # round(len(g) * frac) rounds to 0 for any group smaller than ~1/frac members, which
        # would silently drop EVERY family below that size from a "family-stratified" sample —
        # the opposite of what stratification is for. Floor every non-empty group at 1 row.
        n = max(1, round(len(g) * frac))
        parts.append(g.sample(n=min(n, len(g)), random_state=seed))
    stratified = pd.concat(parts, ignore_index=True)
    if len(stratified) > subset_size:
        # Downsampling here still preserves broad family coverage since `stratified` itself
        # spans every family in the corpus at this point.
        stratified = stratified.sample(n=subset_size, random_state=seed)

    extra_ids: set[str] = set()
    for qid, group in qrels_df.groupby("qid"):
        extra_ids.update(group[group["rel"] >= 2]["did"].tolist()[:cap_per_query])
        extra_ids.update(group[group["rel"] == 0]["did"].tolist()[:cap_per_query])
    extra_df = corpus_df[corpus_df["acc"].isin(extra_ids)]

    combined = pd.concat([stratified, extra_df]).drop_duplicates(subset="acc").reset_index(drop=True)
    return combined


def main(config: dict) -> None:
    base = protein_rag_paths.inject(config)
    from bio_bm25 import BioAwareBM25
    from embedder import ProteinEmbedder
    from dense_retriever import DenseRetriever

    prcfg = config["protein_rag"]
    corpus_path = base / prcfg["corpus_parquet"]
    qrels_path = base / "data" / "qrels.txt"
    queries_path = base / "data" / "splits" / "queries.parquet"

    print(f"[prepare_protein_rag_index] loading corpus {corpus_path} ...")
    corpus_full = pd.read_parquet(corpus_path)
    qrels_df = pd.read_csv(qrels_path, sep=r"\s+", engine="python",
                            names=["qid", "iter", "did", "rel"])
    queries_df = pd.read_parquet(queries_path)

    # qrels-judged documents live almost entirely in test.parquet, not train.parquet (confirmed
    # by direct inspection: of 38,586 unique judged doc accessions, all 38,586 are found in
    # test.parquet vs. only 13,164 in train.parquet) — pull those from test.parquet too so
    # build_subset's qrels-extras union can actually resolve them, or the reranker sees zero
    # positives for most queries regardless of the negative-sampling fix in reranker_pure.py.
    test_path = base / "data" / "splits" / "test.parquet"
    judged_ids = set(qrels_df["did"])
    test_df = pd.read_parquet(test_path)
    test_judged = test_df[test_df["acc"].isin(judged_ids)]
    corpus_full = pd.concat([corpus_full, test_judged], ignore_index=True).drop_duplicates(subset="acc")

    subset = build_subset(corpus_full, qrels_df, prcfg["subset_size"], prcfg["subset_seed"])
    subset = _add_synthetic_domains(subset)
    queries_df = _add_synthetic_domains(queries_df)
    print(f"[prepare_protein_rag_index] subset = {len(subset)} proteins "
          f"({subset['family_id'].nunique()} families)")

    index_dir = repo_path(prcfg["index_dir"])
    index_dir.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(index_dir / "corpus_subset.parquet")
    queries_df.to_parquet(index_dir / "queries.parquet")
    qrels_df.to_csv(index_dir / "qrels.csv", index=False)

    # --- sparse (BM25) ---
    print("[prepare_protein_rag_index] building BM25 index ...")
    bm25_dir = index_dir / "bm25"
    bm25 = BioAwareBM25(k1=prcfg["sparse"]["k1"], b=prcfg["sparse"]["b"],
                         motif_boost=prcfg["sparse"]["motif_boost"],
                         domain_boost=prcfg["sparse"]["domain_boost"],
                         index_dir=str(bm25_dir))
    bm25.index(subset)
    bm25.save(str(bm25_dir / "bm25_meta.pkl"))

    # --- dense (ESM-2) ---
    # Embedding is by far the most expensive step here (hours at real scale on CPU) and this
    # has now actually been interrupted mid-run by a session/machine restart — resumable by
    # design: if a dense index already exists AND its doc_ids exactly match this subset, reuse
    # it (and the embedding vectors reconstructed from it — see below) instead of re-embedding.
    dense_dir = index_dir / "dense"
    subset_accs = subset["acc"].tolist()
    cached_dense_meta = None
    if (dense_dir / "faiss.index").exists() and (dense_dir / "meta.pkl").exists():
        candidate_meta = joblib.load(dense_dir / "meta.pkl")
        if candidate_meta.get("doc_ids") == subset_accs and candidate_meta.get("model") == prcfg["dense_model"]:
            cached_dense_meta = candidate_meta

    embedder = ProteinEmbedder(model_name=prcfg["dense_model"])

    if cached_dense_meta is not None:
        print(f"[prepare_protein_rag_index] found existing dense index matching this exact "
              f"subset+model — reusing it (skips re-embedding {len(subset_accs)} sequences)")
        dense_index = faiss.read_index(str(dense_dir / "faiss.index"))
        embeddings = dense_index.reconstruct_n(0, dense_index.ntotal)  # already L2-normalized
        doc_ids = subset_accs
    else:
        print(f"[prepare_protein_rag_index] embedding with {prcfg['dense_model']} (this "
              f"downloads weights on first run) ...")
        n_docs = len(subset)
        nlist = min(256, max(4, int(n_docs ** 0.5)))
        dense = DenseRetriever(model_name=prcfg["dense_model"], index_type="IVFFlat", nlist=nlist)
        dense.embedder = embedder  # reuse the already-loaded model instead of loading it twice
        dense.build_index(subset, batch_size=16)
        dense.save(str(dense_dir))
        dense_index = faiss.read_index(str(dense_dir / "faiss.index"))
        embeddings = dense_index.reconstruct_n(0, dense_index.ntotal)
        doc_ids = dense.doc_ids

    # --- multi-vector (S_late) ---
    # NOT re-embedded via MultiVectorRetriever.index() (which would redo the exact same
    # forward passes a second time): with no per-domain boundary annotation in this dataset,
    # extract_segments() always falls back to one whole-sequence "full" segment per protein
    # (see _add_synthetic_domains's docstring) — i.e. each doc's one multi-vector "segment" IS
    # its dense embedding. Built directly from the vectors above instead.
    print("[prepare_protein_rag_index] building multi-vector index directly from the dense "
          "embeddings (mathematically the same single-segment vectors, no second embedding pass)")
    doc_vecs = [embeddings[i:i + 1] for i in range(len(doc_ids))]
    mv_dir = index_dir / "multivec"
    mv_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"doc_ids": doc_ids, "doc_vecs": doc_vecs}, mv_dir / "multivec.joblib")

    # --- reranker ---
    print("[prepare_protein_rag_index] training re-ranker ...")
    reranker = EvolutionReranker(min_mem_len=prcfg["mem"]["min_mem_len"])
    n_pairs = reranker.fit(queries_df, subset, qrels_df)
    print(f"[prepare_protein_rag_index] reranker trained on {n_pairs} pairwise examples")
    reranker.save(str(index_dir / "reranker.joblib"))

    print("[prepare_protein_rag_index] done.")


if __name__ == "__main__":
    main(load_config())
