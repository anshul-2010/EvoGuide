"""Reimplementation of ProteinRAG's re_ranker/reranker.py::EvolutionReranker that:
  1. uses our pure-Python compute_all_features (tools/mem_features_pure.py) instead of the
     original's MUMmer-backed one, and
  2. adds save()/load() (the original class has neither — confirmed by direct code read).

Functionally identical otherwise (same FEATURE_COLS, same GradientBoostingClassifier config,
same pairwise-training-pair construction), so it's a drop-in that keeps ProteinRAG's own
approach rather than inventing a new re-ranking scheme.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from tools.mem_features_pure import compute_all_features

FEATURE_COLS = [
    "mem_count", "mem_total_len", "mem_max_len",
    "mem_coverage", "mem_mean_len", "domain_overlap",
    "sparse_score", "dense_score", "len_ratio",
]


class EvolutionReranker:
    def __init__(self, min_mem_len: int = 5):
        self.model = GradientBoostingClassifier(n_estimators=200, max_depth=4)
        self.scaler = StandardScaler()
        self.fitted = False
        self.min_mem_len = min_mem_len

    def _build_training_pairs(self, queries: pd.DataFrame, corpus: pd.DataFrame,
                               qrels: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Two deviations from the original repo's pairwise-pair builder, both confirmed by
        actually running it, not assumed:
        1. It looks for explicit rel==0 rows in qrels to use as negatives — but this qrels.txt
           only ever contains rel==2 rows (standard TREC practice: anything not listed for a
           query is implicitly non-relevant). The original logic therefore always found zero
           negatives. Negatives here are randomly sampled from corpus docs NOT judged relevant
           for that query instead.
        2. It only ever emits label=1 rows (the pos-minus-neg feature delta) — never a label=0
           counterpart — so GradientBoostingClassifier.fit() always raised "y contains 1
           class". This mirrors each delta with the opposite sign and label=0, the standard
           pairwise-to-binary-classification reformulation.
        Returns (X, y) directly rather than a DataFrame of pos_*/neg_* columns, since the
        model trains on the delta, not the raw pos/neg feature values.
        """
        X_rows: list[list[float]] = []
        y_rows: list[int] = []
        corpus_map = corpus.set_index("acc").to_dict("index")
        all_accs = corpus["acc"].tolist()
        qrels_map = qrels.groupby("qid")[["did", "rel"]].apply(
            lambda g: dict(zip(g.did, g.rel))).to_dict()
        rng = np.random.default_rng(42)
        for _, q in queries.iterrows():
            qid = q["acc"]
            rels = qrels_map.get(qid, {})
            positives = [d for d, r in rels.items() if r >= 2 and d in corpus_map]
            if not positives:
                continue
            judged = set(rels.keys())
            neg_pool = [a for a in all_accs if a not in judged]
            if not neg_pool:
                continue
            for pos in positives[:5]:
                neg_sample = rng.choice(neg_pool, size=min(3, len(neg_pool)), replace=False)
                for neg in neg_sample:
                    pos_doc = corpus_map.get(pos, {})
                    neg_doc = corpus_map.get(neg, {})
                    if not pos_doc or not neg_doc:
                        continue
                    pf = compute_all_features(q["seq"], pos_doc.get("seq", ""),
                                               q.get("domains", []), pos_doc.get("domains", []),
                                               min_mem_len=self.min_mem_len)
                    nf = compute_all_features(q["seq"], neg_doc.get("seq", ""),
                                               q.get("domains", []), neg_doc.get("domains", []),
                                               min_mem_len=self.min_mem_len)
                    delta = [pf[c] - nf[c] for c in FEATURE_COLS]
                    X_rows.append(delta)
                    y_rows.append(1)
                    X_rows.append([-d for d in delta])
                    y_rows.append(0)
        return np.array(X_rows), np.array(y_rows)

    def fit(self, queries: pd.DataFrame, corpus: pd.DataFrame, qrels: pd.DataFrame) -> int:
        X, y = self._build_training_pairs(queries, corpus, qrels)
        if len(X) == 0:
            raise RuntimeError(
                "No trainable pairs built — check that queries/corpus/qrels accessions "
                "actually overlap (see tools/prepare_protein_rag_index.py's subset logic).")
        X = self.scaler.fit_transform(X)
        self.model.fit(X, y)
        self.fitted = True
        return len(X)

    def rerank(self, query_seq: str, candidates: list[tuple[str, float]], corpus_map: dict,
               q_domains: list[str] | None = None,
               sparse_scores: dict | None = None,
               dense_scores: dict | None = None) -> list[tuple[str, dict]]:
        """Unlike the original, returns (doc_id, feature_dict) so decomposed scores survive —
        the caller (protein_rag_tool.query_all) attaches S_final from feature_dict['s_final'].

        Also unlike the original (which passes the same fused init_score as both
        sparse_score and dense_score into compute_all_features), this accepts the real
        per-doc sparse/dense score maps when available and only falls back to the fused
        candidate score if a doc_id is missing from one of them.
        """
        if not self.fitted:
            raise RuntimeError("Call fit() before rerank()")
        sparse_scores = sparse_scores or {}
        dense_scores = dense_scores or {}
        rows = []
        for doc_id, init_score in candidates:
            doc = corpus_map.get(doc_id, {})
            f = compute_all_features(query_seq, doc.get("seq", ""),
                                      q_domains or [], doc.get("domains", []),
                                      sparse_score=sparse_scores.get(doc_id, init_score),
                                      dense_score=dense_scores.get(doc_id, init_score),
                                      min_mem_len=self.min_mem_len)
            rows.append((doc_id, f))
        X = self.scaler.transform(np.array([[f[c] for c in FEATURE_COLS] for _, f in rows]))
        scores = self.model.predict_proba(X)[:, 1]
        out = []
        for (doc_id, f), s in zip(rows, scores):
            f = dict(f)
            f["s_final"] = float(s)
            out.append((doc_id, f))
        return sorted(out, key=lambda x: -x[1]["s_final"])

    def save(self, path: str) -> None:
        joblib.dump({"model": self.model, "scaler": self.scaler, "fitted": self.fitted,
                     "min_mem_len": self.min_mem_len}, path)

    @classmethod
    def load(cls, path: str) -> "EvolutionReranker":
        state = joblib.load(path)
        obj = cls(min_mem_len=state["min_mem_len"])
        obj.model = state["model"]
        obj.scaler = state["scaler"]
        obj.fitted = state["fitted"]
        return obj
