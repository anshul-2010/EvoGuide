"""Runtime adapter: wraps the real ProteinRAG repo's sparse/dense/fusion/multi-vector stages
plus our own reranker_pure.EvolutionReranker into ONE query_all() call that returns a ranked
list with every decomposed score kept (S_sparse, S_dense, S_hybrid, S_late, F_MEM/F_domain,
S_final) — this orchestrating function does not exist upstream (HybridRetriever.query() and
EvolutionReranker.rerank() each discard the intermediate scores; confirmed by direct code
read). Requires tools/prepare_protein_rag_index.py to have been run first.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_config, repo_path  # noqa: E402
from tools import protein_rag_paths  # noqa: E402
from tools.base import ScientificTool, ToolResult  # noqa: E402
from tools.reranker_pure import EvolutionReranker  # noqa: E402


class ProteinRAGTool(ScientificTool):
    name = "protein_rag"

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.base = protein_rag_paths.inject(self.config)
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        import faiss
        from bio_bm25 import BioAwareBM25
        from embedder import ProteinEmbedder
        from dense_retriever import DenseRetriever
        from late_interaction import MultiVectorRetriever

        prcfg = self.config["protein_rag"]
        index_dir = repo_path(prcfg["index_dir"])
        if not index_dir.exists():
            raise RuntimeError(
                f"{index_dir} not found — run tools/prepare_protein_rag_index.py first.")

        self.corpus = pd.read_parquet(index_dir / "corpus_subset.parquet")
        self.corpus_map = self.corpus.set_index("acc").to_dict("index")

        self.bm25 = BioAwareBM25.load(str(index_dir / "bm25" / "bm25_meta.pkl"))

        self.embedder = ProteinEmbedder(model_name=prcfg["dense_model"])
        self.dense = DenseRetriever(model_name=prcfg["dense_model"])
        self.dense.embedder = self.embedder  # reuse the loaded model, don't load it twice
        self.dense.index = faiss.read_index(str(index_dir / "dense" / "faiss.index"))
        dense_meta = joblib.load(index_dir / "dense" / "meta.pkl")
        self.dense.doc_ids = dense_meta["doc_ids"]

        mv_state = joblib.load(index_dir / "multivec" / "multivec.joblib")
        self.mv = MultiVectorRetriever(embedder=self.embedder)
        self.mv.doc_ids = mv_state["doc_ids"]
        self.mv.doc_vecs = mv_state["doc_vecs"]
        self.mv._flat_to_doc = []
        flat_vecs = []
        for i, vecs in enumerate(self.mv.doc_vecs):
            for _ in vecs:
                self.mv._flat_to_doc.append(i)
            flat_vecs.append(vecs)
        all_vecs = np.concatenate(flat_vecs, axis=0)
        self.mv._flat_index = faiss.IndexFlatIP(all_vecs.shape[1])
        self.mv._flat_index.add(all_vecs)

        self.reranker = EvolutionReranker.load(str(index_dir / "reranker.joblib"))
        self._loaded = True

    @staticmethod
    def _homology_tier(mem_coverage: float) -> str:
        """Proxy for close/intermediate/remote since this pipeline never computes real BLAST
        E-values anywhere (confirmed by code read) — MEM coverage substitutes as the distance
        signal. Thresholds loosely follow the paper's %ID bands (>40% / 25-40% / <25%)."""
        if mem_coverage >= 0.4:
            return "close"
        if mem_coverage >= 0.2:
            return "intermediate"
        return "remote"

    def query_all(self, seq: str, domains: list[str] | None = None,
                   top_k: int | None = None) -> list[dict]:
        self._load()
        prcfg = self.config["protein_rag"]
        top_k = top_k or prcfg["rerank"]["top_k_final"]

        from hybrid_retriever import linear_fusion, reciprocal_rank_fusion

        sparse_res = self.bm25.query(seq, domains, top_k=prcfg["sparse"]["top_k"])
        dense_res = self.dense.query(seq, top_k=prcfg["dense"]["top_k"])
        sparse_map = dict(sparse_res)
        dense_map = dict(dense_res)

        if prcfg["fusion"]["method"] == "linear":
            fused = linear_fusion(sparse_res, dense_res, alpha=prcfg["fusion"]["alpha"])
        else:
            fused = reciprocal_rank_fusion(sparse_res, dense_res)
        fused_map = dict(fused)

        try:
            mv_res = self.mv.query(seq, domain_hits=None,
                                    n_candidates=prcfg["multivec"]["top_k"] * 4,
                                    top_k=prcfg["multivec"]["top_k"])
            late_map = dict(mv_res)
        except Exception:
            late_map = {}

        rerank_pool_size = max(top_k * 5, 50)
        reranked = self.reranker.rerank(
            seq, fused[:rerank_pool_size], self.corpus_map, q_domains=domains,
            sparse_scores=sparse_map, dense_scores=dense_map)

        results = []
        for doc_id, feat in reranked[:top_k]:
            doc = self.corpus_map.get(doc_id, {})
            results.append({
                "acc": doc_id,
                "family_id": doc.get("family_id"),
                "family_name": doc.get("family_name"),
                "clan_id": doc.get("clan_id"),
                "s_sparse": sparse_map.get(doc_id, 0.0),
                "s_dense": dense_map.get(doc_id, 0.0),
                "s_hybrid": fused_map.get(doc_id, 0.0),
                "s_late": late_map.get(doc_id),  # None if not in the multi-vector candidate set
                "f_mem_count": feat["mem_count"],
                "f_mem_coverage": feat["mem_coverage"],
                "f_domain_overlap": feat["domain_overlap"],
                "s_final": feat["s_final"],
                "homology_tier": self._homology_tier(feat["mem_coverage"]),
            })
        return results

    def run(self, seq: str | None = None, domains: list[str] | None = None,
            top_k: int | None = None, **kwargs) -> ToolResult:
        if not seq:
            return ToolResult(self.name, ok=False, error="query_all requires 'seq'")
        try:
            results = self.query_all(seq, domains, top_k)
        except Exception as e:  # noqa: BLE001
            return ToolResult(self.name, ok=False, error=str(e))
        return ToolResult(self.name, ok=True, output={"results": results})


if __name__ == "__main__":
    tool = ProteinRAGTool()
    test_seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWELVMGDGTRQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"
    result = tool.run(seq=test_seq)
    import json
    print(json.dumps(result.output if result.ok else {"error": result.error}, indent=2, default=str)[:2000])
