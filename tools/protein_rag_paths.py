"""ProteinRAG's repo has no __init__.py anywhere and uses flat, non-relative imports
(e.g. bio_bm25.py does `from kmer_tokenizer import ProteinTokenizer`; dense_retriever.py and
late_interaction.py both do `from embedder import ProteinEmbedder`). Confirmed by direct code
read — this is not pip-installable as a package as-is. Every submodule dir needs to be on
sys.path for these bare imports to resolve, regardless of which module triggers them.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SUBMODULES = ["sparse_retriever", "dense_retriever", "multi_vector", "re_ranker",
               "hybrid_fusion", "evaluation"]


def inject(config: dict) -> Path:
    """Only the submodule dirs go on sys.path — NOT the repo root. `dense_retriever/` contains
    a file also named `dense_retriever.py` (defining class DenseRetriever); if the repo root
    were also on sys.path, `import dense_retriever` becomes ambiguous between that file and the
    `dense_retriever/` directory-as-namespace-package, and whichever resolves first breaks the
    other resolution style (confirmed by hitting exactly this: "'dense_retriever' is not a
    package"). Submodule-dir-only injection matches the repo's own internal flat-import
    convention (`from kmer_tokenizer import ProteinTokenizer`, `from embedder import
    ProteinEmbedder`) — callers here must use that same flat style, e.g. `from dense_retriever
    import DenseRetriever` (resolves to the file), never `from dense_retriever.dense_retriever
    import DenseRetriever`.
    """
    base = Path(config["external"]["protein_rag_path"]).resolve()
    for sub in _SUBMODULES:
        p = str(base / sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    return base
