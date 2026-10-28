"""Hybrid retrieval against Azure AI Search, without PyTorch.

keyword (BM25 on Azure) + vector (MiniLM query embedding from fastembed/ONNX) -> reciprocal rank fusion.
Same recipe as the local HybridRetriever: pool of 20 from each, rrf_k = 60.
If the embedding step fails, it falls back to keyword-only so the service stays up.
"""
import logging

import requests

log = logging.getLogger("copilot")

API_VERSION = "2023-11-01"
SELECT = "id,content,source,page"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_embedder = None


def embed_query(text):
    """384-dim MiniLM embedding (same model that built the index), loaded on first use."""
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding
        _embedder = TextEmbedding(MODEL_NAME)
    return [float(x) for x in next(iter(_embedder.embed([text])))]


def rrf_fuse(rankings, rrf_k=60, k=None):
    fused = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            fused[cid] = fused.get(cid, 0) + 1 / (rrf_k + rank + 1)
    top = sorted(fused, key=fused.get, reverse=True)
    return top if k is None else top[:k]


def _search(endpoint, key, body, index):
    r = requests.post(
        f"{endpoint.rstrip('/')}/indexes/{index}/docs/search?api-version={API_VERSION}",
        headers={"api-key": key, "Content-Type": "application/json"},
        json=body, timeout=20)
    r.raise_for_status()
    return r.json().get("value", [])


def keyword(endpoint, key, query, k=20, index="faa-docs"):
    return _search(endpoint, key, {"search": query, "top": k, "select": SELECT, "queryType": "simple"}, index)


def vector(endpoint, key, query, k=20, index="faa-docs", embed=embed_query):
    body = {"vectorQueries": [{"kind": "vector", "vector": embed(query), "fields": "embedding", "k": k}],
            "select": SELECT, "top": k}
    return _search(endpoint, key, body, index)


def hybrid(endpoint, key, query, k=5, pool=20, index="faa-docs", embed=embed_query, rrf_k=60):
    kw = keyword(endpoint, key, query, pool, index)
    try:
        vec = vector(endpoint, key, query, pool, index, embed)
    except Exception as exc:                      # embedding model or vector query failed
        log.error("vector search failed (%s); using keyword results only", type(exc).__name__)
        return kw[:k]
    by_id = {}
    for hit in kw + vec:
        by_id.setdefault(hit["id"], hit)
    top = rrf_fuse([[h["id"] for h in vec], [h["id"] for h in kw]], rrf_k=rrf_k, k=k)
    return [by_id[i] for i in top]
