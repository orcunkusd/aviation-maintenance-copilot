"""Reciprocal rank fusion (RRF) over several ranked lists of ids."""


def rrf_fuse(rankings, rrf_k=60, k=None):
    fused = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            fused[cid] = fused.get(cid, 0) + 1 / (rrf_k + rank + 1)
    top = sorted(fused, key=fused.get, reverse=True)
    return top if k is None else top[:k]
