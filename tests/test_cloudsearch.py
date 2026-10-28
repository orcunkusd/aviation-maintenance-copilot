from cloudsearch import search as cs


def H(i, text="t"):
    return {"id": f"h{i}", "content": text, "source": "AC.pdf", "page": i}


def fake_search_factory(kw_ids, vec_ids, seen):
    def fake(endpoint, key, body, index):
        seen.append(body)
        ids = vec_ids if "vectorQueries" in body else kw_ids
        return [H(i) for i in ids]
    return fake


def test_rrf_matches_local_recipe():
    assert cs.rrf_fuse([["x", "a"], ["y", "a"]])[0] == "a"
    assert cs.rrf_fuse([["a", "b", "c"]], k=2) == ["a", "b"]


def test_hybrid_sends_keyword_and_vector_queries_and_fuses(monkeypatch):
    seen = []
    monkeypatch.setattr(cs, "_search", fake_search_factory([1, 2, 3], [3, 4, 1], seen))
    hits = cs.hybrid("https://x", "key", "rivet spacing", k=3, embed=lambda q: [0.1, 0.2])
    # h1 and h3 appear in both lists; ties go to the vector ranking, as in the local retriever
    assert [h["id"] for h in hits] == ["h3", "h1", "h4"]
    kw_body = next(b for b in seen if "vectorQueries" not in b)
    vec_body = next(b for b in seen if "vectorQueries" in b)
    assert kw_body["search"] == "rivet spacing" and kw_body["top"] == 20
    vq = vec_body["vectorQueries"][0]
    assert vq["kind"] == "vector" and vq["fields"] == "embedding" and vq["k"] == 20 and vq["vector"] == [0.1, 0.2]


def test_falls_back_to_keyword_when_embedding_fails(monkeypatch):
    seen = []
    monkeypatch.setattr(cs, "_search", fake_search_factory([1, 2, 3], [9], seen))

    def broken(q):
        raise RuntimeError("model missing")

    hits = cs.hybrid("https://x", "key", "q", k=2, embed=broken)
    assert [h["id"] for h in hits] == ["h1", "h2"]


def test_hybrid_returns_full_hit_fields(monkeypatch):
    monkeypatch.setattr(cs, "_search", fake_search_factory([1], [1], []))
    hit = cs.hybrid("https://x", "key", "q", embed=lambda q: [0.0])[0]
    assert {"id", "content", "source", "page"} <= set(hit)
