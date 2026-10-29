import pytest

pytest.importorskip("langgraph")
import compare_systems as cs  # noqa: E402

ITEMS = [{"q": "a?", "answerable": True, "source": "AC.pdf", "page": 1},
         {"q": "b?", "answerable": False}]


def test_run_system_caches_and_scores(monkeypatch, tmp_path):
    monkeypatch.setattr(cs, "CACHE", str(tmp_path / "cache.json"))
    calls = []

    def run(q):
        calls.append(q)
        hits = [{"id": "x", "text": "t", "meta": {"source": "AC.pdf", "page": 1}}]
        return ("Use 3 [1]." if q == "a?" else "I could not find it."), hits, 2

    cache = {}
    rows = cs.run_system("agent", run, ITEMS, cache, "m")
    assert rows[0]["citation_correct"] is True and rows[1]["refused"] is True
    cs.run_system("agent", run, ITEMS, cache, "m")      # second time: served from cache
    assert calls == ["a?", "b?"]


def test_quota_error_is_reported_cleanly(monkeypatch):
    monkeypatch.setattr(cs.time, "sleep", lambda s: None)

    def boom():
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    with pytest.raises(cs.QuotaExhausted):
        cs.with_retry(boom, tries=2)


def test_other_errors_are_not_swallowed(monkeypatch):
    def boom():
        raise ValueError("bug")

    with pytest.raises(ValueError):
        cs.with_retry(boom)


def test_temporary_overload_is_retried_then_succeeds(monkeypatch):
    monkeypatch.setattr(cs.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("503 UNAVAILABLE. This model is currently experiencing high demand.")
        return "ok"

    assert cs.with_retry(flaky) == "ok" and calls["n"] == 3


def test_persistent_overload_stops_cleanly(monkeypatch):
    monkeypatch.setattr(cs.time, "sleep", lambda s: None)

    def down():
        raise RuntimeError("503 UNAVAILABLE")

    with pytest.raises(cs.QuotaExhausted):
        cs.with_retry(down, tries=3)


def test_deadline_exceeded_is_retried(monkeypatch):
    monkeypatch.setattr(cs.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("504 DEADLINE_EXCEEDED. Deadline expired before operation could complete.")
        return "ok"

    assert cs.with_retry(flaky) == "ok"
