import pytest

pytest.importorskip("langgraph")
from agent import check_citations, merge_hits, parse_json, run_agent  # noqa: E402


def hit(i, text="t"):
    return {"id": f"h{i}", "text": text, "meta": {"source": "AC.pdf", "page": i}}


def test_parse_json_handles_fences_and_garbage():
    assert parse_json('```json\n{"queries": ["a"]}\n```', {}) == {"queries": ["a"]}
    assert parse_json("no json here", {"x": 1}) == {"x": 1}
    assert parse_json('{"broken": ', {"x": 1}) == {"x": 1}


def test_merge_hits_round_robin_dedupes_and_limits():
    merged = merge_hits([[hit(1), hit(2)], [hit(1), hit(3)]], limit=3)
    assert [h["id"] for h in merged] == ["h1", "h2", "h3"]


def test_check_citations():
    assert check_citations("Use 3 diameters [1][2].", 2)
    assert not check_citations("Use 3 diameters [3].", 2)
    assert not check_citations("Use 3 diameters.", 2)           # claims need citations
    assert check_citations("I could not find it in the provided documents.", 2)


class FakeLLM:
    """Returns scripted replies in order and records calls."""
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def __call__(self, system, user):
        self.calls.append(system[:20])
        return self.replies.pop(0)


def test_happy_path_uses_three_llm_calls():
    llm = FakeLLM(['{"queries": ["rivet spacing"]}', '{"sufficient": true}', "Use 3 diameters [1]."])
    out = run_agent("What is rivet spacing?", lambda q, k: [hit(1), hit(2)], llm)
    assert out["llm_calls"] == 3 and out["citations_ok"] is True
    assert out["answer"] == "Use 3 diameters [1]."


def test_retry_with_better_query_then_answer():
    seen = []

    def retrieve(q, k):
        seen.append(q)
        return [hit(1)] if q == "better words" else [hit(9)]

    llm = FakeLLM(['{"queries": ["bad"]}', '{"sufficient": false, "better_query": "better words"}',
                   '{"sufficient": true}', "Answer [1]."])
    out = run_agent("q?", retrieve, llm)
    assert seen == ["bad", "better words"]
    assert out["llm_calls"] == 4 and out["retries"] == 1


def test_only_one_retry_even_if_still_insufficient():
    llm = FakeLLM(['{"queries": ["a"]}', '{"sufficient": false, "better_query": "b"}',
                   '{"sufficient": false, "better_query": "c"}', "I could not find it in the provided documents."])
    out = run_agent("q?", lambda q, k: [hit(1)], llm)
    assert out["llm_calls"] == 4 and out["citations_ok"] is True


def test_no_retry_when_better_query_is_missing_or_unchanged():
    llm = FakeLLM(['{"queries": ["a"]}', '{"sufficient": false, "better_query": "a"}', "I could not find it in the provided documents."])
    out = run_agent("q?", lambda q, k: [hit(1)], llm)
    assert out["llm_calls"] == 3 and out["retries"] == 0


def test_bad_llm_json_falls_back_to_original_question():
    seen = []
    llm = FakeLLM(["garbage", "garbage", "Answer [1]."])
    run_agent("my question", lambda q, k: seen.append(q) or [hit(1)], llm)
    assert seen == ["my question"]


def test_invalid_citation_is_flagged():
    llm = FakeLLM(['{"queries": ["a"]}', '{"sufficient": true}', "Made up [7]."])
    assert run_agent("q?", lambda q, k: [hit(1)], llm)["citations_ok"] is False
