import pytest
from fastapi.testclient import TestClient

from api import backend, main

HITS = [
    {"id": "a", "content": "Rivet spacing is 3 to 4 diameters.", "source": "AC 43.13-1B", "page": 100},
    {"id": "b", "content": "Edge distance is 2 diameters.", "source": "AC 43.13-1B", "page": 101},
]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setattr(backend, "search", lambda q, k=5: HITS)
    monkeypatch.setattr(backend, "generate", lambda q, hits: "Use 3 to 4 diameters [1].")
    monkeypatch.setattr(backend, "budget", backend.DailyBudget(100))
    return TestClient(main.app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ask_returns_cited_answer_and_sources(client):
    r = client.post("/ask", json={"question": "What is the rivet spacing?"})
    body = r.json()
    assert r.status_code == 200
    assert body["answer"] == "Use 3 to 4 diameters [1]."
    assert body["refused"] is False and body["llm_used"] is True
    assert [s["page"] for s in body["sources"]] == [100, 101]


def test_refusal_is_flagged(client, monkeypatch):
    monkeypatch.setattr(backend, "generate", lambda q, h: "I could not find it in the provided documents.")
    assert client.post("/ask", json={"question": "Who won the 1998 World Cup?"}).json()["refused"] is True


def test_no_hits_means_refused_without_calling_llm(client, monkeypatch):
    monkeypatch.setattr(backend, "search", lambda q, k=5: [])
    monkeypatch.setattr(backend, "generate", lambda q, h: pytest.fail("LLM must not be called"))
    body = client.post("/ask", json={"question": "something obscure"}).json()
    assert body["refused"] is True and body["llm_used"] is False and body["sources"] == []


def test_budget_exhausted_falls_back_to_passages(client, monkeypatch):
    monkeypatch.setattr(backend, "budget", backend.DailyBudget(0))
    body = client.post("/ask", json={"question": "What is the rivet spacing?"}).json()
    assert body["llm_used"] is False and len(body["sources"]) == 2


def test_llm_failure_falls_back_to_passages(client, monkeypatch):
    def boom(q, h):
        raise RuntimeError("quota")
    monkeypatch.setattr(backend, "generate", boom)
    body = client.post("/ask", json={"question": "What is the rivet spacing?"}).json()
    assert body["llm_used"] is False and len(body["sources"]) == 2


def test_search_failure_is_502(client, monkeypatch):
    def boom(q, k=5):
        raise RuntimeError("down")
    monkeypatch.setattr(backend, "search", boom)
    assert client.post("/ask", json={"question": "What is the rivet spacing?"}).status_code == 502


def test_validation_rejects_short_and_long_questions(client):
    assert client.post("/ask", json={"question": "hi"}).status_code == 422
    assert client.post("/ask", json={"question": "x" * 501}).status_code == 422
    assert client.post("/ask", json={"question": "valid question", "k": 99}).status_code == 422


def test_api_key_enforced_when_configured(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    body = {"question": "What is the rivet spacing?"}
    assert client.post("/ask", json=body).status_code == 401
    assert client.post("/ask", json=body, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/ask", json=body, headers={"X-API-Key": "secret"}).status_code == 200


def test_budget_counts_and_resets():
    b = backend.DailyBudget(2)
    assert b.take() and b.take() and not b.take()
