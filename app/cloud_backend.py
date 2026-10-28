"""Lightweight cloud backend for the Streamlit demo.

Retrieval : Azure AI Search (REST, full-text/BM25 on the free tier)
Generation: Gemini (REST) with a daily cap, falling back to retrieval-only.
Only `requests` is needed, so the app installs in seconds on Streamlit Cloud.
"""
import datetime as dt
import requests

SYSTEM = ("Answer ONLY using the numbered context passages. Cite passages like [1]. "
          "If the passages do not contain the answer, say you could not find it. Keep answers short.")


def search(endpoint: str, key: str, query: str, k: int = 5, index: str = "faa-docs"):
    r = requests.post(
        f"{endpoint.rstrip('/')}/indexes/{index}/docs/search?api-version=2023-11-01",
        headers={"api-key": key, "Content-Type": "application/json"},
        json={"search": query, "top": k, "select": "id,content,source,page", "queryType": "simple"},
        timeout=20)
    r.raise_for_status()
    return r.json().get("value", [])


def generate(api_key: str, question: str, hits, model: str = "gemini-2.5-flash-lite") -> str:
    ctx = "\n\n".join(f"[{i}] ({h['source']} p.{h['page']}) {h['content']}" for i, h in enumerate(hits, 1))
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        json={"systemInstruction": {"parts": [{"text": SYSTEM}]},
              "contents": [{"role": "user", "parts": [{"text": f"Context:\n{ctx}\n\nQuestion: {question}"}]}],
              "generationConfig": {"temperature": 0, "maxOutputTokens": 400}},
        timeout=30)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


class DailyBudget:
    """Process-wide counter (shared by all visitors) so the free LLM quota can't be drained."""
    def __init__(self, limit: int):
        self.limit, self.day, self.n = limit, dt.date.today(), 0

    def take(self) -> bool:
        today = dt.date.today()
        if today != self.day:
            self.day, self.n = today, 0
        if self.n >= self.limit:
            return False
        self.n += 1
        return True
