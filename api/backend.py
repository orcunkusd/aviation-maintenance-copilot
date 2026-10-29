"""Cloud backend for the API: Azure AI Search (retrieval) + Gemini (generation).

All settings come from environment variables so the same code runs locally,
in Docker and on Azure Container Apps.
"""
import datetime as dt
import os
import threading

import requests

SYSTEM = (
    "You are an aviation maintenance documentation assistant. "
    "Answer ONLY using the numbered context passages. "
    "Cite every claim with the passage number in square brackets, like [2]. "
    "If the passages do not contain the answer, say you could not find it in the provided documents. "
    "Do not use outside knowledge. Keep answers short."
)


def search(query: str, k: int = 5):
    from cloudsearch import search as cs
    return cs.hybrid(os.environ["AZURE_SEARCH_ENDPOINT"], os.environ["AZURE_SEARCH_KEY"], query, k=k,
                     index=os.environ.get("AZURE_SEARCH_INDEX", "faa-docs"))


def build_context(hits):
    return "\n\n".join(
        f"[{i}] (source: {h['source']}, page {h['page']})\n{h['content']}" for i, h in enumerate(hits, 1)
    )


def _generation_config(model):
    cfg = {"temperature": 0, "maxOutputTokens": 800}
    if model.startswith("gemini-2.5"):
        cfg["thinkingConfig"] = {"thinkingBudget": 0}   # thinking tokens would eat the answer budget
    return cfg


def _extract_text(data):
    cand = data["candidates"][0]
    if cand.get("finishReason") == "MAX_TOKENS":
        raise ValueError("answer was truncated")        # callers fall back to the retrieved passages
    return cand["content"]["parts"][0]["text"].strip()


def generate(question: str, hits) -> str:
    model = os.environ.get("GEN_MODEL", "gemini-2.5-flash-lite")
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"], "Content-Type": "application/json"},
        json={
            "systemInstruction": {"parts": [{"text": SYSTEM}]},
            "contents": [{"role": "user", "parts": [
                {"text": f"Context passages:\n{build_context(hits)}\n\nQuestion: {question}"}]}],
            "generationConfig": _generation_config(model),
        },
        timeout=30,
    )
    r.raise_for_status()
    return _extract_text(r.json())


class DailyBudget:
    """Process-wide daily cap on LLM calls so the free quota cannot be drained."""

    def __init__(self, limit: int):
        self.limit, self.day, self.n = limit, dt.date.today(), 0
        self._lock = threading.Lock()

    def take(self) -> bool:
        with self._lock:
            today = dt.date.today()
            if today != self.day:
                self.day, self.n = today, 0
            if self.n >= self.limit:
                return False
            self.n += 1
            return True


budget = DailyBudget(int(os.environ.get("DAILY_LLM_LIMIT", "50")))
