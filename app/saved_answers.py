"""Pre-generated cited answers for the demo's example questions.

Used only when the live model is unavailable, and always labelled as saved.
"""
import json
from pathlib import Path


def norm(q: str) -> str:
    return " ".join((q or "").lower().replace("?", " ").split())


def load(path) -> list:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [d for d in data if d.get("question") and d.get("answer") and d.get("sources")]


def find(saved: list, question: str):
    key = norm(question)
    for d in saved:
        if norm(d["question"]) == key:
            return d
    return None
