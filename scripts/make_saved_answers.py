"""Generate saved answers for the demo's example questions.

Run once, on a day the Gemini quota is available, from the repo root:
  AZURE_SEARCH_ENDPOINT=... AZURE_SEARCH_KEY=... GEMINI_API_KEY=... python scripts/make_saved_answers.py
Read the printed answers before committing app/saved_answers.json.
"""
import datetime
import inspect
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))
from cloudsearch import search as cs
from cloud import cloud_backend as be

EXAMPLES = ["What is the minimum rivet spacing?", "How is corrosion on aluminum alloy removed?",
            "How close to a pulley may a splice in a control cable be located?"]


def main():
    endpoint, skey, gkey = (os.environ.get(k) for k in ("AZURE_SEARCH_ENDPOINT", "AZURE_SEARCH_KEY", "GEMINI_API_KEY"))
    if not (endpoint and skey and gkey):
        sys.exit("Set AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KEY and GEMINI_API_KEY")
    model = inspect.signature(be.generate).parameters["model"].default
    out = []
    for q in EXAMPLES:
        hits = cs.hybrid(endpoint, skey, q, k=5)
        answer = be.generate(gkey, q, hits)
        print(f"\n=== {q}\n{answer}\n")
        out.append({
            "question": q, "answer": answer, "model": model,
            "date": datetime.date.today().isoformat(),
            "sources": [{"source": h["source"], "page": h["page"], "content": h["content"]} for h in hits],
        })
    path = ROOT / "app" / "saved_answers.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(out)} answers to {path}. Read them, then commit the file.")


if __name__ == "__main__":
    main()
