"""Score the DEPLOYED retrieval code (keyword vs hybrid) on an eval set, page-aware.
Run:  AZURE_SEARCH_ENDPOINT=... AZURE_SEARCH_KEY=... PYTHONPATH=.:src \
      python scripts/check_cloud_retrieval.py data/eval/questions_hard.json
"""
import json
import os
import sys

from cloudsearch import search as cs
from eval_metrics import expected_pages

ep, key = os.environ["AZURE_SEARCH_ENDPOINT"], os.environ["AZURE_SEARCH_KEY"]
items = [q for q in json.load(open(sys.argv[1])) if q.get("answerable", True)]


def score(fn):
    hit5 = hit10 = rr = 0
    for it in items:
        hits = fn(it["q"])
        ranks = [i for i, h in enumerate(hits, 1)
                 if (not it.get("source") or h["source"] == it["source"]) and h["page"] in expected_pages(it)]
        r = ranks[0] if ranks else None
        hit5 += bool(r and r <= 5)
        hit10 += bool(r and r <= 10)
        rr += 1 / r if r else 0
    n = len(items)
    return f"Hit@5 {hit5}/{n}  Hit@10 {hit10}/{n}  MRR {rr / n:.2f}"


print("keyword only :", score(lambda q: cs.keyword(ep, key, q, 10)))
print("hybrid (RRF) :", score(lambda q: cs.hybrid(ep, key, q, k=10)))
