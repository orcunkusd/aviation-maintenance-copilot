import json
from retriever import HybridRetriever


def norm(s):
    return " ".join(s.replace("–", "-").replace("‑", "-").lower().split())


r = HybridRetriever()
rows = json.load(open("data/eval/questions.json"))
n = hit5 = hit10 = 0

for row in rows:
    if not row["answerable"]:
        continue
    key = norm(row["key"])
    hits = r.hybrid_search(row["q"], k=10)
    rank = None
    for i, h in enumerate(hits, start=1):
        right_place = not row.get("page") or (
            h["meta"]["page"] == row["page"] and h["meta"]["source"] == row["source"]
        )
        if key in norm(h["text"]) and right_place:
            rank = i
            break
    n += 1
    hit5 += rank is not None and rank <= 5
    hit10 += rank is not None
    print(f"rank={rank if rank else 'MISS':>4} | {row['q'][:75]}")

print(f"\nHit@5:  {hit5}/{n}")
print(f"Hit@10: {hit10}/{n}")