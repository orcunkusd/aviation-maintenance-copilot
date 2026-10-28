import json
import time

from rerank_retriever import RerankRetriever


def norm(s):
    return " ".join(s.replace("–", "-").replace("‑", "-").lower().split())


rr = RerankRetriever()
rr.search("warmup", k=5)

SETS = {"easy": "data/eval/questions.json", "hard": "data/eval/questions_hard.json"}
CANDS = [60, 40, 30, 20]


def rank_of(hits, row):
    key = norm(row["key"])
    pages = row.get("pages") or ([row["page"]] if row.get("page") else [])
    for i, h in enumerate(hits, start=1):
        ok_place = not pages or (
            h["meta"]["page"] in pages and h["meta"]["source"] == row["source"]
        )
        if key in norm(h["text"]) and ok_place:
            return i
    return None


print(f"{'candidates':>10} {'set':<6} {'Hit@5':>6} {'MRR':>5} {'search':>9} {'rerank':>9}")
for c in CANDS:
    for name, path in SETS.items():
        rows = [r for r in json.load(open(path)) if r["answerable"]]
        ranks, t_search, t_rerank = [], [], []
        for row in rows:
            t0 = time.time()
            cands = rr.base.hybrid_search(row["q"], k=c, pool=50)
            t1 = time.time()
            scores = rr.model.predict([(row["q"], x["text"]) for x in cands])
            t2 = time.time()
            ordered = [
                x for _, x in sorted(zip(scores, cands), key=lambda p: p[0], reverse=True)
            ][:10]
            ranks.append(rank_of(ordered, row))
            t_search.append(t1 - t0)
            t_rerank.append(t2 - t1)
        n = len(rows)
        h5 = sum(1 for r in ranks if r and r <= 5)
        mrr = sum(1 / r for r in ranks if r) / n
        print(
            f"{c:>10} {name:<6} {h5:>2}/{n:<3} {mrr:>5.2f} "
            f"{sum(t_search) / n:>8.2f}s {sum(t_rerank) / n:>8.2f}s"
        )