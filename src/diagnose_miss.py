import json

from retriever import HybridRetriever


def norm(s):
    return " ".join(s.replace("–", "-").replace("‑", "-").lower().split())


r = HybridRetriever()
n = len(r.ids)
rows = [x for x in json.load(open("data/eval/questions_hard.json")) if x["answerable"]][:2]


def pos(ids, target):
    return ids.index(target) + 1 if target in ids else ">1000"


for row in rows:
    key = norm(row["key"])
    targets = [
        cid
        for cid, d, m in zip(r.ids, r.docs, r.metas)
        if key in norm(d) and m["page"] == row["page"] and m["source"] == row["source"]
    ]
    print("\n" + "=" * 90)
    print("QUESTION:", row["q"])
    print("target chunks:", targets)

    vec = r.vector_search(row["q"], 1000)
    kw = r.keyword_search(row["q"], 1000)
    for tid in targets:
        idx = r.ids.index(tid)
        print(f"\n--- target {tid} ---")
        print("vector rank:", pos(vec, tid), "| keyword rank:", pos(kw, tid))
        print("hybrid rank, pool 20: ", end="")
        h20 = [h["id"] for h in r.hybrid_search(row["q"], k=10, pool=20)]
        print(pos(h20, tid) if tid in h20 else "not in top 10")
        print("hybrid rank, pool 100:", end=" ")
        h100 = [h["id"] for h in r.hybrid_search(row["q"], k=10, pool=100)]
        print(pos(h100, tid) if tid in h100 else "not in top 10")
        print("chunk text:", " ".join(r.docs[idx].split())[:700])

    print("\nTop 3 hybrid results instead:")
    for h in r.hybrid_search(row["q"], k=3):
        print(f"  [{h['meta']['source']} p.{h['meta']['page']}] {' '.join(h['text'].split())[:160]}")