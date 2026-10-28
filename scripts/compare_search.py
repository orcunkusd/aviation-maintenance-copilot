import sys
from retriever import HybridRetriever

q = " ".join(sys.argv[1:])
r = HybridRetriever()
index = {cid: i for i, cid in enumerate(r.ids)}

def show(title, ids):
    print(f"\n=== {title} ===")
    for cid in ids[:3]:
        m = r.metas[index[cid]]
        print(f"[{m['source']} p.{m['page']}] {r.docs[index[cid]][:150]!r}")

show("VECTOR ONLY", r.vector_search(q, 3))
show("KEYWORD ONLY", r.keyword_search(q, 3))
show("HYBRID", [h["id"] for h in r.hybrid_search(q, 3)])