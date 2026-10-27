import sys
from retriever import HybridRetriever

q = " ".join(sys.argv[1:])
r = HybridRetriever()

print(f"\n=== Top 8 hybrid results for: {q} ===")
for h in r.hybrid_search(q, k=8):
    m = h["meta"]
    print(f"\n[{m['source']} p.{m['page']}]\n{h['text'][:300]}")

print("\n=== Raw text search in AC_43.13-1B (is the topic in the corpus?) ===")
for term in ["stop drill", "stop-drill", "cracked", "crack"]:
    pages = sorted({m["page"] for d, m in zip(r.docs, r.metas)
                    if m["source"] == "AC_43.13-1B.pdf" and term in d.lower()})
    print(f"'{term}': {len(pages)} pages, first few: {pages[:15]}")