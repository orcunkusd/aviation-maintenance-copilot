import sys
from llm import generate
import os

from retriever import HybridRetriever

USE_RERANK = os.environ.get("RERANK", "1") == "1"
if USE_RERANK:
    from rerank_retriever import RerankRetriever

    retriever = RerankRetriever()
else:
    retriever = HybridRetriever()

SYSTEM = (
    "You are an aviation maintenance documentation assistant. "
    "Answer ONLY using the numbered context passages. "
    "Cite every claim with the passage number in square brackets, like [2]. "
    "If the passages do not contain the answer, say you could not find it in the provided documents. "
    "Do not use outside knowledge. Keep answers short."
)


def answer(question, k=5):
    hits = retriever.search(question, k=k) if USE_RERANK else retriever.hybrid_search(question, k=k)
    context = "\n\n".join(
        f"[{i+1}] (source: {h['meta']['source']}, page {h['meta']['page']})\n{h['text']}"
        for i, h in enumerate(hits)
    )
    text = generate(SYSTEM, f"Context passages:\n{context}\n\nQuestion: {question}")
    return text, hits


if __name__ == "__main__":
    q = " ".join(sys.argv[1:])
    text, hits = answer(q)
    print(text)
    print("\nSources:")
    for i, h in enumerate(hits):
        print(f" [{i+1}] {h['meta']['source']} p.{h['meta']['page']}")