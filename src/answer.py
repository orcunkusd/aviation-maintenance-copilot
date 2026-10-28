import os
import sys

SYSTEM = (
    "You are an aviation maintenance documentation assistant. "
    "Answer ONLY using the numbered context passages. "
    "Cite every claim with the passage number in square brackets, like [2]. "
    "If the passages do not contain the answer, say you could not find it in the provided documents. "
    "Do not use outside knowledge. Keep answers short."
)

_retriever = None


def get_retriever():
    """Load the retriever on first use (RERANK=1 adds the cross-encoder)."""
    global _retriever
    if _retriever is None:
        if os.environ.get("RERANK", "0") == "1":
            from rerank_retriever import RerankRetriever
            _retriever = RerankRetriever()
        else:
            from retriever import HybridRetriever
            _retriever = HybridRetriever()
    return _retriever


def build_context(hits):
    return "\n\n".join(
        f"[{i+1}] (source: {h['meta']['source']}, page {h['meta']['page']})\n{h['text']}"
        for i, h in enumerate(hits)
    )


def answer(question, k=5):
    from llm import generate
    r = get_retriever()
    hits = r.search(question, k=k) if hasattr(r, "search") else r.hybrid_search(question, k=k)
    text = generate(SYSTEM, f"Context passages:\n{build_context(hits)}\n\nQuestion: {question}")
    return text, hits


if __name__ == "__main__":
    q = " ".join(sys.argv[1:])
    text, hits = answer(q)
    print(text)
    print("\nSources:")
    for i, h in enumerate(hits):
        print(f" [{i+1}] {h['meta']['source']} p.{h['meta']['page']}")
