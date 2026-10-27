import os, sys
from google import genai
from google.genai import types
from retriever import HybridRetriever

MODEL = "gemini-2.5-flash"
client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options=types.HttpOptions(timeout=60000),
)
retriever = HybridRetriever()

SYSTEM = (
    "You are an aviation maintenance documentation assistant. "
    "Answer ONLY using the numbered context passages. "
    "Cite every claim with the passage number in square brackets, like [2]. "
    "If the passages do not contain the answer, say you could not find it in the provided documents. "
    "Do not use outside knowledge. Keep answers short."
)

def answer(question, k=5):
    hits = retriever.hybrid_search(question, k=k)
    context = "\n\n".join(
        f"[{i+1}] (source: {h['meta']['source']}, page {h['meta']['page']})\n{h['text']}"
        for i, h in enumerate(hits)
    )
    resp = client.models.generate_content(
        model=MODEL,
        contents=f"Context passages:\n{context}\n\nQuestion: {question}",
        config=types.GenerateContentConfig(system_instruction=SYSTEM, temperature=0),
    )
    return resp.text, hits

if __name__ == "__main__":
    q = " ".join(sys.argv[1:])
    text, hits = answer(q)
    print(text)
    print("\nSources:")
    for i, h in enumerate(hits):
        print(f" [{i+1}] {h['meta']['source']} p.{h['meta']['page']}")