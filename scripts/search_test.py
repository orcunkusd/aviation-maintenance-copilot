import sys
import chromadb
from sentence_transformers import SentenceTransformer

q = " ".join(sys.argv[1:])
model = SentenceTransformer("all-MiniLM-L6-v2")
col = chromadb.PersistentClient(path="data/chroma").get_collection("faa_docs")
res = col.query(query_embeddings=model.encode([q]).tolist(), n_results=3)
for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
    print(f"\n[{meta['source']} p.{meta['page']}] distance={dist:.3f}\n{doc[:400]}")