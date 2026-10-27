import re
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

def tokenize(text):
    text = text.lower().replace("–", "-")  # AD numbers use en-dashes
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text)

class HybridRetriever:
    def __init__(self, path="data/chroma", collection="faa_docs"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.col = chromadb.PersistentClient(path=path).get_collection(collection)
        data = self.col.get(include=["documents", "metadatas"])
        self.ids = data["ids"]
        self.docs = data["documents"]
        self.metas = data["metadatas"]
        self.bm25 = BM25Okapi([tokenize(d) for d in self.docs])

    def vector_search(self, query, k=5):
        res = self.col.query(
            query_embeddings=self.model.encode([query]).tolist(), n_results=k
        )
        return res["ids"][0]

    def keyword_search(self, query, k=5):
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.ids[i] for i in order]

    def hybrid_search(self, query, k=5, pool=20, rrf_k=60):
        fused = {}
        for ranking in (self.vector_search(query, pool), self.keyword_search(query, pool)):
            for rank, cid in enumerate(ranking):
                fused[cid] = fused.get(cid, 0) + 1 / (rrf_k + rank + 1)
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        index = {cid: i for i, cid in enumerate(self.ids)}
        return [
            {"id": cid, "text": self.docs[index[cid]], "meta": self.metas[index[cid]]}
            for cid in top
        ]