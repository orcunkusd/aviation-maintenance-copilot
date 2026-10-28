from sentence_transformers import CrossEncoder

from retriever import HybridRetriever


class RerankRetriever:
    def __init__(self, base=None, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.base = base or HybridRetriever()
        self.model = CrossEncoder(model_name)

    def search(self, query, k=5, pool=50, candidates=40):
        cands = self.base.hybrid_search(query, k=candidates, pool=pool)
        scores = self.model.predict([(query, c["text"]) for c in cands])
        ranked = sorted(zip(scores, cands), key=lambda x: x[0], reverse=True)
        return [c for _, c in ranked[:k]]