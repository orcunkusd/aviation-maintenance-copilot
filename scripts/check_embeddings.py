"""Does the lightweight (ONNX) embedder reproduce the sentence-transformers vectors used to build the index?
Run:  PYTHONPATH=.:src python scripts/check_embeddings.py   (cosine should be ~1.0)
"""
import numpy as np
from sentence_transformers import SentenceTransformer

from cloudsearch.search import embed_query

texts = ["What is the minimum rivet spacing?", "How is corrosion removed from aluminum?",
         "AD 2020-06-15 applies to Cessna 172 propellers", "torque values for AN bolts"]
st = SentenceTransformer("all-MiniLM-L6-v2")
for t, ref in zip(texts, st.encode(texts)):
    v = np.array(embed_query(t))
    cos = float(v @ ref / (np.linalg.norm(v) * np.linalg.norm(ref)))
    print(f"cosine={cos:.5f}  {t}")
