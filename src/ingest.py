import glob, os
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

CHUNK_CHARS = 1000
OVERLAP = 150
BATCH = 2000


def chunk_text(text):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_CHARS])
        start += CHUNK_CHARS - OVERLAP
    return chunks


model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="data/chroma")
try:
    client.delete_collection("faa_docs")
except Exception:
    pass
col = client.create_collection("faa_docs", metadata={"hnsw:space": "cosine"})

ids, docs, metas = [], [], []
for path in sorted(glob.glob("data/raw/*.pdf")):
    name = os.path.basename(path)
    reader = PdfReader(path)
    before = len(docs)
    for page_no, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        for i, chunk in enumerate(chunk_text(text)):
            if len(chunk.strip()) < 50:
                continue
            ids.append(f"{name}-p{page_no}-c{i}")
            docs.append(chunk)
            metas.append({"source": name, "page": page_no})
    print(f"Read {name}: {len(reader.pages)} pages, {len(docs) - before} chunks")

print(f"Embedding {len(docs)} chunks...")
for i in range(0, len(docs), BATCH):
    batch_docs = docs[i:i + BATCH]
    emb = model.encode(batch_docs, batch_size=64, show_progress_bar=True).tolist()
    col.add(
        ids=ids[i:i + BATCH],
        documents=batch_docs,
        metadatas=metas[i:i + BATCH],
        embeddings=emb,
    )
print(f"Stored {len(docs)} chunks in Chroma.")