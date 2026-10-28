"""Fixed-size character chunking with overlap."""
CHUNK_CHARS = 1000
OVERLAP = 150


def chunk_text(text, size=CHUNK_CHARS, overlap=OVERLAP):
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks
