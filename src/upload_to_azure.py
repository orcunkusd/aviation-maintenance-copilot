import os
import re

import chromadb
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)

ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
CRED = AzureKeyCredential(os.environ["AZURE_SEARCH_KEY"])
INDEX = "faa-docs"
BATCH = 500

fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True),
    SearchableField(name="content", type=SearchFieldDataType.String),
    SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
    SimpleField(name="page", type=SearchFieldDataType.Int32, filterable=True),
    SearchField(
        name="embedding",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=384,
        vector_search_profile_name="vec-profile",
    ),
]
vector_search = VectorSearch(
    algorithms=[
        HnswAlgorithmConfiguration(name="hnsw", parameters=HnswParameters(metric="cosine"))
    ],
    profiles=[VectorSearchProfile(name="vec-profile", algorithm_configuration_name="hnsw")],
)

index_client = SearchIndexClient(ENDPOINT, CRED)
try:
    index_client.delete_index(INDEX)
except Exception:
    pass
index_client.create_index(SearchIndex(name=INDEX, fields=fields, vector_search=vector_search))
print(f"Created index '{INDEX}'")

col = chromadb.PersistentClient(path="data/chroma").get_collection("faa_docs")
data = col.get(include=["documents", "metadatas", "embeddings"])
n = len(data["ids"])
print(f"Read {n} chunks from Chroma")

docs = []
for cid, text, meta, emb in zip(
    data["ids"], data["documents"], data["metadatas"], data["embeddings"]
):
    docs.append(
        {
            "id": re.sub(r"[^A-Za-z0-9_=-]", "_", cid),  # Azure keys allow only these characters
            "content": text,
            "source": meta["source"],
            "page": int(meta["page"]),
            "embedding": [float(x) for x in emb],
        }
    )

search_client = SearchClient(ENDPOINT, INDEX, CRED)
ok = 0
for i in range(0, n, BATCH):
    results = search_client.upload_documents(documents=docs[i : i + BATCH])
    ok += sum(1 for r in results if r.succeeded)
    print(f"  uploaded {min(i + BATCH, n)}/{n}")
print(f"Done: {ok}/{n} documents indexed")