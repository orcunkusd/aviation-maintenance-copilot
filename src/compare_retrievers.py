import json
import os

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from retriever import HybridRetriever

FIELDS = ["content", "source", "page"]


def norm(s):
    return " ".join(s.replace("–", "-").replace("‑", "-").lower().split())


local = HybridRetriever()
from rerank_retriever import RerankRetriever

rr = RerankRetriever(local)
az = SearchClient(
    os.environ["AZURE_SEARCH_ENDPOINT"],
    "faa-docs",
    AzureKeyCredential(os.environ["AZURE_SEARCH_KEY"]),
)
index = {cid: i for i, cid in enumerate(local.ids)}


def from_ids(ids):
    return [
        (local.docs[index[c]], local.metas[index[c]]["source"], local.metas[index[c]]["page"])
        for c in ids
    ]


def az_rows(results):
    return [(r["content"], r["source"], r["page"]) for r in results]


def vec_query(q):
    emb = local.model.encode([q]).tolist()[0]
    return VectorizedQuery(vector=emb, k_nearest_neighbors=10, fields="embedding")


METHODS = {
    "Local keyword (BM25)": lambda q: from_ids(local.keyword_search(q, 10)),
    "Local vector (MiniLM)": lambda q: from_ids(local.vector_search(q, 10)),
    "Local hybrid (RRF)": lambda q: from_ids([h["id"] for h in local.hybrid_search(q, k=10)]),
    "Local hybrid + rerank": lambda q: [
        (h["text"], h["meta"]["source"], h["meta"]["page"]) for h in rr.search(q, k=10)
    ],
    "Azure keyword": lambda q: az_rows(az.search(search_text=q, top=10, select=FIELDS)),
    "Azure vector": lambda q: az_rows(
        az.search(search_text=None, vector_queries=[vec_query(q)], top=10, select=FIELDS)
    ),
    "Azure hybrid": lambda q: az_rows(
        az.search(search_text=q, vector_queries=[vec_query(q)], top=10, select=FIELDS)
    ),
}

rows = [r for r in json.load(open(os.environ.get("EVAL_FILE", "data/eval/questions.json"))) if r["answerable"]]


def rank_of(results, row):
    key = norm(row["key"])
    for i, (text, source, page) in enumerate(results, start=1):
        pages = row.get("pages") or ([row["page"]] if row.get("page") else [])
        right_place = not pages or (page in pages and source == row["source"])
        if key in norm(text) and right_place:
            return i
    return None


print(f"{'Method':<24} {'Hit@5':>6} {'Hit@10':>7} {'MRR':>6}")
for name, fn in METHODS.items():
    ranks = [rank_of(fn(r["q"]), r) for r in rows]
    h5 = sum(1 for x in ranks if x and x <= 5)
    h10 = sum(1 for x in ranks if x)
    mrr = sum(1 / x for x in ranks if x) / len(rows)
    print(f"{name:<24} {h5:>3}/{len(rows)} {h10:>4}/{len(rows)} {mrr:>6.2f}")