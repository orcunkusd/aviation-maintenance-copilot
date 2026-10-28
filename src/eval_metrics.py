"""Answer-level metrics that need no LLM: citations, refusals, summaries."""
import re  # noqa: F401

from citations import cited_numbers  # noqa: F401  (re-exported)




def is_refusal(text):
    return "could not find" in (text or "").lower()


def expected_pages(item):
    if item.get("pages"):
        return list(item["pages"])
    return [item["page"]] if item.get("page") else []


def _match(hit, item):
    src_ok = (not item.get("source")) or hit["meta"]["source"] == item["source"]
    return src_ok and hit["meta"]["page"] in expected_pages(item)


def retrieval_hit(hits, item):
    """Is any retrieved passage from an expected page?"""
    return any(_match(h, item) for h in hits)


def citation_correct(text, hits, item):
    """Does the answer cite at least one passage from an expected page?"""
    return any(1 <= n <= len(hits) and _match(hits[n - 1], item) for n in cited_numbers(text))


def score_row(item, text, hits):
    answerable = bool(item.get("answerable", True))
    refused = is_refusal(text)
    return {
        "q": item["q"],
        "answerable": answerable,
        "refused": refused,
        "retrieval_hit": retrieval_hit(hits, item) if answerable else None,
        "citation_correct": (citation_correct(text, hits, item) and not refused) if answerable else None,
    }


def _rate(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None


def summarize(rows):
    ans = [r for r in rows if r["answerable"]]
    un = [r for r in rows if not r["answerable"]]
    return {
        "n_answerable": len(ans),
        "n_unanswerable": len(un),
        "retrieval_hit": _rate([r["retrieval_hit"] for r in ans]),
        "citation_correct": _rate([r["citation_correct"] for r in ans]),
        "false_refusal": _rate([r["refused"] for r in ans]),
        "correct_refusal": _rate([r["refused"] for r in un]),
        "avg_llm_calls": _rate([r["llm_calls"] for r in rows]),
        "avg_seconds": _rate([r["seconds"] for r in rows]),
    }
