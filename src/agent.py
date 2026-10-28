"""Agentic RAG with LangGraph.

plan      the original question + up to 2 LLM-written search queries
retrieve  hybrid search for each query, merge and de-duplicate
grade     are the passages enough to answer? if not, propose a better query (LLM)
          -> one retry at most
answer    cited answer from the passages only (LLM), same prompt as the baseline
verify    check that every [n] citation points at a real passage (no LLM)

Budget: 3 LLM calls per question, 4 if the retry fires. `llm_calls` is reported.
Retriever and LLM are injected, so tests run without models, keys or internet.
"""
import json
import re
import sys
from typing import List, TypedDict

from langgraph.graph import END, StateGraph

from answer import SYSTEM, build_context
from citations import cited_numbers

MAX_RETRIES = 1
MAX_PASSAGES = 6

PLAN_SYSTEM = (
    "You help search aviation maintenance manuals. Turn the user's question into 1 to 3 short, "
    "self-contained search queries (use 1 unless the question has clearly separate parts). "
    'Reply with JSON only: {"queries": ["..."]}'
)
GRADE_SYSTEM = (
    "You judge search results for an aviation maintenance question. Decide if the passages contain "
    "enough information to answer it. If not, suggest one better search query using different keywords. "
    'Reply with JSON only: {"sufficient": true or false, "better_query": "..."}'
)


class State(TypedDict, total=False):
    question: str
    queries: List[str]
    hits: List[dict]
    retries: int
    sufficient: bool
    needs_retry: bool
    answer: str
    citations_ok: bool
    llm_calls: int
    trace: List[str]


def parse_json(text, default):
    """Pull the first JSON object out of an LLM reply (tolerates code fences and chatter)."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return default
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return default


def merge_hits(per_query_hits, limit=MAX_PASSAGES):
    """Round-robin over the result lists, drop duplicates, keep at most `limit` passages."""
    merged, seen = [], set()
    for rank in range(max((len(h) for h in per_query_hits), default=0)):
        for hits in per_query_hits:
            if rank < len(hits) and hits[rank]["id"] not in seen:
                seen.add(hits[rank]["id"])
                merged.append(hits[rank])
    return merged[:limit]


def check_citations(text, n_passages):
    """True if the answer cites at least one passage and every [n] is a real passage number."""
    nums = cited_numbers(text)
    if not nums:
        return "could not find" in (text or "").lower()   # a refusal needs no citation
    return all(1 <= n <= n_passages for n in nums)


def build_graph(retrieve, llm, k=5):
    """retrieve(query, k) -> list of hit dicts; llm(system, user) -> str."""

    def _call(state, system, user):
        state["llm_calls"] = state.get("llm_calls", 0) + 1
        return llm(system, user)

    def plan(state):
        raw = _call(state, PLAN_SYSTEM, state["question"])
        planned = [q.strip() for q in parse_json(raw, {}).get("queries", []) if isinstance(q, str) and q.strip()]
        # always keep the user's own wording: LLM rewrites can drop the exact words keyword search needs
        queries = [state["question"]] + [q for q in planned if q != state["question"]]
        queries = queries[:3]
        return {"queries": queries, "llm_calls": state["llm_calls"],
                "trace": state.get("trace", []) + [f"plan: {queries}"]}

    def do_retrieve(state):
        per_query = [retrieve(q, k) for q in state["queries"]]
        hits = merge_hits(per_query)
        return {"hits": hits, "trace": state["trace"] + [f"retrieve: {len(hits)} passages"]}

    def grade(state):
        brief = "\n".join(f"[{i}] {h['text'][:300]}" for i, h in enumerate(state["hits"], 1))
        raw = _call(state, GRADE_SYSTEM, f"Question: {state['question']}\n\nPassages:\n{brief}")
        verdict = parse_json(raw, {"sufficient": True})
        ok = bool(verdict.get("sufficient", True))
        better = verdict.get("better_query")
        has_better = isinstance(better, str) and better.strip() and better.strip() not in state["queries"]
        needs_retry = (not ok) and bool(has_better) and state.get("retries", 0) < MAX_RETRIES
        update = {"sufficient": ok, "needs_retry": needs_retry, "llm_calls": state["llm_calls"],
                  "trace": state["trace"] + [f"grade: sufficient={ok}"]}
        if needs_retry:
            update["queries"] = [better.strip()]
        return update

    def route_after_grade(state):
        return "retry" if state["needs_retry"] else "answer"

    def bump_retry(state):
        return {"retries": state.get("retries", 0) + 1,
                "trace": state["trace"] + [f"retry with: {state['queries']}"]}

    def answer_node(state):
        text = _call(state, SYSTEM,
                     f"Context passages:\n{build_context(state['hits'])}\n\nQuestion: {state['question']}")
        return {"answer": text, "llm_calls": state["llm_calls"],
                "trace": state["trace"] + ["answer: written"]}

    def verify(state):
        ok = check_citations(state["answer"], len(state["hits"]))
        return {"citations_ok": ok, "trace": state["trace"] + [f"verify: citations_ok={ok}"]}

    g = StateGraph(State)
    g.add_node("plan", plan)
    g.add_node("retrieve", do_retrieve)
    g.add_node("grade", grade)
    g.add_node("bump_retry", bump_retry)
    g.add_node("answer", answer_node)
    g.add_node("verify", verify)
    g.set_entry_point("plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route_after_grade, {"retry": "bump_retry", "answer": "answer"})
    g.add_edge("bump_retry", "retrieve")
    g.add_edge("answer", "verify")
    g.add_edge("verify", END)
    return g.compile()


def run_agent(question, retrieve, llm, k=5):
    out = build_graph(retrieve, llm, k).invoke({"question": question, "llm_calls": 0, "trace": [], "retries": 0})
    return out


if __name__ == "__main__":
    from answer import get_retriever
    from llm import generate

    retriever = get_retriever()
    search = (lambda q, k: retriever.search(q, k=k)) if hasattr(retriever, "search") \
        else (lambda q, k: retriever.hybrid_search(q, k=k))
    result = run_agent(" ".join(sys.argv[1:]), search, generate)
    print(result["answer"])
    print("\nSources:")
    for i, h in enumerate(result["hits"], 1):
        print(f" [{i}] {h['meta']['source']} p.{h['meta']['page']}")
    print(f"\nTrace: {' -> '.join(result['trace'])}")
    print(f"LLM calls: {result['llm_calls']} | citations ok: {result['citations_ok']}")
