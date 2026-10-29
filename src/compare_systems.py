"""Compare the baseline RAG pipeline with the LangGraph agent on an eval set.

  PYTHONPATH=src python src/compare_systems.py --file data/eval/questions_hard.json --limit 5
  PYTHONPATH=src python src/compare_systems.py --file data/eval/questions_heldout.json --ragas

Results are cached in data/eval/compare_cache.json, so re-running spends no Gemini quota
for questions already answered. If the daily quota runs out the script stops cleanly;
run it again tomorrow and it continues.
"""
import argparse
import datetime as dt
import json
import os
import time

from eval_metrics import score_row, summarize

CACHE = "data/eval/compare_cache.json"


class QuotaExhausted(Exception):
    pass


TRANSIENT = ("429", "RESOURCE_EXHAUSTED", "quota", "500", "502", "503", "504", "INTERNAL", "UNAVAILABLE",
             "DEADLINE_EXCEEDED", "overloaded", "high demand", "timed out", "timeout", "Connection")


def with_retry(fn, tries=5, wait=30):
    """Retry quota and temporary-overload errors with growing waits; anything else is a real bug."""
    for attempt in range(tries):
        try:
            return fn()
        except Exception as exc:
            msg = str(exc)
            if any(t in msg for t in TRANSIENT):
                if attempt == tries - 1:
                    raise QuotaExhausted(msg[:200])
                time.sleep(wait * (attempt + 1))
            else:
                raise


def load_cache():
    return json.load(open(CACHE)) if os.path.exists(CACHE) else {}


def save_cache(cache):
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    json.dump(cache, open(CACHE, "w"), indent=1)


def run_system(name, run_fn, items, cache, model, pause=0.0):
    """run_fn(question) -> (text, hits, llm_calls). Returns scored rows (cached when possible)."""
    rows = []
    for item in items:
        key = f"{name}|{model}|{item['q']}"
        if key not in cache:
            t0 = time.perf_counter()
            result = with_retry(lambda: run_fn(item["q"]))
            text, hits, calls = result[:3]
            cache[key] = {"text": text, "hits": hits, "llm_calls": calls, "trace": result[3] if len(result) > 3 else [],
                          "seconds": round(time.perf_counter() - t0, 2)}
            save_cache(cache)
            time.sleep(pause)
        c = cache[key]
        row = score_row(item, c["text"], c["hits"])
        row.update(llm_calls=c["llm_calls"], seconds=c["seconds"], text=c["text"], hits=c["hits"], trace=c.get("trace", []))
        rows.append(row)
    return rows


def ragas_scores(rows, model):
    """Faithfulness + context precision with Gemini as judge (needs requirements-eval.txt)."""
    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference
    from ragas.run_config import RunConfig

    judge = LangchainLLMWrapper(ChatOpenAI(
        model=model, api_key=os.environ["GEMINI_API_KEY"], temperature=0,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"))
    usable = [r for r in rows if r["answerable"] and not r["refused"]]
    data = EvaluationDataset(samples=[
        SingleTurnSample(user_input=r["q"], response=r["text"],
                         retrieved_contexts=[h["text"] for h in r["hits"]]) for r in usable])
    result = evaluate(data, metrics=[Faithfulness(), LLMContextPrecisionWithoutReference()], llm=judge,
                      run_config=RunConfig(max_workers=1, timeout=180, max_retries=4), show_progress=True)
    df = result.to_pandas()
    out = {}
    for col in df.columns:
        if "faithfulness" in col or "context_precision" in col:
            out[col] = round(float(df[col].mean(skipna=True)), 3)
    out["n_scored"] = len(usable)
    return out


def to_markdown(title, summaries, extra):
    cols = ["n_answerable", "n_unanswerable", "retrieval_hit", "key_in_passages", "citation_correct",
            "false_refusal", "correct_refusal", "avg_llm_calls", "avg_seconds"]
    lines = [f"### {title}", "", "| system | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, s in summaries.items():
        lines.append(f"| {name} | " + " | ".join(str(s[c]) for c in cols) + " |")
    if extra:
        lines += ["", "RAGAS (judge = Gemini): " + json.dumps(extra)]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=os.environ.get("EVAL_FILE", "data/eval/questions_hard.json"))
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions (0 = all)")
    ap.add_argument("--ragas", action="store_true")
    ap.add_argument("--show", action="store_true", help="print every question's outcome")
    ap.add_argument("--pause", type=float, default=4.0, help="seconds between LLM runs (rate limit)")
    args = ap.parse_args()

    from agent import run_agent
    from answer import answer, get_retriever
    from llm import generate

    model = os.environ.get("GEN_MODEL", "gemini-2.5-flash-lite")
    items = json.load(open(args.file))
    if args.limit:
        items = items[:args.limit]
    retriever = get_retriever()
    search = (lambda q, k: retriever.search(q, k=k)) if hasattr(retriever, "search") \
        else (lambda q, k: retriever.hybrid_search(q, k=k))

    def baseline(q):
        text, hits = answer(q)
        return text, hits, 1

    def agent(q):
        out = run_agent(q, search, generate)
        return out["answer"], out["hits"], out["llm_calls"], out["trace"]

    cache, rows, summaries = load_cache(), {}, {}
    try:
        for name, fn in (("baseline", baseline), ("agent_v2", agent)):
            rows[name] = run_system(name, fn, items, cache, model, args.pause)
            summaries[name] = summarize(rows[name])
    except QuotaExhausted as exc:
        save_cache(cache)
        print(f"\nGemini quota or availability problem ({exc}). Progress is saved; run the same command again later.")
        return

    if args.show:
        for name, rs in rows.items():
            print(f"\n== {name} ==")
            for r in rs:
                print(f"- {r['q'][:70]} | hit={r['retrieval_hit']} cited={r['citation_correct']} refused={r['refused']} calls={r['llm_calls']}")
                for step in r.get("trace", []):
                    print(f"    {step}")
    extra = {}
    if args.ragas:
        for name in rows:
            extra[name] = ragas_scores(rows[name], model)

    tag = os.path.splitext(os.path.basename(args.file))[0]
    title = f"{tag} | model {model} | {dt.date.today()}"
    md = to_markdown(title, summaries, extra)
    print("\n" + md)
    os.makedirs("results", exist_ok=True)
    open(f"results/compare_{tag}.md", "w").write(md + "\n")
    json.dump({"summaries": summaries, "ragas": extra, "model": model, "date": str(dt.date.today())},
              open(f"results/compare_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
