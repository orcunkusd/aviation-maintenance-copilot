import json, os, time
from answer import answer

CACHE = "data/eval/cache.json"
PAUSE = 13  # seconds between calls, about 4-5 requests a minute


def norm(s):
    return " ".join(s.replace("–", "-").replace("‑", "-").lower().split())


rows = json.load(open(os.environ.get("EVAL_FILE", "data/eval/questions.json")))
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}


def get(q):
    if q in cache:
        return cache[q]["text"], cache[q]["hits"]
    for attempt in range(3):
        try:
            text, hits = answer(q)
            cache[q] = {"text": text, "hits": hits}
            json.dump(cache, open(CACHE, "w"))
            time.sleep(PAUSE)
            return text, hits
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                print("  rate limited, waiting 65 s...")
                time.sleep(65)
            else:
                print("  ERROR:", msg[:150])
                return None, None
    return None, None


retrieval_hits = answer_hits = refusals = n_ans = n_unans = skipped = 0

for r in rows:
    text, hits = get(r["q"])
    if text is None:
        skipped += 1
        print(f"SKIP (no answer received) | {r['q'][:70]}")
        continue
    t = norm(text)
    refused = "could not find" in t
    if r["answerable"]:
        n_ans += 1
        key = norm(r["key"])
        in_ctx = any(
            key in norm(h["text"])
            and (
                not r.get("page")
                or (h["meta"]["page"] == r["page"] and h["meta"]["source"] == r["source"])
            )
            for h in hits
        )
        in_ans = key in t
        retrieval_hits += in_ctx
        answer_hits += in_ans
        print(f"{'OK ' if in_ans else 'BAD'} retrieved={in_ctx} | {r['q'][:70]}")
    else:
        n_unans += 1
        refusals += refused
        print(f"{'OK ' if refused else 'BAD'} refused={refused} | {r['q'][:70]}")

print(f"\nScored: {n_ans + n_unans} questions, skipped: {skipped}")
print(f"Retrieval hit rate: {retrieval_hits}/{n_ans}")
print(f"Answer contains key: {answer_hits}/{n_ans}")
print(f"Correct refusals: {refusals}/{n_unans}")