import json, time
from answer import answer

def norm(s):
    return " ".join(s.replace("–", "-").replace("‑", "-").lower().split())

rows = json.load(open("data/eval/questions.json"))
retrieval_hits = answer_hits = refusals = n_ans = n_unans = 0

for r in rows:
    try:
        text, hits = answer(r["q"])
    except Exception as e:
        print("ERROR:", str(e)[:150]); time.sleep(20); continue
    t = norm(text)
    refused = "could not find" in t
    if r["answerable"]:
        n_ans += 1
        key = norm(r["key"])
        in_ctx = any(key in norm(h["text"]) for h in hits)
        in_ans = key in t
        retrieval_hits += in_ctx
        answer_hits += in_ans
        print(f"{'OK ' if in_ans else 'BAD'} retrieved={in_ctx} | {r['q'][:70]}")
    else:
        n_unans += 1
        refusals += refused
        print(f"{'OK ' if refused else 'BAD'} refused={refused} | {r['q'][:70]}")
    time.sleep(5)  # stay under free-tier rate limits

print(f"\nRetrieval hit rate: {retrieval_hits}/{n_ans}")
print(f"Answer contains key: {answer_hits}/{n_ans}")
print(f"Correct refusals: {refusals}/{n_unans}")