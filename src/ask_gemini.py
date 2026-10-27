import os, time
from google import genai
from google.genai import types

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options=types.HttpOptions(timeout=30000),  # give up after 30 s
)

models = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-flash-latest"]

for name in models:
    start = time.time()
    try:
        r = client.models.generate_content(
            model=name,
            contents="In two sentences, what is an Airworthiness Directive in aviation?",
        )
        print(f"\n[{name}] {time.time() - start:.1f}s\n{r.text}")
    except Exception as e:
        print(f"\n[{name}] FAILED after {time.time() - start:.1f}s: {str(e)[:300]}")