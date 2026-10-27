import os
from google import genai
from google.genai import types

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options=types.HttpOptions(timeout=30000),
)

for name in ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]:
    try:
        r = client.models.generate_content(model=name, contents="Say OK.")
        print(f"[{name}] WORKS: {r.text.strip()[:30]}")
    except Exception as e:
        print(f"\n[{name}] FAILED:\n{str(e)[:1200]}")