import os
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

print("Models available to your key:")
for m in client.models.list():
    print(" -", m.name)