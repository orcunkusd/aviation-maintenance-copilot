import os

PROVIDER = os.environ.get("GEN_PROVIDER", "gemini")


def generate(system, user):
    if PROVIDER == "azure":
        from openai import AzureOpenAI

        client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version="2024-10-21",
        )
        r = client.chat.completions.create(
            model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=300,
        )
        return r.choices[0].message.content

    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options=types.HttpOptions(timeout=60000),
    )
    r = client.models.generate_content(
        model=os.environ.get("GEN_MODEL", "gemini-2.5-flash-lite"),
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system, temperature=0, max_output_tokens=400
        ),
    )
    return r.text