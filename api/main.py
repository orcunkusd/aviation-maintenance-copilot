import json
import logging
import os
import secrets
import time
from typing import List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from api import backend

log = logging.getLogger("copilot")
logging.basicConfig(level=logging.INFO, format="%(message)s")

app = FastAPI(
    title="Aviation Maintenance Copilot API",
    version="1.0.0",
    description="Cited answers from FAA maintenance documents. Refuses when the documents do not contain the answer.",
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500, examples=["What is the minimum rivet spacing?"])
    k: int = Field(5, ge=1, le=10, description="Number of passages to retrieve")


class Source(BaseModel):
    n: int
    source: str
    page: int
    snippet: str


class AskResponse(BaseModel):
    answer: str
    refused: bool = Field(description="True when the documents did not contain an answer")
    llm_used: bool = Field(description="False when the daily LLM budget is spent or the LLM failed")
    sources: List[Source]
    latency_ms: int


def require_key(x_api_key: Optional[str] = Header(default=None)):
    """If API_KEY is set, callers must send it in the X-API-Key header."""
    expected = os.environ.get("API_KEY")
    if expected and not secrets.compare_digest(x_api_key or "", expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_key)])
def ask(req: AskRequest):
    t0 = time.perf_counter()
    try:
        hits = backend.search(req.question, k=req.k)
    except Exception as exc:  # search backend down or misconfigured
        log.error(json.dumps({"event": "search_error", "error": type(exc).__name__}))
        raise HTTPException(status_code=502, detail="Search backend unavailable")

    sources = [Source(n=i, source=h["source"], page=h["page"], snippet=h["content"][:300])
               for i, h in enumerate(hits, 1)]

    if not hits:
        text, refused, llm_used = "No matching passages found in the documents.", True, False
    elif backend.budget.take():
        try:
            text = backend.generate(req.question, hits)
            refused = "could not find" in text.lower()
            llm_used = True
        except Exception as exc:
            log.error(json.dumps({"event": "llm_error", "error": type(exc).__name__}))
            text, refused, llm_used = "The language model is unavailable; see the retrieved passages.", False, False
    else:
        text, refused, llm_used = "Daily AI quota reached; see the retrieved passages.", False, False

    ms = int((time.perf_counter() - t0) * 1000)
    log.info(json.dumps({"event": "ask", "ms": ms, "hits": len(hits), "refused": refused, "llm_used": llm_used}))
    return AskResponse(answer=text, refused=refused, llm_used=llm_used, sources=sources, latency_ms=ms)
