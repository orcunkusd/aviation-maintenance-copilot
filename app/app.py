import base64
import json
from pathlib import Path

import streamlit as st

from src import cloud_backend as be

ROOT = Path(__file__).parent
st.set_page_config(page_title="Aviation Maintenance Copilot", page_icon="✈️", layout="wide",
                   initial_sidebar_state="collapsed")

# ---- layout tuning: tight padding, hidden chrome, dark theme feel -------------------------
st.markdown("""
<style>
#MainMenu, footer, header [data-testid="stToolbar"], [data-testid="stDecoration"] {visibility:hidden; height:0}
.block-container {max-width:1180px; padding:1.2rem 1.2rem 2rem}
h1 {font-size:clamp(1.6rem,4.5vw,2.5rem)!important; letter-spacing:-.02em; margin-bottom:.1rem}
.sub {color:#8b98b8; margin-bottom:1rem; font-size:1.02rem}
.chip {display:inline-block; padding:.18rem .6rem; margin:0 .3rem .3rem 0; border-radius:999px;
       background:rgba(90,209,255,.12); border:1px solid rgba(90,209,255,.35); font-size:.78rem}
.src {font-size:.82rem; color:#8b98b8}
@media (max-width:640px){ .block-container{padding:.6rem .6rem 1.5rem} }
</style>""", unsafe_allow_html=True)

GREETING = "Hi, I'm Aero. Move your mouse around, I'll keep an eye on you. Press the button and I'll walk you through this project."
LINES = [
    {"emote": "Wave", "mood": None, "text": "I'm the guide to the Aviation Maintenance Copilot: a question answering system for FAA maintenance documents."},
    {"emote": "ThumbsUp", "mood": None, "text": "The problem: a technician has thousands of pages of advisory circulars and airworthiness directives. Finding the right paragraph is slow, and a wrong answer is expensive."},
    {"emote": "ThumbsUp", "mood": None, "text": "The system reads the PDFs, splits them into chunks, and indexes them twice: by meaning with embeddings, and by exact words with BM25. Results are merged with reciprocal rank fusion."},
    {"emote": None, "mood": "Surprised", "text": "Every answer cites its source page. If the documents don't contain the answer, it refuses instead of guessing. That is the safety-critical part."},
    {"emote": "ThumbsUp", "mood": None, "text": "I measured it, not just built it. On a held-out question set, I report Hit at 5 and mean reciprocal rank, and I tested a cross-encoder reranker and found where it helps and where it only adds latency."},
    {"emote": "ThumbsUp", "mood": None, "text": "Cloud side: Azure AI Search holds the index on the free tier, the language model is Gemini's free tier, and this page runs on Streamlit Community Cloud. Total cost: zero."},
    {"emote": "Wave", "mood": None, "text": "Now try it. Ask a maintenance question on the right, or pick an example."},
]


@st.cache_data
def mascot_html() -> str:
    html = (ROOT / "webapp" / "mascot.html").read_text(encoding="utf-8")
    model = base64.b64encode((ROOT / "webapp" / "robot.glb").read_bytes()).decode()
    return (html.replace("__LINES__", json.dumps(LINES)).replace("__GREETING__", json.dumps(GREETING))
                .replace("__MODEL__", model))


@st.cache_resource
def budget() -> be.DailyBudget:
    return be.DailyBudget(int(st.secrets.get("DAILY_LLM_LIMIT", 15)))


def secret(name):
    try:
        return st.secrets.get(name)
    except Exception:
        return None


st.markdown("# ✈️ Aviation Maintenance Copilot")
st.markdown('<div class="sub">Cited answers from FAA advisory circulars. Hybrid retrieval, evaluated, refuses when unsure.</div>'
            '<span class="chip">Azure AI Search</span><span class="chip">Hybrid BM25 + vectors</span>'
            '<span class="chip">Gemini</span><span class="chip">Streamlit</span>', unsafe_allow_html=True)

left, right = st.columns([5, 6], gap="large")

with left:
    st.iframe(mascot_html(), height=600, width="stretch")

with right:
    st.subheader("Ask the manuals")
    examples = ["What is the minimum rivet spacing?", "How is corrosion on aluminum alloy removed?",
                "What are the torque values for AN bolts?"]
    cols = st.columns(len(examples))
    for c, ex in zip(cols, examples):
        if c.button(ex, use_container_width=True):
            st.session_state["q"] = ex
    q = st.text_input("Your question", key="q", placeholder="e.g. How do I inspect a wooden propeller?")

    endpoint, skey, gkey = secret("AZURE_SEARCH_ENDPOINT"), secret("AZURE_SEARCH_KEY"), secret("GEMINI_API_KEY")
    if q:
        if not (endpoint and skey):
            st.warning("Demo backend is not configured (missing Azure Search secrets).")
        else:
            try:
                with st.spinner("Searching the documents..."):
                    hits = be.search(endpoint, skey, q, k=5)
            except Exception as e:
                hits = []
                st.error(f"Search failed: {type(e).__name__}")
            if hits:
                used_llm = False
                if gkey and st.session_state.get("n", 0) < 5 and budget().take():
                    st.session_state["n"] = st.session_state.get("n", 0) + 1
                    try:
                        with st.spinner("Writing a cited answer..."):
                            st.markdown(be.generate(gkey, q, hits))
                        used_llm = True
                    except Exception:
                        st.info("The free language-model quota is busy. Showing the retrieved passages instead.")
                elif gkey:
                    st.info("Free-tier AI quota reached for now. Showing retrieved passages (the retrieval is the core of the system).")
                st.markdown("**Sources**" if used_llm else "**Top passages**")
                for i, h in enumerate(hits, 1):
                    with st.expander(f"[{i}] {h['source']} · page {h['page']}", expanded=not used_llm and i == 1):
                        st.write(h["content"])
            elif endpoint:
                st.info("No matching passages found.")

    st.caption("Retrieval here is keyword-based on Azure's free tier; the full hybrid + reranker pipeline and its evaluation are in the GitHub repo.")
