import base64
import sys
import json
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from cloudsearch import search as cs
from cloud import cloud_backend as be
import saved_answers as sa

ROOT = Path(__file__).parent
SAVED = sa.load(ROOT / "saved_answers.json")
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

GREETING = "Hi, I'm Aero, your guide. Let me walk you through this project and every tool behind it before you ask your first question."
LINES = [
    {"tag": "The problem", "emote": "Wave", "mood": None, "text": "Hi, I'm Aero. This is the Aviation Maintenance Copilot. A technician has thousands of pages of FAA advisory circulars. Finding the right paragraph is slow, and a wrong answer is expensive."},
    {"tag": "The goal", "emote": "ThumbsUp", "mood": None, "text": "The goal: ask a question in plain English and get a short answer with the exact source page, or an honest refusal when the documents do not say."},
    {"tag": "Data: pypdf", "emote": None, "mood": None, "text": "First, the data. The FAA documents are public PDFs. A Python library called pypdf extracts the text page by page, and I split it into overlapping chunks of about one thousand characters so no fact is cut in half."},
    {"tag": "Embeddings", "emote": "ThumbsUp", "mood": None, "text": "Each chunk is turned into numbers that capture its meaning, using the sentence-transformers model MiniLM. That lets the system find passages that mean the same thing even when the words differ."},
    {"tag": "Chroma", "emote": None, "mood": None, "text": "Those vectors are stored in Chroma, a local vector database, which keeps development free and fast."},
    {"tag": "BM25", "emote": None, "mood": None, "text": "Meaning search alone misses exact terms like part numbers or table names. So I also run BM25, a classic keyword search, and merge both rankings with reciprocal rank fusion. That is called hybrid retrieval."},
    {"tag": "Reranker", "emote": "ThumbsUp", "mood": None, "text": "I also tested a cross-encoder reranker that re-reads the top candidates. It improved ranking quality but added several seconds of delay, so I kept it optional. Measuring that trade-off was the point."},
    {"tag": "Azure AI Search", "emote": None, "mood": None, "text": "For the cloud, the same chunks are uploaded to Azure AI Search on the free tier. I checked that it returns the same results as my local search, so moving to the cloud did not lose quality."},
    {"tag": "Gemini", "emote": None, "mood": None, "text": "The answer is written by Google's Gemini model on its free tier. It may only use the retrieved passages, must cite them by number, and has to say so when the answer is not there."},
    {"tag": "Evaluation", "emote": None, "mood": "Surprised", "text": "I measured everything. I wrote easy, hard, and held-out question sets, and track Hit at 5, Hit at 10 and mean reciprocal rank. The held-out set was frozen first, so I could not tune on it."},
    {"tag": "Hosting", "emote": "ThumbsUp", "mood": None, "text": "The code lives on GitHub, was built in GitHub Codespaces, and this page runs on Streamlit Community Cloud. The total cost is zero euros."},
    {"tag": "Your turn", "emote": "Wave", "mood": None, "text": "That is the whole stack. Now it is your turn. Pick an example on the right, or ask your own maintenance question."},
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
                "How close to a pulley may a splice in a control cable be located?"]
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
                    hits = cs.hybrid(endpoint, skey, q, k=5)
            except Exception as e:
                hits = []
                st.error(f"Search failed: {type(e).__name__}")
            if hits:
                used_llm = False
                saved = sa.find(SAVED, q)
                if gkey and st.session_state.get("n", 0) < 5 and budget().take():
                    st.session_state["n"] = st.session_state.get("n", 0) + 1
                    try:
                        with st.spinner("Writing a cited answer..."):
                            st.markdown(be.generate(gkey, q, hits))
                        used_llm = True
                    except Exception:
                        if not saved:
                            st.info("The free language-model quota is busy. Showing the retrieved passages instead.")
                elif gkey and not saved:
                    st.info("Free-tier AI quota reached for now. Showing retrieved passages (the retrieval is the core of the system).")
                saved = None if used_llm else saved
                shown = hits
                if saved:
                    st.info(f"The free language-model quota is used up for now. This is a saved answer, generated on "
                            f"{saved['date']} with {saved['model']}. Type your own question to see live passages.")
                    st.markdown(saved["answer"])
                    shown = [{"source": d["source"], "page": d["page"], "content": d["content"]} for d in saved["sources"]]
                st.markdown("**Sources**" if (used_llm or saved) else "**Top passages**")
                for i, h in enumerate(shown, 1):
                    with st.expander(f"[{i}] {h['source']} · page {h['page']}",
                                     expanded=not used_llm and not saved and i == 1):
                        st.write(h["content"])
            elif endpoint:
                st.info("No matching passages found.")

    st.caption("Retrieval: hybrid search (keyword + MiniLM vectors, fused with RRF) on Azure AI Search. The reranker and the full evaluation are in the GitHub repo.")
