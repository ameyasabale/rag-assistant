"""
ui.py
─────
Streamlit frontend for the RAG Technical Documentation Assistant.

Run with:
    streamlit run ui.py
"""

import streamlit as st
import requests
import json
from datetime import datetime

API_BASE = "http://localhost:8000"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Docs Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main { background-color: #0f1117; }

    .stApp {
        background: linear-gradient(135deg, #0f1117 0%, #1a1d2e 100%);
    }

    /* Header */
    .hero-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.2rem;
        font-weight: 600;
        background: linear-gradient(90deg, #00d4ff, #7b61ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .hero-sub {
        color: #8b8fa8;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    /* Chat bubbles */
    .msg-user {
        background: linear-gradient(135deg, #1e2a4a, #1a2035);
        border-left: 3px solid #7b61ff;
        border-radius: 0 12px 12px 12px;
        padding: 14px 18px;
        margin: 8px 0 8px 40px;
        color: #e2e8f0;
        font-size: 0.95rem;
    }

    .msg-bot {
        background: linear-gradient(135deg, #0d1f1a, #0f2318);
        border-left: 3px solid #00d4ff;
        border-radius: 0 12px 12px 12px;
        padding: 14px 18px;
        margin: 8px 40px 8px 0;
        color: #e2e8f0;
        font-size: 0.95rem;
        line-height: 1.7;
    }

    /* Metadata badges */
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace;
        margin-right: 6px;
        margin-top: 8px;
        font-weight: 600;
    }

    .badge-type    { background: #1e1b4b; color: #818cf8; border: 1px solid #3730a3; }
    .badge-ground  { background: #052e16; color: #4ade80; border: 1px solid #166534; }
    .badge-halluc  { background: #450a0a; color: #f87171; border: 1px solid #991b1b; }
    .badge-retry   { background: #1c1917; color: #d6d3d1; border: 1px solid #57534e; }

    /* Source chips */
    .source-chip {
        display: inline-block;
        background: #1e293b;
        color: #94a3b8;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace;
        margin: 4px 4px 0 0;
        word-break: break-all;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #13161f !important;
        border-right: 1px solid #1e2130;
    }

    /* Input box */
    .stTextInput > div > div > input {
        background: #1a1d2e !important;
        border: 1px solid #2d3148 !important;
        color: #e2e8f0 !important;
        border-radius: 10px !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #7b61ff, #00d4ff) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.5rem !important;
        transition: opacity 0.2s !important;
    }

    .stButton > button:hover { opacity: 0.85 !important; }

    /* Divider */
    hr { border-color: #1e2130 !important; }

    /* Expander */
    .streamlit-expanderHeader {
        background: #1a1d2e !important;
        color: #94a3b8 !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []   # list of {role, content, meta}
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "last_question" not in st.session_state:
    st.session_state.last_question = None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 RAG Assistant")
    st.markdown("<hr>", unsafe_allow_html=True)

    # --- API health check ---
    try:
        r = requests.get(f"{API_BASE}/", timeout=3)
        if r.status_code == 200:
            st.success("✅ API connected", icon="🟢")
        else:
            st.error("⚠️ API error")
    except Exception:
        st.error("❌ API offline — run `python app.py` first")

    st.markdown("<hr>", unsafe_allow_html=True)

    # --- Indexed documents ---
    st.markdown("### 📚 Indexed Documents")
    if st.button("Refresh document list"):
        st.session_state["docs_refreshed"] = True

    try:
        docs_resp = requests.get(f"{API_BASE}/documents", timeout=5)
        if docs_resp.status_code == 200:
            docs_data = docs_resp.json()
            st.caption(f"{docs_data['total_documents']} document(s) in corpus")
            for doc in docs_data["documents"]:
                with st.expander(doc.get("title", doc["source"])[:40]):
                    st.code(doc["source"], language=None)
        else:
            st.warning("Could not load documents.")
    except Exception:
        st.caption("Run `python ingest.py` first to index docs.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # --- Ingest new URL ---
    st.markdown("### ➕ Ingest New Document")
    new_url = st.text_input("Paste a URL to ingest", placeholder="https://docs.example.com/page")
    if st.button("Ingest URL") and new_url.strip():
        with st.spinner("Ingesting…"):
            try:
                resp = requests.post(
                    f"{API_BASE}/ingest",
                    data={"urls": new_url.strip()},
                    timeout=30,
                )
                if resp.status_code == 200:
                    d = resp.json()
                    st.success(f"✅ {d['chunks_stored']} chunks added!")
                else:
                    st.error(f"Error: {resp.json().get('detail', 'Unknown')}")
            except Exception as e:
                st.error(f"Request failed: {e}")

    st.markdown("<hr>", unsafe_allow_html=True)

    # --- Clear chat ---
    if st.button("🗑️ Clear chat"):
        st.session_state.history = []
        st.session_state.last_answer = None
        st.session_state.last_question = None
        st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.caption("Built with LangGraph · FastAPI · ChromaDB · Groq")


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">📖 Technical Docs Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Self-corrective RAG · Hallucination checked · Powered by LangGraph</div>', unsafe_allow_html=True)

# Render chat history
for msg in st.session_state.history:
    if msg["role"] == "user":
        st.markdown(f'<div class="msg-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        meta = msg.get("meta", {})
        answer_html = msg["content"].replace("\n", "<br>")
        st.markdown(f'<div class="msg-bot">🤖 {answer_html}</div>', unsafe_allow_html=True)

        # Badges row
        qtype   = meta.get("query_type", "")
        hcheck  = meta.get("hallucination_check", "grounded")
        retries = meta.get("retries_used", 0)

        badges = ""
        if qtype:
            badges += f'<span class="badge badge-type">⚙ {qtype}</span>'
        if "hallucinated" in hcheck:
            badges += '<span class="badge badge-halluc">⚠ hallucination detected & retried</span>'
        else:
            badges += '<span class="badge badge-ground">✓ grounded</span>'
        if retries > 0:
            badges += f'<span class="badge badge-retry">↩ {retries} retr{"y" if retries==1 else "ies"}</span>'

        st.markdown(f'<div style="margin: 0 40px 4px 0">{badges}</div>', unsafe_allow_html=True)

        # Sources
        sources = meta.get("sources", [])
        if sources:
            chips = "".join(f'<span class="source-chip">🔗 {s}</span>' for s in sources)
            st.markdown(f'<div style="margin: 4px 40px 12px 0">{chips}</div>', unsafe_allow_html=True)

# ── Input row ─────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.text_input(
        label="Ask a question",
        placeholder="e.g. How do I handle HTTP errors in FastAPI?",
        label_visibility="collapsed",
        key="user_input",
    )

with col2:
    send = st.button("Ask →", use_container_width=True)

# ── Handle submit ─────────────────────────────────────────────────────────────
if send and user_input.strip():
    question = user_input.strip()
    st.session_state.history.append({"role": "user", "content": question})
    st.session_state.last_question = question

    with st.spinner("Thinking… (retrieving → grading → generating → checking)"):
        try:
            resp = requests.post(
                f"{API_BASE}/query",
                json={"question": question},
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                st.session_state.last_answer = data["answer"]
                st.session_state.history.append({
                    "role": "bot",
                    "content": data["answer"],
                    "meta": {
                        "query_type":          data.get("query_type"),
                        "hallucination_check": data.get("hallucination_check", "grounded"),
                        "retries_used":        data.get("retries_used", 0),
                        "sources":             data.get("sources", []),
                        "rewritten":           data.get("rewritten_query"),
                    },
                })
            else:
                err = resp.json().get("detail", "Unknown error")
                st.session_state.history.append({
                    "role": "bot",
                    "content": f"❌ API error: {err}",
                    "meta": {},
                })
        except Exception as e:
            st.session_state.history.append({
                "role": "bot",
                "content": f"❌ Could not reach the API: {e}",
                "meta": {},
            })

    st.rerun()

# ── Feedback row (only after a response) ─────────────────────────────────────
if st.session_state.last_answer and st.session_state.last_question:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("Was this answer helpful?")
    fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 6])

    with fb_col1:
        if st.button("👍 Yes"):
            requests.post(f"{API_BASE}/feedback", json={
                "question": st.session_state.last_question,
                "answer":   st.session_state.last_answer,
                "rating":   "thumbs_up",
            }, timeout=5)
            st.success("Thanks!")

    with fb_col2:
        if st.button("👎 No"):
            requests.post(f"{API_BASE}/feedback", json={
                "question": st.session_state.last_question,
                "answer":   st.session_state.last_answer,
                "rating":   "thumbs_down",
            }, timeout=5)
            st.info("Got it, we'll improve!")
