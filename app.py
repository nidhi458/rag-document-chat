"""
RAG Document Chat - Streamlit UI (Dark Theme)
"""

import streamlit as st

st.set_page_config(
    page_title="Doc Chat",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

import sys
import os

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from dotenv import load_dotenv
load_dotenv()

sys.path.append("utils")

from utils.vectorstore import VectorStore
from utils.embeddings import EmbeddingManager
from query import RAGRetriever
from langchain_groq import ChatGroq

# ── DARK THEME CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0f0f0f !important;
    color: #e8e8e8 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #161616 !important;
    border-right: 1px solid #2a2a2a !important;
}
[data-testid="stSidebar"] * {
    color: #e8e8e8 !important;
}

/* ── Main content area ── */
[data-testid="stMain"] {
    background-color: #0f0f0f !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    margin-bottom: 0.25rem;
}

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
    background-color: #1e1e1e !important;
    color: #e8e8e8 !important;
    border: 1px solid #333 !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] {
    background-color: #1e1e1e !important;
    border: 1px solid #333 !important;
    border-radius: 12px !important;
}

/* ── Buttons ── */
[data-testid="stButton"] button {
    background-color: #1e1e1e !important;
    color: #e8e8e8 !important;
    border: 1px solid #333 !important;
    border-radius: 8px !important;
}
[data-testid="stButton"] button:hover {
    background-color: #2a2a2a !important;
    border-color: #555 !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] * { color: #aaa !important; }

/* ── Source chips ── */
.source-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #1e1e1e;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 5px 10px;
    font-size: 12px;
    color: #aaa;
    margin: 4px 4px 0 0;
    line-height: 1.4;
}
.page-badge {
    background: #2a2a2a;
    border-radius: 4px;
    padding: 1px 6px;
    font-size: 11px;
    color: #666;
    margin-left: 4px;
}
.sources-label {
    font-size: 12px;
    color: #555;
    margin: 10px 0 4px 0;
}

/* ── Page header ── */
.page-header {
    padding: 1rem 0 0.75rem 0;
    border-bottom: 1px solid #2a2a2a;
    margin-bottom: 1rem;
}
.page-header h2 {
    font-size: 18px;
    font-weight: 500;
    color: #e8e8e8;
    margin: 0;
}
.page-header p {
    font-size: 13px;
    color: #555;
    margin: 3px 0 0 0;
}

/* ── Doc badge ── */
.doc-badge {
    background: #1a2a1a;
    color: #5a9e5a;
    border: 1px solid #2a3d2a;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 13px;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
for key in ("messages", "retriever", "vectorstore"):
    if key not in st.session_state:
        st.session_state[key] = [] if key == "messages" else None


# ── LLM ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_llm():
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=1024
    )


# ── PIPELINE LOADER ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_rag_pipeline():
    import time
    start = time.time()
    vectorstore = VectorStore(persist_directory="vector_store")
    try:
        count = vectorstore.collection.count()
    except Exception:
        count = 0
    embedding_manager = EmbeddingManager()
    retriever = RAGRetriever(vectorstore, embedding_manager)
    elapsed = time.time() - start
    return vectorstore, retriever, count, elapsed, embedding_manager


# ── INIT PIPELINE ─────────────────────────────────────────────────────────────
if st.session_state.retriever is None:
    with st.spinner("Loading RAG system..."):
        vectorstore, retriever, count, elapsed, embedding_manager = load_rag_pipeline()

    if count == 0:
        st.info("No documents found — running first-time ingestion...")
        from ingest import ingest_documents
        with st.spinner("Indexing documents for the first time..."):
            vectorstore, embedding_manager = ingest_documents()
            retriever = RAGRetriever(vectorstore, embedding_manager)
        st.success("Ingestion complete!")

    st.session_state.vectorstore = vectorstore
    st.session_state.retriever = retriever
    st.toast(f"Ready — {vectorstore.collection.count()} docs indexed ({elapsed:.2f}s)", icon="✅")

vectorstore = st.session_state.vectorstore
retriever = st.session_state.retriever
llm = get_llm()


# ── RAG CALL ─────────────────────────────────────────────────────────────────
def rag_simple(query, retriever, llm):
    results = retriever.retrieve(query, top_k=3)
    context = "\n\n".join([d["content"] for d in results]) if results else ""

    if not context:
        return "No relevant context found in your documents.", []

    prompt = f"""You are an expert assistant.

Answer using the provided context in 2-3 clear sentences.

Rules:
1. Give a direct answer.
2. Use your own words.
3. Do not say "according to the context" or "based on the provided context".

Context:
{context}

Question:
{query}

Answer:
"""
    response = llm.invoke(prompt)
    return response.content, results


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📄 Doc Chat")

    doc_count = 0
    try:
        doc_count = vectorstore.collection.count()
    except Exception:
        pass

    st.markdown(
        f'<div class="doc-badge">📚 {doc_count} documents indexed</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    if st.button("🔄 Rebuild Index", use_container_width=True):
        from ingest import ingest_documents
        with st.spinner("Rebuilding index..."):
            new_vs, new_em = ingest_documents()
        st.cache_resource.clear()
        st.session_state.vectorstore = new_vs
        st.session_state.retriever = RAGRetriever(new_vs, new_em)
        st.success("Index rebuilt!")
        st.rerun()

    if st.button("↺ Reset App", use_container_width=True):
        st.cache_resource.clear()
        st.session_state.clear()
        st.rerun()

    st.markdown(
        "<div style='margin-top:2rem;font-size:11px;color:#444;'>Powered by Groq + RAG</div>",
        unsafe_allow_html=True
    )


# ── MAIN UI ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <h2>Chat with your documents</h2>
  <p>Ask questions over your indexed PDFs</p>
</div>
""", unsafe_allow_html=True)


# ── CHAT HISTORY ─────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("sources"):
            chips_html = '<div class="sources-label">🔗 Sources</div>'
            for src in msg["sources"]:
                chips_html += (
                    f'<span class="source-chip">📄 {src["file"]}'
                    f'<span class="page-badge">p. {src["page"]}</span></span>'
                )
            st.markdown(chips_html, unsafe_allow_html=True)


# ── INPUT ─────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask anything about your documents…"):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if retriever is None:
            st.error("RAG pipeline not loaded.")
            st.stop()

        with st.spinner("Thinking…"):
            answer, results = rag_simple(prompt, retriever, llm)

        st.markdown(answer)

        sources_meta = []
        if results:
            chips_html = '<div class="sources-label">🔗 Sources</div>'
            for doc in results:
                meta = doc.get("metadata", {})
                file = meta.get("source_file", "Unknown")
                page = meta.get("page", "N/A")
                sources_meta.append({"file": file, "page": page})
                chips_html += (
                    f'<span class="source-chip">📄 {file}'
                    f'<span class="page-badge">p. {page}</span></span>'
                )
            st.markdown(chips_html, unsafe_allow_html=True)
        else:
            st.caption("No sources found.")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources_meta
        })