"""
RAG Document Chat - Streamlit UI (Stable Fix Version)
"""

import streamlit as st
import sys
import os
from dotenv import load_dotenv

# 🔥 FIX: suppress torch + streamlit watcher crash (IMPORTANT for Cloud)
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

sys.path.append("utils")

from utils.vectorstore import VectorStore
from utils.embeddings import EmbeddingManager
from query import RAGRetriever
from langchain_groq import ChatGroq

# ── ENV ─────────────────────────
load_dotenv()

# ── PAGE CONFIG ─────────────────
st.set_page_config(
    page_title="RAG Doc Chat",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── SESSION STATE ───────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None


# ── LLM ─────────────────────────
@st.cache_resource
def get_llm():
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=1024
    )


# ── RAG PIPELINE ────────────────
@st.cache_resource(show_spinner=True)
def load_rag_pipeline():
    vectorstore = VectorStore()
    embedding_manager = EmbeddingManager()
    retriever = RAGRetriever(vectorstore, embedding_manager)
    return retriever


# ── SAFE RAG CALL ───────────────
def rag_simple(query, retriever, llm, top_k=3):

    results = retriever.retrieve(query, top_k=top_k)

    context = "\n\n".join([d["content"] for d in results]) if results else ""

    if not context:
        return "No relevant context found in your documents."

    prompt = f"""
You are an expert assistant.

Answer using the provided context in 2-3 clear sentences.

Context:
{context}

Question:
{query}

Answer:
"""

    with st.spinner("Generating answer..."):
        response = llm.invoke(prompt)

    return response.content


# ── INIT PIPELINE (FIXED - NO LOOP) ──
if st.session_state.retriever is None:
    st.session_state.retriever = load_rag_pipeline()

retriever = st.session_state.retriever
llm = get_llm()


# ── SIDEBAR ──────────────────────
with st.sidebar:
    st.title("📄 RAG Doc Chat")

    vs = getattr(retriever, "vectorstore", None)

    doc_count = 0
    try:
        if vs is not None and hasattr(vs, "collection") and vs.collection is not None:
            doc_count = vs.collection.count()
    except:
        doc_count = 0

    st.info(f"**{doc_count}** documents indexed and ready")

    st.markdown("---")

    if st.button("↺ Reset App"):
        st.cache_resource.clear()
        st.session_state.clear()
        st.rerun()

    st.caption("Built with RAG + Groq")

# ── MAIN UI ──────────────────────
st.title("Chat with your Documents")
st.markdown("Ask questions over your indexed documents")


# ── CHAT HISTORY ────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ── INPUT ────────────────────────
if prompt := st.chat_input("Ask anything..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        if retriever is None:
            st.error("RAG pipeline not loaded")
            st.stop()

        answer = rag_simple(prompt, retriever, llm, top_k=3)
        st.markdown(answer)

        # ── SOURCES ──
        st.markdown("**Sources:**")
        results = retriever.retrieve(prompt, top_k=3)

        for i, doc in enumerate(results, 1):
            meta = doc.get("metadata", {})
            source = meta.get("source_file", "Unknown")
            page = meta.get("page", "N/A")

            with st.expander(f"{i}. {source} · page {page}"):
                st.write(doc["content"][:500] + "...")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )


# ── FOOTER ──────────────────────
st.markdown("---")
st.markdown("*Powered by Groq + RAG Pipeline*")