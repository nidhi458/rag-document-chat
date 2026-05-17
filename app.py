"""
RAG Document Chat - Streamlit UI
"""
import chromadb
chromadb.config.Settings(anonymized_telemetry=False)

import streamlit as st
import sys
import os

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv

load_dotenv()

sys.path.append("utils")

from utils.vectorstore import VectorStore
from utils.embeddings import EmbeddingManager
from query import RAGRetriever
from langchain_groq import ChatGroq

# ── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Doc Chat",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── SESSION STATE ────────────────────────────────────────────────────────────
# Initialise all keys together — avoids the double-check bug where the second
# `if "retriever" not in st.session_state` block was silently skipped because
# the first block had already added the key (as None).
for key in ("messages", "retriever", "vectorstore"):
    if key not in st.session_state:
        st.session_state[key] = [] if key == "messages" else None


# ── LLM ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_llm():
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=1024
    )


# ── PIPELINE LOADER ──────────────────────────────────────────────────────────
# st.write() calls removed from inside the cached function — Streamlit does not
# guarantee UI calls work reliably inside @st.cache_resource.
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
    return vectorstore, retriever, count, elapsed


# ── INIT PIPELINE (runs once per session) ───────────────────────────────────
if st.session_state.retriever is None:
    with st.spinner("Loading RAG system..."):
        vectorstore, retriever, count, elapsed = load_rag_pipeline()

    if count == 0:
        # Fresh deploy (e.g. Streamlit Cloud) — no persisted vector store found.
        # Auto-ingest PDFs from data/documents/ on first run.
        st.info("No documents indexed yet — running first-time ingestion...")
        from ingest import ingest_documents
        with st.spinner("Indexing documents for the first time..."):
            vectorstore, embedding_manager = ingest_documents()
            retriever = RAGRetriever(vectorstore, embedding_manager)
        st.success("Ingestion complete!")

    st.session_state.vectorstore = vectorstore
    st.session_state.retriever = retriever
    st.toast(f"Pipeline ready — {vectorstore.collection.count()} docs indexed ({elapsed:.2f}s)", icon="✅")

vectorstore = st.session_state.vectorstore
retriever = st.session_state.retriever
llm = get_llm()


# ── RAG CALL ─────────────────────────────────────────────────────────────────
def rag_simple(query, retriever, llm):
    results = retriever.retrieve(query, top_k=3)
    context = "\n\n".join([d["content"] for d in results]) if results else ""

    if not context:
        return "No relevant context found in your documents."

    prompt = f"""You are an expert assistant.

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


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 RAG Doc Chat")

    doc_count = 0
    try:
        doc_count = vectorstore.collection.count()
    except Exception:
        pass

    st.info(f"📚 {doc_count} documents indexed")
    st.markdown("---")

    if st.button("🔄 Build / Rebuild Index"):
        from ingest import ingest_documents

        with st.spinner("Indexing documents..."):
            new_vectorstore, new_embedding_manager = ingest_documents()

        # ── FIX: clear cache AND update session state so the new pipeline is used
        st.cache_resource.clear()
        st.session_state.vectorstore = new_vectorstore
        st.session_state.retriever = RAGRetriever(new_vectorstore, new_embedding_manager)

        st.success("Index rebuilt successfully!")
        st.rerun()

    if st.button("↺ Reset App"):
        st.cache_resource.clear()
        st.session_state.clear()
        st.rerun()

    st.caption("Built with RAG + Groq")


# ── MAIN UI ───────────────────────────────────────────────────────────────────
st.title("Chat with your Documents")
st.markdown("Ask questions over your indexed documents")

# ── CHAT HISTORY ─────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── INPUT ─────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if retriever is None:
            st.error("RAG pipeline not loaded.")
            st.stop()

        answer = rag_simple(prompt, retriever, llm)
        st.markdown(answer)

        st.markdown("**Sources:**")
        results = retriever.retrieve(prompt, top_k=3)

        if not results:
            st.write("No sources found.")
        else:
            for i, doc in enumerate(results, 1):
                meta = doc.get("metadata", {})
                source = meta.get("source_file", "Unknown")
                page = meta.get("page", "N/A")

                with st.expander(f"{i}. {source} · page {page}"):
                    st.write(doc["content"][:500] + "...")

        st.session_state.messages.append({"role": "assistant", "content": answer})

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("*Powered by Groq + RAG Pipeline*")