"""
RAG Document Chat - Streamlit UI (Stable Version)
"""

import streamlit as st
import sys
import os
from dotenv import load_dotenv

sys.path.append("utils")

from utils.vectorstore import VectorStore
from utils.embeddings import EmbeddingManager
from query import RAGRetriever
from langchain_groq import ChatGroq

# ── ENV ─────────────────────────────────────
load_dotenv()

# ── PAGE CONFIG ─────────────────────────────
st.set_page_config(
    page_title="RAG Doc Chat",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── SESSION STATE ────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "embedding_manager" not in st.session_state:
    st.session_state.embedding_manager = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None


# ── SAFE VECTORSTORE ACCESS ─────────────────
def get_vectorstore():
    return st.session_state.vectorstore


# ── RAG PIPELINE LOADER ─────────────────────
@st.cache_resource
def load_rag_pipeline():
    try:
        with st.spinner("Loading RAG pipeline..."):

            vectorstore = VectorStore()
            embedding_manager = EmbeddingManager()
            retriever = RAGRetriever(vectorstore, embedding_manager)

            st.session_state.vectorstore = vectorstore
            st.session_state.embedding_manager = embedding_manager
            st.session_state.retriever = retriever

            return retriever

    except Exception as e:
        st.error(f"Pipeline load failed: {e}")
        return None


# ── SAFE RETRIEVER ACCESS ───────────────────
def get_retriever():
    if st.session_state.retriever is None:
        st.session_state.retriever = load_rag_pipeline()
    return st.session_state.retriever


# ── LLM ─────────────────────────────────────
@st.cache_resource
def get_llm():
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=1024
    )


# ── SAFE RAG FUNCTION ───────────────────────
def rag_simple(query, retriever, llm, top_k=3):

    if retriever is None:
        return "RAG system not initialized."

    try:
        results = retriever.retrieve(query, top_k=top_k)
    except Exception as e:
        return f"Retrieval error: {e}"

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


# ── INIT PIPELINE ───────────────────────────
retriever = get_retriever()
llm = get_llm()


# ── SIDEBAR ────────────────────────────────
with st.sidebar:
    st.title("📄 RAG Doc Chat")

    vs = get_vectorstore()

    doc_count = 0
    if vs and hasattr(vs, "collection") and vs.collection:
        try:
            doc_count = vs.collection.count()
        except:
            doc_count = 0

    st.info(f"**{doc_count}** documents indexed and ready")

    st.markdown("---")

    if st.button("↺ Refresh Index"):
        st.cache_resource.clear()
        st.session_state.retriever = None
        st.rerun()

    st.caption("Built with RAG + Groq")


# ── MAIN UI ────────────────────────────────
st.title("Chat with your Documents")
st.markdown("Ask questions over AWS, Stripe, OpenAI docs")


# ── CHAT HISTORY ───────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ── INPUT ───────────────────────────────────
if prompt := st.chat_input("Ask anything..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        if retriever is None:
            st.error("RAG pipeline failed to initialize.")
            st.stop()

        answer = rag_simple(prompt, retriever, llm, top_k=3)
        st.markdown(answer)

        # ── SOURCES ──
        st.markdown("**Sources:**")

        try:
            results = retriever.retrieve(prompt, top_k=3)
        except Exception:
            results = []

        for i, doc in enumerate(results, 1):
            meta = doc.get("metadata", {})
            source = meta.get("source_file", "Unknown")
            page = meta.get("page", "N/A")

            with st.expander(f"{i}. {source} · page {page}"):
                st.write(doc["content"][:500] + "...")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )


# ── FOOTER ─────────────────────────────────
st.markdown("---")
st.markdown("*Powered by Groq + RAG Pipeline*")