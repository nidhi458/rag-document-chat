"""
RAG Document Chat - Streamlit UI (Stable Fix Version)
"""

import streamlit as st
import sys
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dotenv import load_dotenv

# 🔥 IMPORTANT: prevent torch/streamlit watcher crash
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
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── SESSION STATE ───────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


# ── LLM ─────────────────────────
@st.cache_resource
def get_llm():
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=1024
    )


# ── PIPELINE LOADER ─────────────
@st.cache_resource(show_spinner=True)
def load_rag_pipeline():
    vectorstore = VectorStore(persist_directory="vector_store")

    try:
        count = vectorstore.collection.count()
    except:
        count = 0

    st.write("VECTOR DB COUNT:", count)

    embedding_manager = EmbeddingManager()
    retriever = RAGRetriever(vectorstore, embedding_manager)

    return vectorstore, retriever

# ── INIT PIPELINE ───────────────
if st.session_state.retriever is None:
    vectorstore, retriever = load_rag_pipeline()
    st.session_state.vectorstore = vectorstore
    st.session_state.retriever = retriever


vectorstore = st.session_state.vectorstore
retriever = st.session_state.retriever
llm = get_llm()


# ── RAG CALL ────────────────────
def rag_simple(query, retriever, llm):

    results = retriever.retrieve(query, top_k=3)
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


# ── SIDEBAR ──────────────────────
with st.sidebar:
    st.title("📄 RAG Doc Chat")
    doc_count = 0
    try:
        doc_count = vectorstore.collection.count()
    except:
        doc_count = 0
    st.info(f"📚 {doc_count} documents indexed")

    st.markdown("---")

    if st.button("🔄 Build / Rebuild Index"):
        from ingest import ingest_documents

        with st.spinner("Indexing documents..."):
            vectorstore, embedding_manager = ingest_documents()

        st.success("Index built successfully!")
        st.rerun()
    if st.button("🧹 Reset Vector DB"):
        import shutil
        shutil.rmtree("vector_store", ignore_errors=True)
        st.success("Deleted DB. Re-run ingestion.")

    st.caption("Built with RAG + Groq")


# ── MAIN UI ──────────────────────
st.title("Chat with your Documents")
st.markdown("Ask questions over your indexed documents")
st.markdown("About AWS APIs, stripe payments and OpenAI docs")


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

        answer = rag_simple(prompt, retriever, llm)
        st.markdown(answer)

        # ── SOURCES ──
        st.markdown("**Sources:**")
        results = retriever.retrieve(prompt, top_k=3)

        if not results:
            st.write("No sources found")
        else:
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