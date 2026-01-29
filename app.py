"""
RAG Document Chat - Streamlit UI
"""
import streamlit as st
import sys
sys.path.append("utils")

from utils.vectorstore import VectorStore
from utils.embeddings import EmbeddingManager
from query import RAGRetriever
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Page config
st.set_page_config(
    page_title="RAG Doc Chat",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.chat-message {
    padding: 1rem;
    border-radius: 1rem;
    margin-bottom: 1rem;
}
.user { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
.assistant { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "embedding_manager" not in st.session_state:
    st.session_state.embedding_manager = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None

@st.cache_resource
def load_rag_pipeline():
    """Load RAG components once"""
    with st.spinner("Loading your 608 documents..."):
        vectorstore = VectorStore()
        embedding_manager = EmbeddingManager()
        retriever = RAGRetriever(vectorstore, embedding_manager)
        
        st.session_state.vectorstore = vectorstore
        st.session_state.embedding_manager = embedding_manager
        st.session_state.retriever = retriever
        
        st.success(f"Loaded {vectorstore.collection.count()} documents!")
    return retriever

def rag_simple(query: str, retriever, llm, top_k: int = 3) -> str:
    """Your existing rag_simple function (copy from query.py)"""
    results = retriever.retrieve(query, top_k=top_k)
    context = "\n\n".join([doc['content'] for doc in results]) if results else ""
    
    if not context:
        return "No relevant context found in your documents."
    
    prompt = f"""Use the following context from your documents to answer concisely.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
    
    with st.spinner("Generating answer..."):
        response = llm.invoke(prompt)
    return response.content

# Initialize LLM
@st.cache_resource
def get_llm():
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",  # Your working model
        temperature=0.1,
        max_tokens=1024
    )

# Sidebar
with st.sidebar:
    st.title("RAG Doc Chat")
    st.info(f"**Docs Indexed:** {st.session_state.vectorstore.collection.count() if st.session_state.vectorstore else 'Loading...'}")
    
    st.markdown("---")
    if st.button("Refresh Index", type="secondary"):
        st.cache_resource.clear()
        st.rerun()
    
    st.markdown("---")
    st.caption("Built with ❤️ using your RAG pipeline")

# Main chat interface
st.title("Chat with your Documents")
st.markdown("**Ask anything about your AWS, Stripe, and OpenAI PDFs!**")

# Load RAG pipeline
if st.session_state.retriever is None:
    retriever = load_rag_pipeline()
else:
    retriever = st.session_state.retriever

llm = get_llm()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about AWS APIs, Stripe payments, or OpenAI agents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching your 608 documents..."):
            answer = rag_simple(prompt, retriever, llm, top_k=3)
        
        # Show sources
        st.markdown("**Sources:**")
        results = st.session_state.retriever.retrieve(prompt, top_k=3)
        for i, doc in enumerate(results, 1):
            with st.expander(f"Source {i}: {doc['metadata'].get('source_file', 'Unknown')} (Sim: {doc['similarity_score']:.3f})"):
                st.write(doc['content'][:500] + "...")
        
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

# Footer
st.markdown("---")
st.markdown("*Powered by your custom RAG pipeline + llama-3.1-8b-instant*")
