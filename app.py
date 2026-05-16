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

st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #111827 40%, #1e293b 100%);
    color: white;
}

/* Remove ugly default padding */
.block-container {
    padding-top: 2rem;
    padding-bottom: 1rem;
    max-width: 1100px;
}

/* Title */
h1 {
    font-size: 3rem !important;
    font-weight: 800 !important;
    color: white !important;
    letter-spacing: -1px;
    margin-bottom: 0.3rem;
}

/* Subtitle */
p {
    font-size: 1rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.95);
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    border-radius: 18px;
    padding: 1rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
}

/* User message */
[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
}

/* Assistant message */
[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
    background: rgba(255,255,255,0.06);
}

/* Chat input */
.stChatInputContainer {
    background: rgba(15,23,42,0.95);
    border-top: 1px solid rgba(255,255,255,0.08);
    padding-top: 1rem;
}

/* Input box */
textarea {
    background: rgba(255,255,255,0.06) !important;
    color: white !important;
    border-radius: 14px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.6rem 1rem;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 20px rgba(37,99,235,0.35);
}

/* Expander */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.04);
    border-radius: 10px;
    color: white !important;
}

/* Info box */
.stAlert {
    border-radius: 14px;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.15);
    border-radius: 10px;
}

/* Hide Streamlit branding */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}

/* Sidebar toggle button */
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    position: fixed !important;
    top: 1rem;
    left: 1rem;
    z-index: 999999 !important;
    background: rgba(30,41,59,0.95) !important;
    border-radius: 12px !important;
    padding: 0.35rem !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}

/* Icon color */
[data-testid="collapsedControl"] svg {
    fill: white !important;
    color: white !important;
}

</style>
""", unsafe_allow_html=True)


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = VectorStore()
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
    """Your existing rag_simple function"""
    results = retriever.retrieve(query, top_k=top_k)
    context = "\n\n".join([doc['content'] for doc in results]) if results else ""
    
    if not context:
        return "No relevant context found in your documents."
    
    prompt = f"""You are an expert assistant.

Answer the user's question in a natural and professional way using the information below.

Rules:
1. Give a direct answer.
2. Use your own words.
3. Provide a helpful answer in 2 to 3 sentences.
4. Do NOT say phrases like "according to the provided context" or "based on the provided context".
5. Respond as if you are confidently answering from knowledge.

Information:
{context}

Question:
{query}

Answer:"""
    
    with st.spinner("Generating answer..."):
        response = llm.invoke(prompt)
    return response.content, results

# Initialize LLM
@st.cache_resource
def get_llm():
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
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
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching your 608 documents..."):
            answer, results = rag_simple(prompt, retriever, llm, top_k=3)

        # SHOW ANSWER FIRST
        st.markdown(answer)

        # SHOW SOURCES AFTER ANSWER
        st.markdown("**Sources:**")
        

        for i, doc in enumerate(results, 1):
            source_name = doc["metadata"].get("source_file", "Unknown")
            page_num = doc["metadata"].get("page", "N/A")

            with st.expander(f"Source {i}: {source_name} (Page {page_num})"):
                st.write(doc["content"][:500] + "...")

        # Save assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

# Footer
st.markdown("---")
st.markdown("*Powered by your custom RAG pipeline + llama-3.1-8b-instant*")



