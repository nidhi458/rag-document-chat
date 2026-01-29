import sys
from typing import List, Dict, Any
sys.path.append("utils")  
from utils.vectorstore import VectorStore  
from utils.embeddings import EmbeddingManager

class RAGRetriever:
    """Handles query-based retrieval from the vector store"""

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        """
        Initialize the retriever

        Args:
        vector_store: Vector store containing document embeddings
        embedding_manager: Manager for generating query embeddings
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query

        Args:
            query: The search query
            top_k: Number of top results to return
            score_threshold: Min similarity score threshold
        
        Returns:
            List of dictionaries containing retrieved documents and metadata
        """
        print(f"Retrieving documents for query: '{query}' ")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")

        #Generate query embedding
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        #Search in vector store
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results = top_k
            )

            #Process results
            retrieved_docs = []

            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
    
                    if distance <= 0.8:  
                        similarity_score = 1 / (1 + distance)

                        retrieved_docs.append({
                            "id": doc_id,
                            "content": document,
                            "metadata": metadata,
                            "similarity_score": similarity_score,
                            "distance": distance,
                            "rank": i + 1
                        })

                
                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
            else:
                print("No documents found")
            
            return retrieved_docs
            
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []



from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()


groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant",  
    temperature=0.1,
    max_tokens=1024
)

# from langchain_ollama import ChatOllama

# # REPLACE LLM initialization:
# llm = ChatOllama(
#     model="llama3.2",  # Runs locally
#     temperature=0.1
# )

## 2. Simple RAG function: retrieve context + generate response
def rag_simple(query: str, retriever, llm, top_k: int = 3) -> str:
    results = retriever.retrieve(query, top_k=top_k)
    context = "\n\n".join([doc['content'] for doc in results]) if results else ""
    if not context:
        return "No relevant context found."
    prompt = f"""Use the following context to answer the question concisely.
Context: {context}
Question: {query}
Answer:"""
    response = llm.invoke(prompt)
    return response.content

# SINGLE CLEAN MAIN (DELETE all other main/test code)
if __name__ == "__main__":
    print("🔄 Initializing RAG system...")
    vectorstore = VectorStore()
    embedding_manager = EmbeddingManager()
    retriever = RAGRetriever(vectorstore, embedding_manager)
    
    print(f"💾 Loaded {vectorstore.collection.count()} documents")
    print("💬 RAG Ready! Type 'quit' to exit")
    print("-" * 50)
    
    while True:
        query = input("\n❓ Ask: ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        answer = rag_simple(query, retriever, llm, top_k=3)
        print(f"\n💡 Answer: {answer}")
        print("-" * 50)
