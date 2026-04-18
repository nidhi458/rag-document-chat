import sys
from typing import List, Dict, Any
sys.path.append("utils")

from utils.vectorstore import VectorStore
from utils.embeddings import EmbeddingManager


class RAGRetriever:
    """Handles query-based retrieval from the vector store"""

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query
        """
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")

        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        try:
            # Search in vector store
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k * 5
            )

            retrieved_docs = []

            if results["documents"] and results["documents"][0]:
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                ids = results["ids"][0]

                # DEDUPLICATE SOURCES BY FILE + PAGE
                seen_sources = set()

                for i, (doc_id, document, metadata, distance) in enumerate(
                    zip(ids, documents, metadatas, distances)
                ):

                    source_key = (
                        metadata.get("source_file"),
                        metadata.get("page")
                    )

                    if distance <= 0.8 and source_key not in seen_sources:
                        seen_sources.add(source_key)   # IMPORTANT FIX

                        similarity_score = 1 / (1 + distance)

                        retrieved_docs.append({
                            "id": doc_id,
                            "content": document,
                            "metadata": metadata,
                            "similarity_score": similarity_score,
                            "distance": distance,
                            "rank": len(retrieved_docs) + 1
                        })

                    if len(retrieved_docs) >= top_k:
                        break

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


def rag_simple(query: str, retriever, llm, top_k: int = 3) -> str:
    """
    Simple RAG: retrieve context + generate response
    """
    results = retriever.retrieve(query, top_k=top_k)
    context = "\n\n".join([doc["content"] for doc in results]) if results else ""

    if not context:
        return "No relevant context found."

    prompt = f"""You are an expert assistant.

Answer the user's question in a natural and professional way using the information below.

Rules:
1. Give a direct answer.
2. Use your own words.
3. Be concise but informative.
4. Do not say phrases like "according to the context" or "based on the provided context".

Information:
{context}

Question:
{query}

Answer:"""

    response = llm.invoke(prompt)
    return response.content


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

        if query.lower() in ["quit", "exit", "q"]:
            print("👋 Goodbye!")
            break

        answer = rag_simple(query, retriever, llm, top_k=3)
        print(f"\n💡 Answer: {answer}")
        print("-" * 50)