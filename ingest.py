"""
Complete RAG document ingestion pipeline
"""
from pathlib import Path
import sys

# Add utils to path
sys.path.append("utils")

from utils.loaders import process_all_pdfs
from utils.splitter import split_documents
from utils.embeddings import EmbeddingManager
from utils.vectorstore import VectorStore

def ingest_documents(pdf_directory: str = "data/documents"):
    print("Starting RAG ingestion pipeline...")

    # 1. LOAD PDFs
    documents = process_all_pdfs(pdf_directory)

    # 2. SPLIT
    chunks = split_documents(documents)

    # 3. EMBEDDINGS
    embedding_manager = EmbeddingManager()
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedding_manager.generate_embeddings(texts)

    # 4. STORE — delete old collection first, then recreate
    vectorstore = VectorStore()
    vectorstore.client.delete_collection(vectorstore.collection_name)        # ← add this
    vectorstore.collection = vectorstore.client.create_collection(           # ← and this
        name=vectorstore.collection_name
    )

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    vectorstore.add_documents(embeddings, texts, metadatas)

    print("Ingestion complete! Stored docs:", vectorstore.collection.count())
    return vectorstore, embedding_manager

if __name__ == "__main__":
    # Run pipeline
    vectorstore, embedding_manager = ingest_documents()
    print("\n Pipeline ready! Run: python query.py")
