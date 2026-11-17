
"""
Complete RAG document ingestion pipeline
"""

import streamlit as st

from utils.loaders import process_all_pdfs
from utils.splitter import split_documents
from utils.embeddings import EmbeddingManager
from utils.vectorstore import VectorStore


@st.cache_resource(show_spinner=False)
def ingest_documents(pdf_directory: str = "data/documents"):

    print("Starting RAG ingestion pipeline...")

    # 1. Initialize vector store and embedding model
    vectorstore = VectorStore()
    embedding_manager = EmbeddingManager()

    # 2. Reuse previously indexed documents
    existing_count = vectorstore.collection.count()

    if existing_count > 0:
        print(
            f"Existing collection found: "
            f"{existing_count} chunks."
        )
        print("Skipping duplicate ingestion.")

        return vectorstore, embedding_manager

    # 3. Load PDFs
    documents = process_all_pdfs(pdf_directory)

    if not documents:
        raise ValueError(
            f"No PDF documents found in {pdf_directory}"
        )

    # 4. Split documents
    chunks = split_documents(documents)

    if not chunks:
        raise ValueError("No text chunks generated.")

    # 5. Generate embeddings
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    embeddings = embedding_manager.generate_embeddings(
        texts
    )

    # 6. Store embeddings without deleting collection
    vectorstore.add_documents(
        embeddings,
        texts,
        metadatas
    )

    print(
        "Ingestion complete! Stored docs:",
        vectorstore.collection.count()
    )

    return vectorstore, embedding_manager


if __name__ == "__main__":
    vectorstore, embedding_manager = ingest_documents()
    print("\nPipeline ready! Run: python query.py")
