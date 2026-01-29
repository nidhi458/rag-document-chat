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
    """Complete pipeline: load → split → embed → store"""
    
    print("Starting RAG ingestion pipeline...")
    
    # 1. LOAD PDFs
    print("\n Step 1: Loading PDFs...")
    documents = process_all_pdfs(pdf_directory)
    print(f"✓ Loaded {len(documents)} pages")
    
    # 2. SPLIT into chunks
    print("\n Step 2: Splitting documents...")
    chunks = split_documents(documents)
    print(f"✓ Created {len(chunks)} chunks")
    
    # 3. GENERATE embeddings
    print("\n Step 3: Generating embeddings...")
    embedding_manager = EmbeddingManager()
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedding_manager.generate_embeddings(texts)
    print(f"✓ Generated embeddings shape: {embeddings.shape}")
    
    # 4. STORE in vector DB
    print("\n Step 4: Storing in vector database...")
    vectorstore = VectorStore()
    vectorstore.add_documents(chunks, embeddings)
    print("Ingestion complete!")
    
    return vectorstore, embedding_manager

if __name__ == "__main__":
    # Run pipeline
    vectorstore, embedding_manager = ingest_documents()
    print("\n Pipeline ready! Run: python query.py")
