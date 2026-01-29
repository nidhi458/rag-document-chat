# RAG Document Chat

A Retrieval-Augmented Generation (RAG) application that enables users to query PDF documents using semantic search and a large language model.

---

## Problem Statement

Large documents such as technical PDFs, API references, and guides contain valuable information, but extracting precise answers from them is time-consuming. Traditional keyword-based search (Ctrl+F) often fails because it cannot understand semantic meaning, synonyms, or context.

Using a Large Language Model (LLM) alone is also unreliable, as the model may hallucinate answers that are not grounded in the provided documents.

This project solves the problem by using **Retrieval-Augmented Generation (RAG)**, where relevant document content is first retrieved using semantic search and then provided as context to the LLM for accurate and grounded answers.

---

## Architecture Overview

The system follows a standard RAG pipeline:

1. PDF documents are loaded from a directory
2. Documents are split into smaller overlapping chunks
3. Each chunk is converted into an embedding (vector representation)
4. Embeddings are stored in a vector database (ChromaDB)
5. A user query is converted into an embedding
6. The vector database retrieves the most semantically similar chunks
7. Retrieved chunks are passed as context to the LLM
8. The LLM generates a final answer grounded in the document content

---

## Tech Stack

* Python
* SentenceTransformers (`all-MiniLM-L6-v2`)
* ChromaDB (vector database)
* Groq LLM (Llama-3.1-8B-Instant)
* Streamlit (UI)
* LangChain (document loading and utilities)

---

## How the System Works

### Document Ingestion

PDF documents are loaded using document loaders and split into smaller text chunks. Chunking is required because embedding models and LLMs have context length limits. Smaller chunks also improve retrieval accuracy.

Chunk overlap is used to preserve context across adjacent chunks so that important information is not lost at chunk boundaries.

### Embeddings

Each text chunk is converted into a numerical vector (embedding) using a sentence-transformer model. These embeddings capture the semantic meaning of the text, allowing similarity comparisons beyond exact keyword matching.

### Vector Storage

All embeddings, along with their corresponding text and metadata, are stored in ChromaDB. This allows efficient similarity search during query time.

### Retrieval

When a user submits a query:

* The query is converted into an embedding
* Cosine similarity is used internally by the vector database to find the top-k most relevant document chunks
* The most relevant chunks are selected as context

### Answer Generation

The retrieved context is combined with the user query and passed to the LLM using a structured prompt. Since the LLM answers strictly based on retrieved document content, hallucinations are significantly reduced.

---

## Limitations

* No reranking of retrieved documents
* Simple prompt structure without advanced prompt optimization
* No evaluation metrics for answer quality
* No authentication or multi-user support

---

## Future Improvements

* Add reranking for better retrieval quality
* Improve prompt engineering
* Support more document formats (Word, HTML)
* Convert Streamlit app to a FastAPI backend
* Add logging and monitoring
* Deploy the application publicly

---

## How to Run Locally

1. Clone the repository
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Add required environment variables (e.g., Groq API key)
4. Run the ingestion pipeline:

   ```bash
   python ingest.py
   ```
5. Start the application:

   ```bash
   streamlit run app.py
   ```

---

## Demo

The application runs locally using Streamlit and allows users to ask questions about the indexed PDF documents.

---

*This project was built as part of a structured GenAI learning roadmap to understand and implement Retrieval-Augmented Generation from scratch.*
