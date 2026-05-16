import numpy as np
import chromadb
import uuid
import os
import shutil
from typing import List, Any


class VectorStore:
    """Manages document embeddings in a ChromaDB vector store"""

    def __init__(
        self,
        collection_name: str = "pdf_documents",
        persist_directory: str = "data/vector_store"
    ):
        """
        Initialize the vector store

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector store
        """

        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None

        self._initialize_store()

    def _initialize_store(self):
        """Initialize ChromaDB client and collection"""

        try:
            # Remove old incompatible vector store
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)

            # Recreate directory
            os.makedirs(self.persist_directory, exist_ok=True)

            # Create ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.persist_directory
            )

            # Create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name
            )

            print(f"Vector store initialized.")
            print(f"Collection name: {self.collection_name}")
            print(f"Documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """
        Add documents and embeddings to vector store

        Args:
            documents: List of LangChain documents
            embeddings: Corresponding embeddings
        """

        if len(documents) != len(embeddings):
            raise ValueError(
                "Number of documents must match number of embeddings"
            )

        print(f"Adding {len(documents)} documents to vector store...")

        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):

            # Unique ID
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            # Metadata
            metadata = dict(doc.metadata)

            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)

            metadatas.append(metadata)

            # Document text
            documents_text.append(doc.page_content)

            # Embedding
            embeddings_list.append(embedding.tolist())

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )

            print(
                f"Successfully added {len(documents)} documents."
            )

            print(
                f"Total documents in collection: "
                f"{self.collection.count()}"
            )

        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise


# Initialize vector store
vectorstore = VectorStore()
print(vectorstore)