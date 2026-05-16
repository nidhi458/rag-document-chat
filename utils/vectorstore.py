import chromadb
import os

class VectorStore:
    def __init__(self, persist_directory="vector_store", collection_name="pdf_documents"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        self.client = None
        self.collection = None

        self._initialize_store()

    def _initialize_store(self):
        """Stable ChromaDB initialization (Streamlit Cloud safe)"""

        try:
            os.makedirs(self.persist_directory, exist_ok=True)

            # ✅ FIX: force safe client config (avoids tenant system crash)
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=chromadb.config.Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # ✅ FIX: avoid recreate/delete logic completely
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name
            )

            print("Vector store initialized successfully")
            print(f"Collection: {self.collection_name}")

        except Exception as e:
            print(f"Vector store init error: {e}")
            self.client = None
            self.collection = None

    def add_documents(self, embeddings, texts, metadatas):
        if self.collection is None:
            raise Exception("Vector store not initialized")

        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=[str(i) for i in range(len(texts))]
        )

    def query(self, query_embedding, top_k=3):
        if self.collection is None:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        docs = []
        if results and results.get("documents"):
            for i in range(len(results["documents"][0])):
                docs.append({
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {}
                })

        return docs