import os
import uuid


class VectorStore:
    def __init__(self, persist_directory="vector_store", collection_name="pdf_documents"):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.persist_directory = os.path.join(base_dir, persist_directory)
        self.collection_name = collection_name

        self.client = None
        self.collection = None

        self._initialize_store()

    def _initialize_store(self):
        import chromadb
        import os

        try:
            os.makedirs(self.persist_directory, exist_ok=True)

            # IMPORTANT: safe client init (prevents tenant crash)
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=chromadb.Settings(
                    anonymized_telemetry=False
                )
            )

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name
            )

            print("Vector store initialized safely")
            print("Collection:", self.collection_name)

        except Exception as e:
            print("Vector DB init failed:", e)
            raise

    def add_documents(self, embeddings, texts, metadatas):
        if self.collection is None:
            raise Exception("Vector store not initialized")

        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=[str(uuid.uuid4()) for _ in range(len(texts))]
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