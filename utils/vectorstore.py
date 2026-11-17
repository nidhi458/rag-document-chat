
import os
import hashlib


class VectorStore:
    def __init__(
        self,
        persist_directory="vector_store",
        collection_name="pdf_documents"
    ):
        base_dir = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

        self.persist_directory = os.path.join(
            base_dir, persist_directory
        )

        self.collection_name = collection_name
        self.client = None
        self.collection = None

        self._initialize_store()

    def _initialize_store(self):
        import chromadb

        os.makedirs(
            self.persist_directory,
            exist_ok=True
        )

        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=chromadb.Settings(
                    anonymized_telemetry=False
                ),
                tenant="default_tenant",
                database="default_database"
            )

            self.collection = (
                self.client.get_or_create_collection(
                    name=self.collection_name
                )
            )

            print("Vector store initialized safely")
            print("Collection:", self.collection_name)

        except Exception as e:
            print("Vector DB init failed:", e)
            raise

    def add_documents(self, embeddings, texts, metadatas):

        if self.collection is None:
            raise RuntimeError("Vector store not initialized")

        if not texts:
            return

        if len(texts) != len(metadatas) or len(texts) != len(embeddings):
            raise ValueError("Embeddings, texts and metadata counts differ")

        # Generate deterministic IDs to prevent duplicates
        ids = [
            hashlib.sha256(
                f"{i}:{text}".encode("utf-8")
            ).hexdigest()
            for i, text in enumerate(texts)
        ]

        # Insert or update existing records
        self.collection.upsert(
            embeddings=(
                embeddings.tolist()
                if hasattr(embeddings, "tolist")
                else embeddings
            ),
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )

        print(f"Stored {len(texts)} document chunks.")

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
                    "metadata": (
                        results["metadatas"][0][i]
                        if results.get("metadatas")
                        else {}
                    )
                })

        return docs
