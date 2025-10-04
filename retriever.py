
import os
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions


class Retriever:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        # ensure directory
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = None
        self.embedding_model = SentenceTransformer("all-mpnet-base-v2")
        # create or get collection
        try:
            self.collection = self.client.get_collection("nasa_exoplanets")
        except Exception:
            self.collection = self.client.create_collection("nasa_exoplanets")

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Naive sliding window chunker"""
        chunks = []
        start = 0
        length = len(text)
        while start < length:
            end = min(start + chunk_size, length)
            chunks.append(text[start:end])
            start = end - overlap
            if start < 0:
                start = 0
        return chunks

    def add_documents_from_string(self, text: str, source: str = "inline"):
        chunks = self._chunk_text(text)
        texts = [c.strip() for c in chunks if c.strip()]
        if not texts:
            return
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        ids = [f"{source}_{i}" for i in range(len(texts))]
        metadatas = [{"source": source, "chunk_index": i} for i in range(len(texts))]
        self.collection.add(documents=texts, embeddings=embeddings.tolist(), metadatas=metadatas, ids=ids)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q_emb = self.embedding_model.encode([query], convert_to_numpy=True)[0].tolist()
        results = self.collection.query(query_embeddings=[q_emb], n_results=top_k)
        # results contain ids, documents, metadatas, distances
        out = []
        for i in range(len(results.get("ids", [[]])[0])):
            out.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results.get("distances", [[]])[0][i],
            })
        return out
