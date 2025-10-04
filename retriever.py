
import os
import json
from typing import List, Dict, Any, Optional

import numpy as np
from config import GEMMA_API_KEY, client

# Try to import a Gemma SDK placeholder if available. Keep None if not installed.
try:
    import gemma as _gemma_placeholder
except Exception:
    _gemma_placeholder = None

class Retriever:
    """Lightweight, file-backed retriever using Gemma embeddings and numpy.

    Stores documents and metadata in a JSON index and embeddings in a numpy .npy file.
    This keeps the setup minimal and removes heavy local models.
    """

    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = os.path.abspath(persist_directory)
        os.makedirs(self.persist_directory, exist_ok=True)

        self.index_path = os.path.join(self.persist_directory, "index.json")
        self.emb_path = os.path.join(self.persist_directory, "embeddings.npy")
        # pick an embedding model; can be overridden with GEMMA_EMBEDDING_MODEL env var
        self.embedding_model = os.environ.get("GEMMA_EMBEDDING_MODEL", "text-embedding-3-small")

        # placeholder.client api key is configured by config.py at import time

        # in-memory structures
        self.docs: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None

        self._load_index()

    def _load_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.emb_path):
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    self.docs = json.load(f)
                self.embeddings = np.load(self.emb_path)
            except Exception:
                # if loading fails, start fresh
                self.docs = []
                self.embeddings = None
        else:
            self.docs = []
            self.embeddings = None

    def _save_index(self):
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self.docs, f, ensure_ascii=False)
        if self.embeddings is not None:
            np.save(self.emb_path, self.embeddings)

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Simple sliding-window chunker (character-based)."""
        chunks = []
        start = 0
        length = len(text)
        while start < length:
            end = min(start + chunk_size, length)
            chunks.append(text[start:end])
            start = end - overlap
            if start < 0:
                start = 0
            if end == length:
                break
        return chunks

    def _get_embedding(self, text: str) -> List[float]:
    # Gemma (via placeholder client) embedding API call for a single input
    # require either legacy gemma.api_key or the modern client from config
        from config import client as _client
        if (_gemma_placeholder is None or not getattr(_gemma_placeholder, 'api_key', None)) and _client is None:
            # No GEMMA credentials: fall back to a local sentence-transformers model
            try:
                from sentence_transformers import SentenceTransformer
            except Exception:
                raise RuntimeError("GEMMA_API_KEY not configured and sentence-transformers not installed; cannot create embeddings")
            # use a small, fast model suitable for local use
            model = SentenceTransformer('all-MiniLM-L6-v2')
            vec = model.encode(text)
            return vec.tolist()
        # Prefer the modern client if available
        if _client is not None:
            resp = _client.embeddings.create(model=self.embedding_model, input=[text])
            return resp.data[0].embedding
        # Fallback to legacy API surface on the placeholder SDK if present
        if _gemma_placeholder is not None:
            resp = _gemma_placeholder.Embedding.create(model=self.embedding_model, input=[text])
            return resp["data"][0]["embedding"]
        # No embedding provider available
        raise RuntimeError("No Gemma client available and sentence-transformers not installed; cannot create embeddings")

    def add_documents_from_string(self, text: str, source: str = "inline"):
        """Chunk text, compute embeddings via Gemma, and persist to disk."""
        chunks = self._chunk_text(text)
        texts = [c.strip() for c in chunks if c.strip()]
        if not texts:
            return

        # compute embeddings one-by-one to avoid large single requests
        new_embs = []
        for t in texts:
            emb = self._get_embedding(t)
            new_embs.append(np.array(emb, dtype=np.float32))

        new_embs = np.vstack(new_embs)

        start_index = len(self.docs)
        ids = [f"{source}_{start_index + i}" for i in range(len(texts))]
        for i, t in enumerate(texts):
            self.docs.append({
                "id": ids[i],
                "document": t,
                "metadata": {"source": source, "chunk_index": start_index + i},
            })

        if self.embeddings is None or self.embeddings.size == 0:
            self.embeddings = new_embs
        else:
            self.embeddings = np.vstack([self.embeddings, new_embs])

        self._save_index()

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return top_k documents nearest to the query using cosine similarity.

        distance is returned as 1 - cosine_similarity so smaller is better.
        """
        if not self.docs or self.embeddings is None or self.embeddings.shape[0] == 0:
            return []

        q_emb = np.array(self._get_embedding(query), dtype=np.float32)

        # cosine similarity
        emb_norms = np.linalg.norm(self.embeddings, axis=1)
        q_norm = np.linalg.norm(q_emb)
        denom = emb_norms * (q_norm if q_norm != 0 else 1e-10)
        denom[denom == 0] = 1e-10
        sims = (self.embeddings @ q_emb) / denom

        top_k = min(top_k, sims.shape[0])
        idx = np.argsort(-sims)[:top_k]

        out: List[Dict[str, Any]] = []
        for i in idx:
            d = self.docs[int(i)]
            out.append({
                "id": d.get("id"),
                "document": d.get("document"),
                "metadata": d.get("metadata"),
                "distance": float(1.0 - float(sims[int(i)])),
            })
        return out

