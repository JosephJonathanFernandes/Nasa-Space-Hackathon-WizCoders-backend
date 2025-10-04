
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
            # No Gemma credentials: try local sentence-transformers model first, then a deterministic
            # hashing-based fallback if sentence-transformers isn't installed. The fallback is
            # deterministic but low-quality for semantic search; it's only to keep the app running
            # when no external embedding provider is available.
            try:
                from sentence_transformers import SentenceTransformer
            except Exception:
                allow_fallback = os.environ.get("GEMMA_ALLOW_LOCAL_FALLBACK", "1")
                if allow_fallback.lower() in ("1", "true", "yes"):
                    # use a deterministic hash-based embedding as a last-resort fallback
                    return self._local_hash_embedding(text, dim=384)
                # provide a clearer, actionable error message if fallback disabled
                raise RuntimeError(
                    "No Gemma client configured (GEMMA_API_KEY/GEMMA_BASE_URL) and the 'sentence-transformers' package is not installed.\n"
                    "Install it with: pip install sentence-transformers\n"
                    "Or set GEMMA_API_KEY / GEMMA_BASE_URL environment variables so the service client can be used.\n"
                    "If you want a low-quality deterministic local fallback instead, set GEMMA_ALLOW_LOCAL_FALLBACK=1 in the environment."
                )
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
        # This path should be unreachable because the earlier branch either
        # returns or raises, but keep a defensive error message.
        raise RuntimeError(
            "No Gemma client available and sentence-transformers not installed; cannot create embeddings.\n"
            "Install sentence-transformers or provide GEMMA_API_KEY / GEMMA_BASE_URL."
        )

    def _local_hash_embedding(self, text: str, dim: int = 384) -> List[float]:
        """Deterministic fallback embedding based on SHA256 hashing.

        This produces a pseudo-embedding of length `dim` with values in [-1, 1].
        It's deterministic and fast but not semantically meaningful. Use only when
        real embeddings are unavailable and you prefer the app to run.
        """
        import hashlib

        out = []
        i = 0
        # Generate enough hash material by hashing text||counter
        while len(out) < dim:
            h = hashlib.sha256()
            h.update(text.encode('utf-8'))
            h.update(i.to_bytes(2, 'little'))
            digest = h.digest()
            # split digest into 8-byte chunks and convert to floats
            for j in range(0, len(digest), 4):
                if len(out) >= dim:
                    break
                chunk = digest[j:j+4]
                val = int.from_bytes(chunk, 'little', signed=False)
                # normalize into [-1, 1]
                out.append((val / 0xFFFFFFFF) * 2.0 - 1.0)
            i += 1
        return out[:dim]

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

