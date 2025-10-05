"""gemma_rag.py

Plug-and-play script to:
- create embeddings via OpenRouter (Gemma-compatible)
- build a simple FAISS (optional) or NumPy retriever
- query Gemma (chat completions) with retrieved context

Usage examples:
  # Ingest documents from ./docs and save index
  python gemma_rag.py ingest --docs-dir ./docs --out index.npz

  # Query the saved index
  python gemma_rag.py query --index index.npz --query "How are exoplanets detected using transits?"

This script will fall back to a NumPy-based retriever if faiss is not installed.
Set your OpenRouter API key in the environment as OPENROUTER_API_KEY or create a .env file.
"""

from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import requests
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # it's ok if python-dotenv isn't installed; we just rely on env vars
    pass

try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False


OPENROUTER_EMBED_URL = "https://openrouter.ai/api/v1/embeddings"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemma-2-9b-it:free"


def get_api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("GEMMA_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment. Set it or add .env file.")
    return api_key


def call_embeddings(texts: List[str], model: str = DEFAULT_MODEL) -> List[List[float]]:
    """Call OpenRouter embeddings endpoint and return list of vectors."""
    api_key = get_api_key()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "input": texts}
    res = requests.post(OPENROUTER_EMBED_URL, headers=headers, json=payload)
    res.raise_for_status()
    data = res.json()
    # OpenRouter returns data: [{embedding: [...]}, ...]
    return [entry["embedding"] for entry in data["data"]]


def call_chat(messages: List[dict], model: str = DEFAULT_MODEL, temperature: float = 0.0) -> str:
    api_key = get_api_key()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature}
    res = requests.post(OPENROUTER_CHAT_URL, headers=headers, json=payload)
    res.raise_for_status()
    data = res.json()
    # data['choices'][0]['message']['content'] per OpenRouter chat completions
    return data["choices"][0]["message"]["content"]


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


class SimpleRetriever:
    def __init__(self, vectors: np.ndarray, docs: List[str]):
        # vectors: (N, D) numpy array
        self.vectors = np.array(vectors, dtype=np.float32)
        # normalize for cosine similarity
        self.vectors = np.vstack([normalize(v) for v in self.vectors])
        self.docs = docs
        self.dim = self.vectors.shape[1]
        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(self.vectors)
        else:
            self.index = None

    def search(self, query_vector: np.ndarray, k: int = 4) -> List[Tuple[int, float]]:
        q = normalize(np.array(query_vector, dtype=np.float32))
        if self.index is not None:
            D, I = self.index.search(np.expand_dims(q, axis=0), k)
            # faiss returns D (similarities) and I (indices)
            results = [(int(I[0][i]), float(D[0][i])) for i in range(len(I[0])) if I[0][i] != -1]
            return results
        else:
            sims = np.dot(self.vectors, q)
            topk = np.argsort(-sims)[:k]
            return [(int(i), float(sims[i])) for i in topk]


def ingest_documents(docs_dir: Path) -> Tuple[List[str], np.ndarray]:
    """Load .txt and .md files from docs_dir and return (docs, vectors).

    If docs_dir doesn't exist or is empty, a small fallback sample is returned.
    """
    if not docs_dir.exists() or not docs_dir.is_dir():
        print(f"Docs dir {docs_dir} not found — using sample docs.")
        docs = [
            "Exoplanets are planets that orbit stars outside our solar system.",
            "The transit method detects exoplanets by measuring dips in a star's brightness.",
            "Radial velocity measures wobble in a star caused by orbiting planets.",
            "Direct imaging captures photons from the planet itself and is challenging."
        ]
    else:
        docs = []
        for p in sorted(docs_dir.iterdir()):
            if p.suffix.lower() in {".txt", ".md"}:
                try:
                    docs.append(p.read_text(encoding="utf-8"))
                except Exception:
                    print(f"Warning: couldn't read {p}")
        if not docs:
            print(f"No text files found in {docs_dir} — using sample docs.")
            docs = [
                "Exoplanets are planets that orbit stars outside our solar system.",
                "The transit method detects exoplanets by measuring dips in a star's brightness.",
            ]

    print(f"Creating embeddings for {len(docs)} documents...")
    vectors = call_embeddings(docs)
    vectors = np.array(vectors, dtype=np.float32)
    return docs, vectors


def save_index(path: Path, docs: List[str], vectors: np.ndarray) -> None:
    # Save as .npz for portability
    np.savez_compressed(path, vectors=vectors, docs=np.array(docs, dtype=object))
    print(f"Saved index to {path}")
    # if faiss installed, optionally save faiss index file too
    if HAS_FAISS:
        try:
            faiss.write_index(faiss.IndexFlatIP(vectors.shape[1]), str(path) + ".faiss")
        except Exception:
            pass


def load_index(path: Path) -> SimpleRetriever:
    d = np.load(path, allow_pickle=True)
    vectors = d["vectors"]
    docs = list(d["docs"])
    return SimpleRetriever(vectors, docs)


def answer_with_context(retriever: SimpleRetriever, query: str, k: int = 4) -> str:
    q_emb = call_embeddings([query])[0]
    hits = retriever.search(np.array(q_emb, dtype=np.float32), k=k)
    context_texts = []
    for idx, score in hits:
        context_texts.append(f"[doc-{idx} score={score:.4f}]\n" + retriever.docs[idx])

    # Compose messages: system instruction + user content with retrieved context
    system = (
        "You are a helpful research assistant. Use the provided retrieved documents to answer the user's question. "
        "If the answer is not found in the docs, say so and provide general knowledge."
    )
    user_content = """
    Use the following retrieved documents (separated below) to answer the question.
    ---
    {}\n
    Question: {}
    """.format("\n---\n".join(context_texts), query)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]

    answer = call_chat(messages)
    return answer


def main():
    parser = argparse.ArgumentParser(description="Gemma RAG demo using OpenRouter API")
    sub = parser.add_subparsers(dest="cmd")

    p_ingest = sub.add_parser("ingest", help="Ingest docs and create index")
    p_ingest.add_argument("--docs-dir", type=Path, default=Path("./docs"), help="Directory with .txt/.md files")
    p_ingest.add_argument("--out", type=Path, default=Path("index.npz"), help="Output index file (.npz)")

    p_query = sub.add_parser("query", help="Query a saved index")
    p_query.add_argument("--index", type=Path, default=Path("index.npz"), help="Index file to load")
    p_query.add_argument("--query", type=str, required=True, help="Text question to ask")
    p_query.add_argument("--k", type=int, default=4, help="Top-k docs to retrieve")

    args = parser.parse_args()

    if args.cmd == "ingest":
        docs, vectors = ingest_documents(args.docs_dir)
        save_index(args.out, docs, vectors)
    elif args.cmd == "query":
        retriever = load_index(args.index)
        ans = answer_with_context(retriever, args.query, k=args.k)
        print("\n===== ANSWER =====\n")
        print(ans)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
