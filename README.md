# RAG Backend (Prototype)

This folder contains a small prototype Retrieval-Augmented Generation (RAG) backend using FastAPI, SentenceTransformers for embeddings, and Chroma as the vector store.

Folders / files
- `app.py` - FastAPI server exposing `/ingest` and `/query` endpoints
- `ingest.py` - simple ingestion utility to load text files from a directory, chunk them, embed and store into Chroma
- `retriever.py` - helper functions for retrieving top-k passages
- `query_example.py` - small client that calls `/query`
- `requirements.txt` - python deps

Notes
- This is a prototype. For production use Pinecone/Weaviate/managed DB and secure your API keys.

Setup

1. Create a virtualenv and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

2. Create your `.env` from the example and set `OPENAI_API_KEY`:

```powershell
copy .env.example .env
# edit .env and set OPENAI_API_KEY
```

Running

```powershell
uvicorn app:app --reload --port 8000
```

Endpoints

- `POST /ingest` - upload text files to ingest into Chroma
- `POST /chat` - query the RAG chat. Request payload: `{ "question":"...","top_k":5, "stream": false }`


