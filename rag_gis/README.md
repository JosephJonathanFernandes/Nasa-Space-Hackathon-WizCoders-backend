# NASA Space Hackathon — Backend

This repository contains the backend prototype for the NASA Hackathon project. The RAG (Retrieval Augmented Generation) service lives under `backend/RAG` and provides endpoints for ingesting documents and querying a retrieval-augmented LLM chat endpoint.

Quick start

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r backend\RAG\requirements.txt
```

2. Copy env example and set your GROQ (or other) API key:

```powershell
copy backend\RAG\.env.example backend\RAG\.env
# Edit backend\RAG\.env and set GROQ_API_KEY or GROQ_BASE_URL (or other provider keys)
```

3. Run the RAG server (FastAPI):

```powershell
uvicorn backend.RAG.app:app --reload --port 8000
```

Contributing

Please follow the code style used in the project and supply small, focused PRs. See `backend/RAG/README.md` for RAG-specific instructions.

Quick demo: RAG script (GROQ / provider)

After setting your `GROQ_API_KEY` (or provider key, see above), you can run a small end-to-end demo that builds embeddings and answers queries using the configured provider.

Set API key in PowerShell (or add to `.env` in this folder):

```powershell
setx GROQ_API_KEY "your_groq_key_here"
```

Create a docs folder with some `.txt` or `.md` files (or skip to use the built-in sample docs), then:

```powershell

# Ingest documents and build a local index (script name may vary)
python backend\RAG\rag_demo.py ingest --docs-dir backend\RAG\docs --out backend\RAG\index.npz

# Query the saved index
python backend\RAG\rag_demo.py query --index backend\RAG\index.npz --query "How are exoplanets detected using the transit method?"
```

The script uses the configured provider (GROQ or other) for embeddings and chat. Make sure `GROQ_API_KEY` (or the provider's key) is set.
