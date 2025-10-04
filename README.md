# NASA Space Hackathon — Backend

This repository contains the backend prototype for the NASA Hackathon project. The RAG (Retrieval Augmented Generation) service lives under `backend/RAG` and provides endpoints for ingesting documents and querying a retrieval-augmented LLM chat endpoint.

Quick start

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r backend\RAG\requirements.txt
```

2. Copy env example and set your OpenAI key:

```powershell
copy backend\RAG\.env.example backend\RAG\.env
# Edit backend\RAG\.env and set OPENAI_API_KEY
```

3. Run the RAG server (FastAPI):

```powershell
uvicorn backend.RAG.app:app --reload --port 8000
```

Contributing

Please follow the code style used in the project and supply small, focused PRs. See `backend/RAG/README.md` for RAG-specific instructions.
