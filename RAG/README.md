# NASA Space Hackathon — Backend

This repository contains the backend prototype for the NASA Hackathon project. The RAG (Retrieval Augmented Generation) service lives under `backend/RAG` and provides endpoints for ingesting documents and querying a retrieval-augmented LLM chat endpoint.

Quick start

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r backend\RAG\requirements.txt
```

2. Copy env example and set your Gemma key:

```powershell
copy backend\RAG\.env.example backend\RAG\.env
# Edit backend\RAG\.env and set GEMMA_API_KEY or GEMMA_BASE_URL and GEMMA_TOKEN
```

3. Run the RAG server (FastAPI):

```powershell
uvicorn backend.RAG.app:app --reload --port 8000
```

Contributing

Please follow the code style used in the project and supply small, focused PRs. See `backend/RAG/README.md` for RAG-specific instructions.

Quick demo: Gemma RAG script

After setting your OpenRouter API key (see below), you can run a small end-to-end demo that builds embeddings and answers queries using Gemma.

Set API key in PowerShell (or add to `.env` in this folder):

```powershell
setx OPENROUTER_API_KEY "your_openrouter_key_here"
```

Create a docs folder with some `.txt` or `.md` files (or skip to use the built-in sample docs), then:

```powershell
# Ingest documents and build a local index
python backend\RAG\gemma_rag.py ingest --docs-dir backend\RAG\docs --out backend\RAG\index.npz

# Query the saved index
python backend\RAG\gemma_rag.py query --index backend\RAG\index.npz --query "How are exoplanets detected using the transit method?"
```

The script uses OpenRouter (https://openrouter.ai) endpoints for embeddings and chat. Make sure `OPENROUTER_API_KEY` is set.

Model prediction endpoint

Place your pickled model artifacts in `backend/RAG/models/`. The server will look for these filenames by default:
- `lgbm_model.pkl` or `model.pkl` (the trained model)
- `scaler.pkl` (scikit-learn scaler used at training time)
- `le.pkl` or `label_encoder.pkl` (LabelEncoder used to decode labels)

Once files are placed, start the server and POST JSON to `/predict` on the running FastAPI server, for example:

```json
{ "features": [9.488036, 2.9575, 615.8, 2.26, 793.0, 93.59, 5455.0, 4.467, 0.927, 291.93423, 48.141651] }
```

The endpoint will return a JSON object containing the raw model output and a decoded `prediction` when a label encoder is present.
