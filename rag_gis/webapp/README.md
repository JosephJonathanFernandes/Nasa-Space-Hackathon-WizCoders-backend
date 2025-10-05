RAG Chat (React + FastAPI)

This minimal scaffold exposes a FastAPI backend that uses your existing Chroma persistence (chroma_db) and a React frontend.

Backend (Python)

- Path: `webapp/backend`
- Requirements: listed in `requirements.txt`

Quick start (recommended inside your project root):

# create and activate a venv (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# install backend deps
pip install -r webapp\backend\requirements.txt

# copy .env if you want to provide GROQ_API_KEY
copy webapp\backend\.env.example .env
# edit .env to set GROQ_API_KEY if you want LLM answers

# run backend
python -m uvicorn webapp.backend.main:app --host 0.0.0.0 --port 8000 --reload

Frontend (React + Vite)

- Path: `webapp/frontend`

# install frontend deps (PowerShell)
cd webapp\frontend
npm install
npm run dev

Notes
- The backend will attempt to load your Chroma database from `chroma_db` with collection name `testing` (the same names used in your notebook). Make sure the data exists and the backend process has read access.
- If you set `GROQ_API_KEY` in `.env`, the backend will try to use `langchain_groq.ChatGroq` for LLM answers. If not set, the API returns retrieved passages as the answer fallback.
- I kept CORS wide open for local development. Lock it down before production.

Next steps (I can help):
- Wire up conversational context and chat history for multi-turn QA.
- Deploy to a server or Dockerize both backend and frontend.
- Add authentication and rate-limiting.
