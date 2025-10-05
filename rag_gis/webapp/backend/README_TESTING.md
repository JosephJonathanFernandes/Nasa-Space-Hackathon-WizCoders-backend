Testing the webapp backend

Prerequisites
- Python environment with dependencies from `backend/requirements.txt` installed (use a virtualenv).
- Install dev deps: `pip install pytest httpx fastapi`

Run tests (Windows PowerShell)

```powershell
# from repository root
cd backend\rag_gis\webapp\backend
python -m pytest -q
```

Notes
- Tests use FastAPI's TestClient. If your environment doesn't have the optional packages used by the app (langchain, chroma, etc.), the tests mock those parts.
