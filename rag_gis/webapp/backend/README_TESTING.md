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

Endpoint: /upload_fits/

This endpoint accepts a single file upload (FITS) and returns a PNG plot of the light curve produced by `lightkurve`.

Example (PowerShell):

```powershell
# Upload a FITS file and save the returned image
curl -X POST "http://127.0.0.1:8000/upload_fits/" -F "file=@C:\path\to\example.fits" --output lc_plot.png
```

Notes:
- Ensure `matplotlib` and `lightkurve` are installed (they were added to `backend/requirements.txt`).
- The endpoint uses `lightkurve.read` to parse the FITS file and will pick the first LightCurve if a collection is returned.
