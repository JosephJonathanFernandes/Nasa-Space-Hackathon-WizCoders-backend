This project requires either:

1) A Gemma/OpenRouter-compatible API endpoint (preferred):
   - Set GEMMA_BASE_URL to the endpoint URL (for OpenRouter/Gemma-like HTTP API).
   - Set GEMMA_API_KEY (or OPENROUTER_API_KEY) to the bearer token if required.

   Example (PowerShell):
   $env:GEMMA_API_KEY = 'your_token_here'
   $env:GEMMA_BASE_URL = 'https://api.openrouter.example'

2) Or install local sentence-transformers for on-machine embeddings:
   - pip install sentence-transformers

Notes:
- If neither option is available the server will raise a RuntimeError when trying to compute embeddings.
- To install dependencies for the full FastAPI app (recommended):
  pip install -r requirements.txt

- On Windows PowerShell you can set environment variables for the current session using:
  $env:GEMMA_API_KEY = "your_api_key_here"
  # or
  $env:GEMMA_BASE_URL = "https://api.openrouter.example"

- To persist environment variables across sessions, use System Settings -> Environment Variables, or set them in your PowerShell profile.

PDF ingestion support
- The project now supports ingesting PDF files using `ingest.py`. This requires `PyPDF2` which is included in `requirements.txt`.
- Usage:
  python ingest.py <folder-containing-pdfs-and-txt>

The script will recursively find `.txt` and `.pdf` files under the folder and ingest their text into the retriever.

Local deterministic fallback embedding (optional):
- By default the app will use a deterministic SHA256-based embedding fallback when no Gemma client
  and no `sentence-transformers` are available. This keeps the server running for development, but
  embeddings produced this way are NOT semantically meaningful. To disable the fallback, set:
  $env:GEMMA_ALLOW_LOCAL_FALLBACK = '0'

Example: enable fallback (default):
  $env:GEMMA_ALLOW_LOCAL_FALLBACK = '1'

Example: explicitly disable fallback:
  $env:GEMMA_ALLOW_LOCAL_FALLBACK = '0'
