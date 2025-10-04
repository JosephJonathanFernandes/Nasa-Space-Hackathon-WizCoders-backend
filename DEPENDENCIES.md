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
