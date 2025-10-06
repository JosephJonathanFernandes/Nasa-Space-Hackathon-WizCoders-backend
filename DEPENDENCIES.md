# Dependencies & Configuration

This document explains the options for embeddings and LLM backends used by the Exoscope backend, environment variables you may need to set, and PowerShell examples for local development on Windows.

Supported options

1) GROQ API key (preferred)
   - If you are using GROQ (or a GROQ-compatible provider) for embeddings, retrieval, or chat, set `GROQ_API_KEY` and optionally `GROQ_BASE_URL`.

   PowerShell example (current session):

   ```powershell
   $env:GROQ_API_KEY = 'your_groq_key_here'
   $env:GROQ_BASE_URL = 'https://api.sanity.io' # or your provider's endpoint
   ```

2) GROQ API key (Sanity GROQ or other GROQ-compatible services)
   - If you are using a GROQ API for embeddings or search, set `GROQ_API_KEY` and optionally `GROQ_BASE_URL`.
   - Update the backend configuration to prefer GROQ when `GROQ_API_KEY` is present.

   PowerShell example:

   ```powershell
   $env:GROQ_API_KEY = 'your_groq_key_here'
   $env:GROQ_BASE_URL = 'https://api.sanity.io' # or your provider's endpoint
   ```

   Notes:
   - The backend will attempt to use configured API keys in the following preference order: GROQ (if configured), other configured provider (e.g., OpenRouter), local sentence-transformers fallback.
   - If you want the backend to use GROQ only for retrieval while keeping another provider for chat, configure both keys and the appropriate client selectors in your env or settings file.

3) Local sentence-transformers (on-machine embeddings)
   - Install local models when you want embeddings to be computed locally:

   ```powershell
   python -m pip install -r requirements.txt
   pip install sentence-transformers
   ```

   - Note: Local sentence-transformers require extra disk space and may have significant CPU/GPU requirements depending on the model.

Fallback behavior

- If no external API is available and `sentence-transformers` is not installed, the app can fall back to a deterministic SHA256-based embedding placeholder for development. This is NOT semantically meaningful and should only be used for local testing.
- To explicitly allow or disallow fallback, set the environment variable `ALLOW_LOCAL_FALLBACK` to `'1'` (allow) or `'0'` (disable). Default: allow.

PowerShell examples

Set environment variables for a single PowerShell session:

```powershell
   $env:GROQ_API_KEY = 'your_groq_key_here'
   $env:OPENROUTER_API_KEY = 'your_openrouter_key_here'
   $env:OPENROUTER_BASE_URL = 'https://api.openrouter.example'
   $env:ALLOW_LOCAL_FALLBACK = '1'
```

Persist variables across sessions by adding them to System Environment Variables or your PowerShell profile.

Ingestion and PDF support

- PDF ingestion requires `PyPDF2` (included in `requirements.txt` in RAG scripts). Use `ingest.py` in the RAG toolchain to ingest `.txt` and `.pdf` files into the retriever index.

Troubleshooting

- If embeddings look wrong or similarity is poor, double-check which API key is being used and whether a local model is inadvertently being used as a fallback.
- For connection errors, verify `*_BASE_URL` and network access, and that API keys are valid and not expired.

Examples

- Ingest and build a local index (PowerShell):

```powershell


python backend\RAG\rag_demo.py ingest --docs-dir backend\RAG\docs --out backend\RAG\index.npz
```

- Query the saved index:

```powershell


python backend\RAG\rag_demo.py query --index backend\RAG\index.npz --query "How are exoplanets detected using the transit method?"
```

If you'd like I can also add a small example `env.example` file and a short code sample showing how to initialize the GROQ client in the backend code — tell me the preferred Python library or client conventions you want to use (we can implement a tiny wrapper if needed).