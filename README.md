# Exoscope — Backend

This folder contains the backend services for the Exoscope project created for the NASA Space Apps Global Hackathon. The backend provides the FastAPI services, model utilities, and RAG (retrieval-augmented generation) tooling used by the frontend application.

Important: The frontend and backend are maintained as separate repositories. Run each service independently during development and point the frontend to the backend URL using environment variables or `src/services/api.ts`.

## Table of contents

- Quick start
- Configuration
- Development
- RAG / Ingestion
- Dependencies
- Contributing
- Code of Conduct
- Security
- License

## Quick start

Prerequisites

- Python 3.10+ (or compatible)
- pip
- Optional: virtualenv or venv

Local development (example - PowerShell)

```powershell
# create and activate a virtual environment
python -m venv .venv; .\.venv\Scripts\Activate.ps1

# install dependencies
pip install -r requirements.txt

# run the FastAPI server
uvicorn rag_gis.webapp.backend.main:app --reload --port 8000
```

If your environment requires different module paths, consult the `rag_gis/README.md` and `DEPENDENCIES.md` for RAG-specific setup and examples.

## Configuration

- See `backend/DEPENDENCIES.md` for details on provider options (GROQ, OpenRouter, or local `sentence-transformers`) and environment variables such as `GROQ_API_KEY`, `GROQ_BASE_URL`, and `ALLOW_LOCAL_FALLBACK`.
- Copy any `.env.example` files in RAG subfolders to `.env` and set credentials as required.

## RAG / Ingestion

The repository contains a RAG prototype under `rag_gis/` with scripts for ingesting PDF/TXT sources and building a local retriever index. See `rag_gis/README.md` for a detailed walkthrough and demo commands.

## Dependencies

See `DEPENDENCIES.md` and `requirements.txt` for the full list of Python packages and configuration notes.

## Contributing

Please follow `CONTRIBUTING.md` (already present). Short summary:

- Open issues for bugs or features first.
- Create small, focused pull requests against `main`.
- Include tests where applicable.

## Code of Conduct

This project follows the Contributor Covenant. See `CODE_OF_CONDUCT.md` for details and reporting contacts.

## Security

Please see `SECURITY.md` in this folder for security disclosure guidance. If `SECURITY.md` does not exist yet, report privately to the maintainers (provide preferred contact) instead of public issues.

## License

This repository is licensed under the MIT License — see `LICENSE` for details.

## Acknowledgments

- NASA Kepler & TESS mission data
- NASA Space Apps Global Hackathon community

---

If you want me to refine any section (add SMOKE tests, create a Dockerfile, or add GitHub Actions), tell me which and I'll add it.
