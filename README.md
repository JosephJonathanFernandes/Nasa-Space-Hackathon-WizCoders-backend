
# Exoscope Backend — Professional Edition

## Problem Statement

This backend powers Exoscope, a NASA Space Apps project, providing secure, modular, and scalable APIs for retrieval-augmented generation (RAG), ML model serving, and data ingestion. Designed for clarity, extensibility, and security, it is ready for production and open-source collaboration.

## Architecture

- **src/**: Core logic (API, models, services, utils, config)
- **tests/**: Unit and integration tests
- **docs/**: Architecture, design, and contribution docs
- **config/**: Environment and deployment configuration
- **scripts/**: Automation and developer utilities

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for a detailed overview.

## Tech Stack
- Python 3.10+
- FastAPI (API layer)
- Pydantic (validation)
- Uvicorn (ASGI server)
- ChromaDB (vector DB)
- Pytest (testing)
- Pre-commit, Black, Ruff (linting/formatting)

## Setup & Usage

1. **Clone and prepare environment**
	```sh
	git clone <repo-url>
	cd backend
	python -m venv .venv
	source .venv/bin/activate  # or .venv\\Scripts\\Activate.ps1 on Windows
	pip install -r requirements.txt
	cp .env.example .env  # Set secrets and config
	```
2. **Run the API**
	```sh
	uvicorn src.api.main:app --reload --port 8000
	```
3. **Run tests**
	```sh
	pytest
	```

## Configuration
- All secrets/config via `.env` (never hardcoded)
- See `.env.example` and `config/` for details

## Security
- No secrets in code (see [SECURITY.md](SECURITY.md))
- Input validation and error handling throughout
- Dependency updates and static checks required

## Testing
- Tests in `tests/` (unit & integration)
- Example: `pytest tests/unit/`
- Coverage goal: 90%+

## Dev Experience
- Lint: `ruff src/`
- Format: `black src/`
- Pre-commit: see `.pre-commit-config.yaml`
- CI: GitHub Actions (see below)

## GitHub Actions CI
- Lint, test, and security scan on every PR
- Example workflow in `.github/workflows/ci.yml`

## Value
- Modular, secure, and production-ready
- Easy to extend and maintain
- Recruiter- and reviewer-friendly

---

For more, see [docs/](docs/) and [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md).
## Live demo / Deployments

