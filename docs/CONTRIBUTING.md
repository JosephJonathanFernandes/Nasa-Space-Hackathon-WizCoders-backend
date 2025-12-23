# Contributing Guidelines

Thank you for considering contributing!

## How to Contribute
- Fork the repository and create a new branch
- Make your changes with clear, descriptive commits
- Ensure code passes linting and tests
- Submit a pull request with a clear description

## Code Style
- Follow PEP8, Clean Code, and SOLID principles
- Use meaningful names and docstrings
- Avoid hardcoded secrets; use environment variables (see `.env.example`)

## Security
- Never commit credentials or secrets
- Validate and sanitize all inputs


## Linting & Formatting
- Run `ruff src/` and `black src/` before committing
- Use `pre-commit install` to enable automatic checks

## Tests
- Add or update unit/integration tests for your changes
- Run all tests (`pytest`) before submitting

## Communication
- Be respectful and collaborative
- See CODE_OF_CONDUCT.md for behavior standards
