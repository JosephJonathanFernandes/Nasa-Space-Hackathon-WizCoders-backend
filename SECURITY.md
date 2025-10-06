# Security Policy

If you discover a security vulnerability in the backend, please report it responsibly.

Preferred reporting

- Do NOT open a public GitHub issue for security problems.
- Contact the maintainers privately at <security@your-domain.example> (replace with a real email), or use the contact listed in `CODE_OF_CONDUCT.md`.

What to include in your report

- Affected component or endpoint
- Steps to reproduce
- Proof-of-concept code or exploit (where safe)
- Suggested mitigations or fixes

Response

We will acknowledge the report within 3 business days and work with you to remediate the issue. We appreciate responsible disclosure and may credit reporters in release notes (with permission).

Security best practices for contributors

- Do not commit secrets to the repository (use environment variables or a secrets manager).
- Run unit tests and static checks locally before pushing.
- Validate and sanitize all inputs, especially uploaded files.
