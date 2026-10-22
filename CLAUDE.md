# Argus — AI QA Agent

Argus is an AI agent that tests web apps from plain-English goals. It drives a real
browser with Playwright, decides each action with Claude via tool use, and reports
pass/fail with evidence.

## About the developer
I'm building this project to learn. Explain the reasoning behind design decisions
and any non-obvious code. Prefer clear, simple code over clever code.

## Repository layout
- `backend/` — Python package (`src/argus/`): agent loop, browser tools, API (later)
- `frontend/` — React + TypeScript dashboard (added later)
- `docs/decisions/` — short records of design decisions

## Commands (run from `backend/`)
- Start Postgres and Redis (from repo root): `docker compose up -d`
- Install dependencies: `uv sync`
- Run the agent: `uv run argus "<url>" "<goal>"`
- Tests: `uv run pytest`
- Lint and format: `uv run ruff check . && uv run ruff format .`

## Conventions
- Python 3.12; type hints on all functions; Pydantic models for structured data
- Call the Anthropic SDK directly (no LangChain or agent frameworks)
- Every agent loop has a maximum step limit
- Tool errors are returned to the model as data, never raised to crash the loop
- Secrets live in `.env` (never committed); document new variables in `.env.example`
- Add or update tests for every behavior change

## Don'ts
- Don't add new dependencies without asking first
- Don't read or edit `.env`
- Don't commit or push; I review and commit changes myself