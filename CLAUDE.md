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
- Apply database migrations: `uv run alembic upgrade head`
- Create a migration after changing models: `uv run alembic revision --autogenerate -m "..."`
- Run the agent: `uv run argus "<url>" "<goal>"`
- Run the API in dev mode (auto-reload; docs at http://127.0.0.1:8000/docs):
  `uv run fastapi dev src/argus/api.py`
- Run the worker that executes queued runs (needs Redis; prefork is the only pool
  that enforces the task time limits; --max-tasks-per-child restarts the process
  every 20 runs to free memory left behind by Chromium):
  `uv run celery -A argus.worker:celery_app worker --pool=prefork --concurrency=1 --max-tasks-per-child=20 --loglevel=info`
- Watch a run live in the terminal (API must be running):
  `uv run python scripts/watch_run.py <run_id>`
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