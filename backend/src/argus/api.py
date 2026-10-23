"""HTTP API for creating and reading runs.

Each new run is saved as "queued" and handed to the Celery worker (worker.py),
which executes it and writes the progress back to the database.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.responses import JSONResponse
from sqlalchemy import func, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from argus.celery_app import EXECUTE_RUN_TASK, celery_app
from argus.db import Run, get_async_engine, get_async_sessionmaker
from argus.schemas import RunCreate, RunDetailOut, RunOut

HEALTH_TIMEOUT_SECONDS = 3

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    # On shutdown, close pooled database connections cleanly.
    await get_async_engine().dispose()


app = FastAPI(title="Argus", lifespan=lifespan)


async def get_session() -> AsyncIterator[AsyncSession]:
    """One session per request, closed when the request ends.

    Endpoints receive it through Depends(), which lets tests swap in a session
    for the test database via app.dependency_overrides.
    """
    async with get_async_sessionmaker()() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_session)]


# 202 Accepted: the run is recorded now and will be executed later by a worker.
@app.post("/runs", status_code=status.HTTP_202_ACCEPTED)
async def create_run(body: RunCreate, session: SessionDep) -> RunOut:
    run = Run(url=str(body.url), goal=body.goal)
    session.add(run)
    # Commit before enqueueing: a fast worker must find the row when it looks.
    await session.commit()

    try:
        # Sending by name means the API never imports worker.py (and with it the
        # agent, Playwright and the Anthropic SDK); it only publishes a message.
        # send_task does blocking network I/O (and retries if Redis is down), so it
        # runs in a thread to keep the event loop free for other requests.
        await asyncio.to_thread(
            celery_app.send_task, EXECUTE_RUN_TASK, args=[str(run.id)]
        )
    except Exception:
        # Deliberately broad: whatever stopped the enqueue (Redis down, bad
        # REDIS_URL, ...), the run must not sit in "queued" forever.
        logger.exception("Could not queue run %s", run.id)
        run.status = "error"
        run.reason = "could not queue run"
        run.finished_at = func.now()
        await session.commit()

    # created_at (and finished_at, if set above) are filled in by Postgres. Reading
    # them would trigger a lazy load, which async SQLAlchemy forbids (it would be
    # hidden I/O), so fetch them explicitly.
    await session.refresh(run)
    return RunOut.model_validate(run)


@app.get("/runs/{run_id}")
async def get_run(run_id: UUID, session: SessionDep) -> RunDetailOut:
    # selectinload fetches all steps in one extra query, up front, instead of
    # lazily on first access (which async sessions don't allow).
    run = await session.get(Run, run_id, options=[selectinload(Run.steps)])
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return RunDetailOut.model_validate(run)


@app.get("/runs")
async def list_runs(
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[RunOut]:
    # Newest first. id breaks ties between equal timestamps, so pages don't
    # overlap or skip rows when paginating.
    query = (
        select(Run)
        .order_by(Run.created_at.desc(), Run.id.desc())
        .limit(limit)
        .offset(offset)
    )
    runs = await session.scalars(query)
    return [RunOut.model_validate(run) for run in runs]


@app.get("/health", response_model=None)
async def health(session: SessionDep) -> dict[str, str] | JSONResponse:
    try:
        async with asyncio.timeout(HEALTH_TIMEOUT_SECONDS):
            await session.execute(text("SELECT 1"))
    except (SQLAlchemyError, OSError, TimeoutError):
        # Report the outage as a normal response, not a crash: load balancers and
        # monitors read the 503 status code.
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "error", "database": "unreachable"},
        )
    return {"status": "ok", "database": "ok"}
