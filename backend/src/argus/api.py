"""HTTP API for creating, reading and watching runs.

Each new run is saved as "queued" and handed to the Celery worker (worker.py),
which executes it and writes the progress back to the database. The WebSocket
endpoint streams that progress live, from the events the worker publishes.
"""

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated
from uuid import UUID

import redis.asyncio as aioredis
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import JSONResponse
from redis.asyncio.client import PubSub
from redis.exceptions import RedisError
from sqlalchemy import func, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from argus import events
from argus.celery_app import EXECUTE_RUN_TASK, celery_app
from argus.config import get_settings
from argus.db import Run, get_async_engine, get_async_sessionmaker
from argus.schemas import ErrorOut, HealthOut, RunCreate, RunDetailOut, RunOut

HEALTH_TIMEOUT_SECONDS = 3
SUBSCRIBE_TIMEOUT_SECONDS = 5
NOT_FOUND_CLOSE_CODE = 4404  # 4000-4999 is free for apps; 4404 mirrors HTTP 404

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    # On shutdown, close pooled database and Redis connections cleanly.
    await get_async_engine().dispose()
    await _redis_client().aclose()


app = FastAPI(title="Argus", lifespan=lifespan)


async def get_session() -> AsyncIterator[AsyncSession]:
    """One session per request, closed when the request ends.

    Endpoints receive it through Depends(), which lets tests swap in a session
    for the test database via app.dependency_overrides.
    """
    async with get_async_sessionmaker()() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_session)]


@lru_cache
def _redis_client() -> aioredis.Redis:
    # One client (and connection pool) for the whole app, created on first use.
    return aioredis.Redis.from_url(get_settings().redis_url)


async def get_redis() -> aioredis.Redis:
    """The async Redis client, as a dependency so tests can swap it."""
    return _redis_client()


RedisDep = Annotated[aioredis.Redis, Depends(get_redis)]


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


# `responses` only documents extra outcomes in the OpenAPI schema (and /docs);
# FastAPI doesn't validate them. The 200 response comes from the return type.
@app.get(
    "/runs/{run_id}",
    responses={404: {"model": ErrorOut, "description": "No run with this id"}},
)
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


# response_model is set explicitly because the return type also allows a
# JSONResponse (the 503), which isn't a schema FastAPI could document.
@app.get(
    "/health",
    response_model=HealthOut,
    responses={503: {"model": HealthOut, "description": "The database is unreachable"}},
)
async def health(session: SessionDep) -> HealthOut | JSONResponse:
    try:
        async with asyncio.timeout(HEALTH_TIMEOUT_SECONDS):
            await session.execute(text("SELECT 1"))
    except (SQLAlchemyError, OSError, TimeoutError):
        # Report the outage as a normal response, not a crash: load balancers and
        # monitors read the 503 status code.
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=HealthOut(status="error", database="unreachable").model_dump(),
        )
    return HealthOut(status="ok", database="ok")


async def load_snapshot(session: AsyncSession, run_id: UUID) -> RunDetailOut | None:
    """The run and its steps as stored right now, or None if there is no such run."""
    run = await session.get(Run, run_id, options=[selectinload(Run.steps)])
    return None if run is None else RunDetailOut.model_validate(run)


@app.websocket("/ws/runs/{run_id}")
async def watch_run(
    websocket: WebSocket, run_id: UUID, session: SessionDep, redis_client: RedisDep
) -> None:
    """Stream a run: a snapshot of everything so far, then live events.

    Order matters. We subscribe to the run's channel *before* reading the
    snapshot, so every change committed after the read has its event delivered
    to us (the worker publishes after committing). Changes committed between
    subscribing and reading show up twice, in the snapshot and as an event, and
    are dropped from the live stream. Result: no gaps and no duplicates.
    """
    # Accept first: closing with our own code (4404) needs an open connection.
    # Closing before accepting would turn into a plain HTTP 403 instead.
    await websocket.accept()
    pubsub = redis_client.pubsub()
    try:
        await _subscribe(pubsub, events.channel(run_id))
        snapshot = await load_snapshot(session, run_id)
        # Release the database connection now: the socket may stay open for
        # minutes, and an idle open transaction would hold it all that time.
        await session.close()

        if snapshot is None:
            await websocket.close(code=NOT_FOUND_CLOSE_CODE, reason="Run not found")
            return
        message = events.SnapshotMessage(run_id=run_id, data=snapshot)
        await websocket.send_text(message.model_dump_json())

        if snapshot.status in events.FINISHED_STATUSES:
            # Nothing more will happen, so say so and close.
            data = events.FinishedData.model_validate(snapshot, from_attributes=True)
            await websocket.send_text(
                events.finished_event(run_id, data).model_dump_json()
            )
            await websocket.close()
            return

        if await _forward_until_finished(websocket, pubsub, snapshot):
            await websocket.close()
    except WebSocketDisconnect:
        pass  # The client left while we were sending; cleanup below covers it.
    finally:
        # Always runs, however the connection ended. Errors are ignored here: if
        # Redis is unreachable, closing the connection drops the subscription anyway.
        with contextlib.suppress(RedisError, OSError):
            await pubsub.unsubscribe()
        await pubsub.aclose()


async def _subscribe(pubsub: PubSub, channel: str) -> None:
    """Subscribe, and return only once Redis has confirmed the subscription.

    subscribe() just sends the command. Until Redis has processed it, events
    published on other connections (the worker's) would not reach us, so the
    "no gaps" guarantee depends on waiting for the confirmation.
    """
    await pubsub.subscribe(channel)
    async with asyncio.timeout(SUBSCRIBE_TIMEOUT_SECONDS):
        while True:
            reply = await pubsub.get_message(timeout=None)
            if reply is not None and reply["type"] == "subscribe":
                return


async def _forward_until_finished(
    websocket: WebSocket, pubsub: PubSub, snapshot: RunDetailOut
) -> bool:
    """Forward live events until the run finishes (True) or the client leaves (False).

    Both are awaited at once: without also listening to the client, a watcher
    who closed the tab would keep its subscription until the run's next event,
    which for a stuck run might never come.
    """
    forward = asyncio.create_task(_forward_events(websocket, pubsub, snapshot))
    disconnect = asyncio.create_task(_wait_for_disconnect(websocket))
    done, pending = await asyncio.wait(
        {forward, disconnect}, return_when=asyncio.FIRST_COMPLETED
    )
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    if forward in done:
        forward.result()  # Re-raises if forwarding failed
        return True
    return False


async def _forward_events(
    websocket: WebSocket, pubsub: PubSub, snapshot: RunDetailOut
) -> None:
    """Send each new live event to the client; return after the finished event."""
    # Step ids only increase, so anything at or below the highest id already
    # sent was in the snapshot. Same idea for a status the client already has.
    last_step_id = max((step.id for step in snapshot.steps), default=0)
    last_status = snapshot.status
    async for message in pubsub.listen():
        if message["type"] != "message":
            continue  # E.g. subscription confirmations
        event = events.RunEvent.model_validate_json(message["data"])
        if event.type == "step":
            if event.data["id"] <= last_step_id:
                continue
            last_step_id = event.data["id"]
        elif event.type == "status":
            if event.data["status"] == last_status:
                continue
            last_status = event.data["status"]
        await websocket.send_text(event.model_dump_json())
        if event.type == "finished":
            return


async def _wait_for_disconnect(websocket: WebSocket) -> None:
    # Clients don't send anything we use; we only read to notice when they leave.
    while (await websocket.receive())["type"] != "websocket.disconnect":
        pass
