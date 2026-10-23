"""Live run events, sent from the worker to the API through Redis pub/sub.

The worker publishes an event after each change it commits; the API's WebSocket
endpoint forwards them to browsers. Pub/sub doesn't store anything: subscribers
only get events published while they're listening. The database stays the source
of truth, which is why a new watcher first gets a snapshot loaded from it.
"""

import logging
from datetime import datetime
from functools import lru_cache
from typing import Any, Literal
from uuid import UUID

import redis
from pydantic import BaseModel

from argus.config import get_settings
from argus.db import RunStatus
from argus.schemas import RunDetailOut, RunStepOut

logger = logging.getLogger(__name__)

EventType = Literal["status", "step", "finished"]
FINISHED_STATUSES: frozenset[str] = frozenset({"passed", "failed", "error"})

# Keep a stalled Redis from stalling a run: publishing gives up after this long.
PUBLISH_TIMEOUT_SECONDS = 2


class RunEvent(BaseModel):
    """One message on a run's channel. `data` depends on `type` (see below)."""

    type: EventType
    run_id: UUID
    data: dict[str, Any]


class StatusData(BaseModel):
    """data of a "status" event: the run changed status (e.g. it started)."""

    status: RunStatus
    started_at: datetime | None


class FinishedData(BaseModel):
    """data of a "finished" event: the run's final result."""

    status: Literal["passed", "failed", "error"]
    reason: str | None
    step_count: int
    input_tokens: int
    output_tokens: int
    finished_at: datetime | None


class SnapshotMessage(BaseModel):
    """Not an event: the first WebSocket message, with the run as stored so far."""

    type: Literal["snapshot"] = "snapshot"
    run_id: UUID
    data: RunDetailOut


def channel(run_id: UUID) -> str:
    return f"run:{run_id}"


def status_event(run_id: UUID, data: StatusData) -> RunEvent:
    return RunEvent(type="status", run_id=run_id, data=data.model_dump(mode="json"))


def step_event(run_id: UUID, step: RunStepOut) -> RunEvent:
    # The step's database id lets the API drop events it already sent in a
    # snapshot (ids only grow, so "id <= highest sent" means "already sent").
    return RunEvent(type="step", run_id=run_id, data=step.model_dump(mode="json"))


def finished_event(run_id: UUID, data: FinishedData) -> RunEvent:
    return RunEvent(type="finished", run_id=run_id, data=data.model_dump(mode="json"))


@lru_cache
def get_redis() -> redis.Redis:
    """The worker's Redis client, created on first use (like the engines in db.py)."""
    return redis.Redis.from_url(
        get_settings().redis_url,
        socket_timeout=PUBLISH_TIMEOUT_SECONDS,
        socket_connect_timeout=PUBLISH_TIMEOUT_SECONDS,
    )


def publish(event: RunEvent) -> None:
    """Publish `event` to its run's channel. Best effort: never raises for Redis
    problems, because live updates are a convenience and the run must go on.

    Catches only Redis, network and URL errors, not every Exception: a broad
    catch would also swallow Celery's SoftTimeLimitExceeded and let a run blow
    through its time limit.
    """
    try:
        get_redis().publish(channel(event.run_id), event.model_dump_json())
    except (redis.RedisError, OSError, ValueError) as error:
        logger.warning(
            "Could not publish %s event for run %s: %s", event.type, event.run_id, error
        )
