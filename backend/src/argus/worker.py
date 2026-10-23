"""Celery worker: executes the runs the API queues.

Redis (the broker) only carries the run id. The run, its steps and its result live
in Postgres, so there is no Celery result backend. After each change is committed,
an event is published on the run's Redis channel (events.py) for live watchers.

The Celery app itself is defined in celery_app.py; this module adds the task to it.
Start it from backend/ (prefork is the only pool that enforces the time limits;
--max-tasks-per-child restarts the process now and then, which frees memory that
Chromium and Playwright leave behind):
  uv run celery -A argus.worker:celery_app worker --pool=prefork --concurrency=1 \
    --max-tasks-per-child=20
"""

import uuid
from datetime import UTC, datetime
from typing import Any, NamedTuple

from celery.exceptions import SoftTimeLimitExceeded
from celery.utils.log import get_task_logger
from sqlalchemy import update
from sqlalchemy.orm import Session

from argus.agent import run_agent
from argus.celery_app import EXECUTE_RUN_TASK, celery_app
from argus.db import Run, RunStep, get_sessionmaker
from argus.events import (
    FinishedData,
    StatusData,
    finished_event,
    publish,
    status_event,
    step_event,
)
from argus.schemas import RunStepOut

logger = get_task_logger(__name__)

# At the soft limit, Celery raises SoftTimeLimitExceeded inside the task, which
# still has time to record the error. At the hard limit it kills the process.
SOFT_TIME_LIMIT_SECONDS = 280
TIME_LIMIT_SECONDS = 300


@celery_app.task(
    name=EXECUTE_RUN_TASK,
    soft_time_limit=SOFT_TIME_LIMIT_SECONDS,
    time_limit=TIME_LIMIT_SECONDS,
)
def execute_run(run_id: str) -> None:
    """Run the agent for one queued run, saving progress to the database as it goes.

    Never leaves the run in "running": every way out of the agent ends with a
    final status. The exception is a process that is killed (the hard time limit,
    or running out of memory), which no code in the task can react to.
    """
    try:
        run_uuid = uuid.UUID(run_id)
    except ValueError:
        logger.error("Invalid run id %r", run_id)
        return

    with get_sessionmaker()() as session:
        claimed = _claim(session, run_uuid)
        if claimed is None:
            # Unknown id, or another delivery of this task already took the run.
            logger.info("Run %s is not queued; skipping", run_id)
            return
        try:
            # Inside the try, like everything after the claim: if anything here
            # raises, the except blocks below still give the run a final status.
            publish(
                status_event(
                    run_uuid,
                    StatusData(status="running", started_at=claimed.started_at),
                )
            )
            _execute(session, run_uuid, claimed.url, claimed.goal)
        except SoftTimeLimitExceeded:
            logger.error("Run %s hit the time limit", run_id)
            _mark_error(session, run_uuid, "time limit exceeded")
        except Exception as error:
            logger.exception("Run %s crashed", run_id)
            _mark_error(session, run_uuid, f"unexpected error: {type(error).__name__}")


def _now() -> datetime:
    """The current time, for recording when something happened.

    Postgres's now() is the time the *transaction* started, not the time of the
    statement, so it is wrong for events inside a long transaction. Event times
    are therefore taken here, in Python, at the moment they happen.
    """
    return datetime.now(UTC)


class Claim(NamedTuple):
    url: str
    goal: str
    started_at: datetime


def _claim(session: Session, run_id: uuid.UUID) -> Claim | None:
    """Move the run from "queued" to "running" and return what the agent needs.

    Returns None if the run wasn't queued. Checking and setting happen in one
    UPDATE, so the database guarantees only one caller can win. Reading the status
    first and then writing it would let two deliveries of the same task both see
    "queued" and both run it. RETURNING hands back what the agent needs in the same
    statement, so no separate read (and no transaction) is left open afterwards.
    """
    started_at = _now()
    row = session.execute(
        update(Run)
        .where(Run.id == run_id, Run.status == "queued")
        .values(status="running", started_at=started_at)
        .returning(Run.url, Run.goal)
    ).one_or_none()
    session.commit()
    return None if row is None else Claim(row.url, row.goal, started_at)


def _execute(session: Session, run_id: uuid.UUID, url: str, goal: str) -> None:
    # Every database write below is its own short transaction that ends with a
    # commit. Nothing stays open while the browser launches or Claude is thinking,
    # which can take many seconds: an open transaction would hold a connection
    # "idle in transaction" and pin now() to a stale time.
    # Each event is published only after its commit, so anyone told about a
    # change can also read it from the database.

    def save_step(
        step: int,
        tool_name: str,
        tool_input: dict[str, Any],
        result: str,
        is_error: bool,
    ) -> None:
        # Committing each step means GET /runs/{id} shows progress while the run is
        # still going, and the steps survive if the run crashes later.
        run_step = RunStep(
            run_id=run_id,
            step_number=step,
            tool_name=tool_name,
            tool_input=tool_input,  # Already masked by the agent
            result=result,
            is_error=is_error,
            created_at=_now(),  # When the step happened, not when a transaction began
        )
        session.add(run_step)
        # flush sends the INSERT, which gives the step its id. The event is built
        # now, before commit: afterwards the object is expired, and reading it
        # again would start a new transaction that stays open until the next step.
        session.flush()
        event = step_event(run_id, RunStepOut.model_validate(run_step))
        session.execute(update(Run).where(Run.id == run_id).values(step_count=step))
        session.commit()
        publish(event)

    result = run_agent(url, goal, on_step=save_step)

    finished = FinishedData(
        status=result.status,
        reason=result.reason,
        step_count=result.steps,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        finished_at=_now(),
    )
    session.execute(update(Run).where(Run.id == run_id).values(**finished.model_dump()))
    session.commit()
    publish(finished_event(run_id, finished))


def _mark_error(session: Session, run_id: uuid.UUID, reason: str) -> None:
    # If the error came from a failed commit, the session refuses all further
    # work until it is rolled back. Steps that were already committed are kept.
    session.rollback()
    finished_at = _now()
    # RETURNING fills in the counts the run had reached, for the finished event.
    row = session.execute(
        update(Run)
        .where(Run.id == run_id)
        .values(status="error", reason=reason, finished_at=finished_at)
        .returning(Run.step_count, Run.input_tokens, Run.output_tokens)
    ).one()
    session.commit()
    finished = FinishedData(
        status="error",
        reason=reason,
        step_count=row.step_count,
        input_tokens=row.input_tokens,
        output_tokens=row.output_tokens,
        finished_at=finished_at,
    )
    publish(finished_event(run_id, finished))
