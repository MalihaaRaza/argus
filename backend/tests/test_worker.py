"""The execute_run task, called directly (no Redis or worker process) against argus_test.

A real worker would send the task through Redis. Calling the task function directly
runs the same code synchronously, which is all these tests need.
"""

import json
import logging
import time
import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest
import redis
from celery.exceptions import SoftTimeLimitExceeded
from fakes import FakeBrowser, FakeClient, FakeRedis, tool_use
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session, sessionmaker

import argus.events
import argus.worker
from argus.agent import StepCallback, _run_loop
from argus.db import Run
from argus.models import RunResult
from argus.worker import execute_run

pytestmark = pytest.mark.db


@pytest.fixture(autouse=True)
def use_test_db(
    engine: Engine, clean_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(argus.worker, "get_sessionmaker", lambda: sessionmaker(engine))


@pytest.fixture(autouse=True)
def fake_redis(monkeypatch: pytest.MonkeyPatch) -> FakeRedis:
    """Every worker test publishes into this fake, never to a real Redis."""
    fake = FakeRedis()
    monkeypatch.setattr(argus.events, "get_redis", lambda: fake)
    return fake


def published_events(fake: FakeRedis, run_id: uuid.UUID) -> list[dict[str, Any]]:
    assert all(channel == f"run:{run_id}" for channel, _ in fake.published)
    return [json.loads(message) for _, message in fake.published]


def scripted_agent(
    monkeypatch: pytest.MonkeyPatch, turns: list[list[SimpleNamespace]]
) -> None:
    """Replace run_agent with the real loop driven by scripted model turns."""

    def fake_run_agent(url: str, goal: str, on_step: StepCallback) -> RunResult:
        return _run_loop(FakeClient(turns), FakeBrowser(), goal, 5, on_step)

    monkeypatch.setattr(argus.worker, "run_agent", fake_run_agent)


def failing_agent(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    """Replace run_agent with one that saves a step, then raises `error`."""

    def fake_run_agent(url: str, goal: str, on_step: StepCallback) -> RunResult:
        on_step(1, "click", {"ref": "e1"}, "Clicked e1.", False)
        raise error

    monkeypatch.setattr(argus.worker, "run_agent", fake_run_agent)


def add_run(engine: Engine, status: str = "queued") -> uuid.UUID:
    with Session(engine) as session:
        run = Run(url="https://shop.test", goal="Log in", status=status)
        session.add(run)
        session.commit()
        return run.id


def load_run(engine: Engine, run_id: uuid.UUID) -> Run:
    # expire_on_commit=False keeps the loaded values readable after the session closes.
    with Session(engine, expire_on_commit=False) as session:
        run = session.get_one(Run, run_id)
        _ = run.steps  # Load the steps while the session is still open
        return run


LOGIN_TURNS = [
    [
        tool_use("a", "type_text", ref="e4", text="standard_user"),
        tool_use("b", "type_text", ref="e5", text="hunter2"),  # e5 is a password
        tool_use("c", "click", ref="e1"),
    ],
    [tool_use("d", "finish", passed=True, reason="Logged in")],
]


def test_successful_run_is_saved(
    engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripted_agent(monkeypatch, LOGIN_TURNS)
    run_id = add_run(engine)

    execute_run(str(run_id))

    run = load_run(engine, run_id)
    assert run.status == "passed"
    assert run.reason == "Logged in"
    assert run.step_count == 2  # Model calls, not tool calls
    assert run.started_at is not None
    assert run.finished_at is not None
    assert run.started_at <= run.finished_at
    assert [(s.step_number, s.tool_name) for s in run.steps] == [
        (1, "type_text"),
        (1, "type_text"),
        (1, "click"),
        (2, "finish"),
    ]
    assert run.steps[2].result == "Clicked e1."
    assert run.steps[3].tool_input == {"passed": True, "reason": "Logged in"}


def test_password_is_masked_in_saved_steps(
    engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripted_agent(monkeypatch, LOGIN_TURNS)
    run_id = add_run(engine)

    execute_run(str(run_id))

    inputs = [s.tool_input for s in load_run(engine, run_id).steps]
    assert inputs[0] == {"ref": "e4", "text": "standard_user"}  # Not a password
    assert inputs[1] == {"ref": "e5", "text": "••••"}
    assert "hunter2" not in str(inputs)


def test_token_totals_are_saved(
    engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripted_agent(monkeypatch, LOGIN_TURNS)
    run_id = add_run(engine)

    execute_run(str(run_id))

    run = load_run(engine, run_id)
    assert (run.input_tokens, run.output_tokens) == (200, 40)  # 2 calls of 100/20


def test_failed_verdict_is_saved(
    engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripted_agent(
        monkeypatch, [[tool_use("a", "finish", passed=False, reason="No cart")]]
    )
    run_id = add_run(engine)

    execute_run(str(run_id))

    run = load_run(engine, run_id)
    assert (run.status, run.reason, run.step_count) == ("failed", "No cart", 1)


def test_unexpected_exception_marks_run_error(
    engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    failing_agent(monkeypatch, RuntimeError("boom"))
    run_id = add_run(engine)

    execute_run(str(run_id))  # Must not raise

    run = load_run(engine, run_id)
    assert run.status == "error"
    assert run.reason == "unexpected error: RuntimeError"
    assert run.finished_at is not None
    # The step saved before the crash is kept.
    assert [s.tool_name for s in run.steps] == ["click"]
    assert run.step_count == 1


def test_soft_time_limit_marks_run_error(
    engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Celery raises this inside the task when the soft limit passes.
    failing_agent(monkeypatch, SoftTimeLimitExceeded())
    run_id = add_run(engine)

    execute_run(str(run_id))

    run = load_run(engine, run_id)
    assert (run.status, run.reason) == ("error", "time limit exceeded")
    assert run.finished_at is not None


@pytest.mark.parametrize("status", ["running", "passed", "error"])
def test_run_that_is_not_queued_is_skipped(
    engine: Engine, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    def must_not_run(*args: Any, **kwargs: Any) -> RunResult:
        raise AssertionError("the agent should not run")

    monkeypatch.setattr(argus.worker, "run_agent", must_not_run)
    run_id = add_run(engine, status=status)

    execute_run(str(run_id))

    run = load_run(engine, run_id)
    assert run.status == status  # Unchanged
    assert run.started_at is None
    assert run.steps == []


@pytest.mark.parametrize("run_id", [str(uuid.uuid4()), "not-a-uuid"])
def test_unknown_or_invalid_id_is_ignored(run_id: str) -> None:
    execute_run(run_id)  # Logs and returns; doesn't raise


def test_step_times_are_when_steps_happen(
    engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The first step comes after a slow start (in real runs: launching the browser
    # and waiting for Claude). Steps used to get Postgres's now(), the start time of
    # a transaction opened before that wait, so the first step was dated too early.
    windows: list[tuple[datetime, datetime]] = []

    def slow_agent(url: str, goal: str, on_step: StepCallback) -> RunResult:
        for step in (1, 2):
            time.sleep(0.3)
            before = datetime.now(UTC)
            on_step(step, "click", {"ref": "e1"}, "Clicked e1.", False)
            windows.append((before, datetime.now(UTC)))
        return RunResult(status="passed", reason="ok", steps=2)

    monkeypatch.setattr(argus.worker, "run_agent", slow_agent)
    run_id = add_run(engine)

    execute_run(str(run_id))

    run = load_run(engine, run_id)
    for saved, (before, after) in zip(run.steps, windows, strict=True):
        assert before <= saved.created_at <= after
    assert run.started_at is not None and run.finished_at is not None
    assert run.steps[0].created_at - run.started_at >= timedelta(seconds=0.3)
    assert run.steps[-1].created_at <= run.finished_at


def test_no_transaction_is_held_while_the_agent_runs(
    engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    def idle_transactions() -> int:
        # Other connections to argus_test that sit inside an unfinished transaction.
        with engine.connect() as conn:
            return conn.scalar(
                text(
                    "SELECT count(*) FROM pg_stat_activity"
                    " WHERE datname = current_database()"
                    " AND state LIKE 'idle in transaction%'"
                    " AND pid <> pg_backend_pid()"
                )
            )

    seen: list[int] = []

    def agent(url: str, goal: str, on_step: StepCallback) -> RunResult:
        seen.append(idle_transactions())  # While the browser / Claude would work
        on_step(1, "click", {"ref": "e1"}, "Clicked e1.", False)
        seen.append(idle_transactions())  # Between steps
        return RunResult(status="passed", reason="ok", steps=1)

    monkeypatch.setattr(argus.worker, "run_agent", agent)
    run_id = add_run(engine)

    execute_run(str(run_id))

    assert seen == [0, 0]
    assert load_run(engine, run_id).status == "passed"


def test_events_are_published_after_each_change(
    engine: Engine, monkeypatch: pytest.MonkeyPatch, fake_redis: FakeRedis
) -> None:
    scripted_agent(monkeypatch, LOGIN_TURNS)
    run_id = add_run(engine)

    execute_run(str(run_id))

    sent = published_events(fake_redis, run_id)
    assert [e["type"] for e in sent] == [
        "status",
        "step",
        "step",
        "step",
        "step",
        "finished",
    ]
    assert all(e["run_id"] == str(run_id) for e in sent)
    assert sent[0]["data"]["status"] == "running"

    run = load_run(engine, run_id)
    # Each step event carries the saved step, including its database id.
    assert [e["data"]["id"] for e in sent[1:5]] == [s.id for s in run.steps]
    assert sent[2]["data"]["tool_input"] == {"ref": "e5", "text": "••••"}  # Masked
    assert sent[-1]["data"] == {
        "status": "passed",
        "reason": "Logged in",
        "step_count": 2,
        "input_tokens": 200,
        "output_tokens": 40,
        "finished_at": run.finished_at.isoformat().replace("+00:00", "Z"),
    }


def test_crashed_run_publishes_finished_error(
    engine: Engine, monkeypatch: pytest.MonkeyPatch, fake_redis: FakeRedis
) -> None:
    failing_agent(monkeypatch, RuntimeError("boom"))
    run_id = add_run(engine)

    execute_run(str(run_id))

    sent = published_events(fake_redis, run_id)
    assert [e["type"] for e in sent] == ["status", "step", "finished"]
    assert sent[-1]["data"]["status"] == "error"
    assert sent[-1]["data"]["reason"] == "unexpected error: RuntimeError"
    assert sent[-1]["data"]["step_count"] == 1  # Read back from the run


def test_skipped_run_publishes_nothing(engine: Engine, fake_redis: FakeRedis) -> None:
    run_id = add_run(engine, status="passed")
    execute_run(str(run_id))
    assert fake_redis.published == []


def test_publish_failure_does_not_break_the_run(
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
    fake_redis: FakeRedis,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_redis.error = redis.ConnectionError("Redis is down")
    scripted_agent(monkeypatch, LOGIN_TURNS)
    run_id = add_run(engine)

    execute_run(str(run_id))

    run = load_run(engine, run_id)
    assert run.status == "passed"
    assert len(run.steps) == 4  # Everything was still saved
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 6  # One per event that couldn't be sent
    assert "Could not publish status event" in warnings[0].getMessage()


def test_publish_does_not_swallow_the_soft_time_limit(
    engine: Engine, monkeypatch: pytest.MonkeyPatch, fake_redis: FakeRedis
) -> None:
    # If the limit fires while publishing, the run must still stop and be marked.
    # Celery raises it once, so only the first publish (the status event) fails.
    real_publish = fake_redis.publish

    def publish_hit_by_time_limit(channel: str, message: str) -> int:
        monkeypatch.setattr(fake_redis, "publish", real_publish)
        raise SoftTimeLimitExceeded()

    monkeypatch.setattr(fake_redis, "publish", publish_hit_by_time_limit)
    scripted_agent(monkeypatch, LOGIN_TURNS)
    run_id = add_run(engine)

    execute_run(str(run_id))

    run = load_run(engine, run_id)
    assert (run.status, run.reason) == ("error", "time limit exceeded")
    assert run.steps == []  # The agent never started
    finished = published_events(fake_redis, run_id)[-1]
    assert (finished["type"], finished["data"]["reason"]) == (
        "finished",
        "time limit exceeded",
    )
