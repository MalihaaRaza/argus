"""The execute_run task, called directly (no Redis or worker process) against argus_test.

A real worker would send the task through Redis. Calling the task function directly
runs the same code synchronously, which is all these tests need.
"""

import uuid
from types import SimpleNamespace
from typing import Any

import pytest
from celery.exceptions import SoftTimeLimitExceeded
from fakes import FakeBrowser, FakeClient, tool_use
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

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
