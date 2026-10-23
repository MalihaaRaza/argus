"""The /ws/runs/{run_id} WebSocket, against argus_test and a real Redis.

Real pub/sub on purpose: the endpoint's guarantees (no gaps, no duplicates) depend
on when Redis delivers messages, which a fake would only imitate. Channels are
named after random run ids, so tests never hear each other or a real worker.
"""

import asyncio
import socket
import threading
import time
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import pytest
import redis
import uvicorn
from api_overrides import override_redis, override_session
from fastapi.testclient import TestClient
from sqlalchemy import Engine, update
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from starlette.websockets import WebSocketDisconnect
from websockets.sync.client import connect

import argus.api
from argus import events
from argus.api import app
from argus.config import get_settings
from argus.db import Run, RunStep
from argus.schemas import RunDetailOut, RunStepOut

pytestmark = pytest.mark.db


@pytest.fixture(scope="session")
def redis_url() -> str:
    url = get_settings().redis_url
    try:
        redis.Redis.from_url(url, socket_connect_timeout=2).ping()
    except (redis.RedisError, OSError):
        pytest.skip("Redis is not reachable")
    return url


@pytest.fixture
def client(test_url: URL, redis_url: str, clean_db: None) -> Iterator[TestClient]:
    override_session(test_url)
    override_redis(redis_url)
    yield TestClient(app)
    app.dependency_overrides.clear()


def add_run(engine: Engine, status: str, steps: int = 0) -> uuid.UUID:
    with Session(engine) as session:
        run = Run(url="https://shop.test", goal="Log in", status=status)
        session.add(run)
        session.commit()
        for number in range(1, steps + 1):
            save_step(engine, run.id, number, publish=False)
        return run.id


def save_step(engine: Engine, run_id: uuid.UUID, number: int, publish: bool) -> None:
    """Save a step the way the worker does: commit first, then publish its event."""
    with Session(engine) as session:
        step = RunStep(
            run_id=run_id,
            step_number=number,
            tool_name="click",
            tool_input={"ref": f"e{number}"},
            result="ok",
            is_error=False,
            created_at=datetime.now(UTC),
        )
        session.add(step)
        session.commit()
        if publish:
            events.publish(events.step_event(run_id, RunStepOut.model_validate(step)))


def finish_data(reason: str = "Logged in") -> events.FinishedData:
    return events.FinishedData(
        status="passed",
        reason=reason,
        step_count=3,
        input_tokens=300,
        output_tokens=60,
        finished_at=datetime.now(UTC),
    )


def finish_run(engine: Engine, run_id: uuid.UUID) -> None:
    """Record the final result, then publish the finished event (worker order)."""
    data = finish_data()
    with Session(engine) as session:
        session.execute(update(Run).where(Run.id == run_id).values(**data.model_dump()))
        session.commit()
    events.publish(events.finished_event(run_id, data))


def receive_until_closed(ws: Any) -> tuple[list[dict[str, Any]], int]:
    """All messages until the server closes the socket, and its close code."""
    messages = []
    with pytest.raises(WebSocketDisconnect) as closed:
        while True:
            messages.append(ws.receive_json())
    return messages, closed.value.code


def test_unknown_run_closes_with_4404(client: TestClient) -> None:
    with client.websocket_connect(f"/ws/runs/{uuid.uuid4()}") as ws:
        messages, code = receive_until_closed(ws)
    assert messages == []
    assert code == 4404


def test_finished_run_gets_snapshot_then_finished(
    client: TestClient, engine: Engine
) -> None:
    run_id = add_run(engine, status="running", steps=2)
    finish_run(engine, run_id)

    with client.websocket_connect(f"/ws/runs/{run_id}") as ws:
        messages, code = receive_until_closed(ws)

    snapshot, finished = messages
    assert snapshot["type"] == "snapshot"
    assert snapshot["run_id"] == str(run_id)
    assert snapshot["data"]["status"] == "passed"
    assert [s["step_number"] for s in snapshot["data"]["steps"]] == [1, 2]
    assert finished["type"] == "finished"
    assert finished["data"]["status"] == "passed"
    assert finished["data"]["reason"] == "Logged in"
    assert code == 1000  # Normal closure


def test_mid_run_join_has_no_gaps_and_no_duplicates(
    client: TestClient, engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = add_run(engine, status="running", steps=1)
    real_load_snapshot = argus.api.load_snapshot

    async def load_snapshot_during_a_run(
        session: AsyncSession, run_id: uuid.UUID
    ) -> RunDetailOut | None:
        # The endpoint has subscribed by now. A step committed at this moment
        # is in the snapshot AND arrives as a live event: a duplicate to drop.
        # (Same for a status the snapshot already shows.)
        save_step(engine, run_id, 2, publish=True)
        events.publish(
            events.status_event(
                run_id, events.StatusData(status="running", started_at=None)
            )
        )
        snapshot = await real_load_snapshot(session, run_id)
        # Committed after the snapshot was read: only the live event has it.
        # Subscribing after reading would lose these (a gap).
        save_step(engine, run_id, 3, publish=True)
        finish_run(engine, run_id)
        return snapshot

    monkeypatch.setattr(argus.api, "load_snapshot", load_snapshot_during_a_run)

    with client.websocket_connect(f"/ws/runs/{run_id}") as ws:
        snapshot = ws.receive_json()
        # Backstop: if the endpoint had missed the real finished event (a gap), it
        # would wait forever and hang this test. This one makes it close instead,
        # so a gap fails the assertions below. A correct endpoint has already
        # forwarded the real one and stopped listening, so it never sends this.
        backstop = finish_data(reason="backstop")
        events.publish(events.finished_event(run_id, backstop))
        live, code = receive_until_closed(ws)

    assert [s["step_number"] for s in snapshot["data"]["steps"]] == [1, 2]
    assert [m["type"] for m in live] == ["step", "finished"]
    assert live[0]["data"]["step_number"] == 3
    assert live[1]["data"]["reason"] == "Logged in"  # The real one
    # Every step exactly once, in order.
    all_steps = snapshot["data"]["steps"] + [live[0]["data"]]
    ids = [s["id"] for s in all_steps]
    assert ids == sorted(set(ids)) and len(ids) == 3
    assert code == 1000


def test_live_events_are_forwarded(client: TestClient, engine: Engine) -> None:
    # The client is already watching when the run starts and progresses.
    run_id = add_run(engine, status="queued")

    with client.websocket_connect(f"/ws/runs/{run_id}") as ws:
        snapshot = ws.receive_json()
        events.publish(
            events.status_event(
                run_id, events.StatusData(status="running", started_at=None)
            )
        )
        save_step(engine, run_id, 1, publish=True)
        finish_run(engine, run_id)
        live, code = receive_until_closed(ws)

    assert snapshot["data"]["status"] == "queued"
    assert [m["type"] for m in live] == ["status", "step", "finished"]
    assert live[0]["data"]["status"] == "running"
    assert code == 1000


@pytest.fixture
def live_server(client: TestClient) -> Iterator[str]:
    """The app on a real uvicorn server in a background thread; yields its ws:// URL.

    Needed for disconnect tests: TestClient cancels the endpoint when the test's
    socket closes, so cleanup would run even if the endpoint never noticed the
    disconnect. A real server only *tells* the endpoint, like in production.
    (Depends on `client` for its dependency overrides.)
    """
    with socket.socket() as sock:  # Ask the OS for a free port
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    # lifespan="off": the app's shutdown would close the real (non-test) clients.
    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, lifespan="off", log_level="warning"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started:
        assert time.monotonic() < deadline, "uvicorn did not start"
        time.sleep(0.01)
    yield f"ws://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


def test_disconnect_unsubscribes(
    live_server: str, engine: Engine, redis_url: str
) -> None:
    run_id = add_run(engine, status="running")  # Running: the endpoint keeps listening
    channel = events.channel(run_id)
    check = redis.Redis.from_url(redis_url)

    def subscribers() -> int:
        [(_, count)] = check.pubsub_numsub(channel)
        return count

    with connect(f"{live_server}/ws/runs/{run_id}") as ws:
        ws.recv()  # The snapshot: by now the endpoint is subscribed
        assert subscribers() == 1
    # Leaving the block closes the socket from the client side.

    deadline = time.monotonic() + 3
    while subscribers() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert subscribers() == 0


class LateConfirmationPubSub:
    """A pub/sub whose subscription only takes effect when Redis's reply arrives."""

    def __init__(self) -> None:
        self.active = False

    async def subscribe(self, channel: str) -> None:
        pass  # The command is sent, but Redis hasn't processed it yet

    async def get_message(self, timeout: float | None) -> dict[str, Any]:
        await asyncio.sleep(0.05)  # Redis takes a moment to process it
        self.active = True
        return {"type": "subscribe", "channel": b"run:x", "data": 1}


@pytest.mark.anyio
async def test_subscribe_waits_for_confirmation(anyio_backend: str) -> None:
    # Returning before Redis confirms would let the snapshot be read while the
    # subscription isn't active yet, and events published then would be lost.
    pubsub = LateConfirmationPubSub()
    await argus.api._subscribe(pubsub, "run:x")  # type: ignore[arg-type]
    assert pubsub.active


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
