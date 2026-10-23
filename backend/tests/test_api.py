"""The HTTP API, called in-process against the argus_test database.

ASGITransport hands requests straight to the app, with no server or port involved.
"""

import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest
from api_overrides import override_session
from httpx import ASGITransport, AsyncClient
from sqlalchemy import Engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session

import argus.api
from argus.api import app
from argus.celery_app import EXECUTE_RUN_TASK
from argus.db import RunStep

pytestmark = [pytest.mark.db, pytest.mark.anyio]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class FakeSendTask:
    """Stands in for celery_app.send_task: records each (task name, args) instead of
    sending it to Redis. Set `error` to make it raise that error instead."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, list[Any]]] = []
        self.error: Exception | None = None

    def __call__(self, name: str, args: list[Any]) -> None:
        if self.error is not None:
            raise self.error
        self.sent.append((name, args))


@pytest.fixture
def fake_send_task(monkeypatch: pytest.MonkeyPatch) -> FakeSendTask:
    send_task = FakeSendTask()
    monkeypatch.setattr(argus.api.celery_app, "send_task", send_task)
    return send_task


@pytest.fixture
async def client(
    test_url: URL, clean_db: None, fake_send_task: FakeSendTask
) -> AsyncIterator[AsyncClient]:
    # fake_send_task is requested here so no test can accidentally queue a real task.
    override_session(test_url)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()


async def create_run(client: AsyncClient, goal: str = "Add a backpack") -> Any:
    response = await client.post(
        "/runs", json={"url": "https://shop.test", "goal": goal}
    )
    assert response.status_code == 202
    return response.json()


async def test_create_run(client: AsyncClient) -> None:
    response = await client.post(
        "/runs", json={"url": "https://shop.test", "goal": "  Add a backpack  "}
    )

    assert response.status_code == 202
    run = response.json()
    uuid.UUID(run["id"])  # Raises if it isn't a valid UUID
    assert run["status"] == "queued"
    assert run["url"] == "https://shop.test/"  # HttpUrl normalizes the URL
    assert run["goal"] == "Add a backpack"  # Stripped
    assert (run["step_count"], run["input_tokens"], run["output_tokens"]) == (0, 0, 0)
    assert run["reason"] is None
    assert run["created_at"] is not None
    assert run["started_at"] is None
    assert run["finished_at"] is None


async def test_create_run_queues_task(
    client: AsyncClient, fake_send_task: FakeSendTask
) -> None:
    run = await create_run(client)
    assert fake_send_task.sent == [(EXECUTE_RUN_TASK, [run["id"]])]


async def test_create_run_when_queueing_fails(
    client: AsyncClient, fake_send_task: FakeSendTask
) -> None:
    fake_send_task.error = ConnectionError("Redis is down")

    run = await create_run(client)  # Still 202: the run was recorded

    assert run["status"] == "error"
    assert run["reason"] == "could not queue run"
    assert run["finished_at"] is not None

    # The error was saved, not just returned.
    response = await client.get(f"/runs/{run['id']}")
    assert response.json()["status"] == "error"


async def test_get_run_with_steps_in_order(client: AsyncClient, engine: Engine) -> None:
    run = await create_run(client)
    # No endpoint adds steps yet (the worker will), so insert them directly.
    with Session(engine) as session:
        for number, tool in enumerate(["click", "type_text", "finish"], start=1):
            session.add(
                RunStep(
                    run_id=uuid.UUID(run["id"]),
                    step_number=number,
                    tool_name=tool,
                    tool_input={"ref": f"e{number}"},
                    result="ok",
                    is_error=False,
                )
            )
        session.commit()

    response = await client.get(f"/runs/{run['id']}")

    assert response.status_code == 200
    detail = response.json()
    assert detail["id"] == run["id"]
    assert [s["step_number"] for s in detail["steps"]] == [1, 2, 3]
    assert [s["tool_name"] for s in detail["steps"]] == ["click", "type_text", "finish"]
    assert detail["steps"][1]["tool_input"] == {"ref": "e2"}
    assert detail["steps"][0]["is_error"] is False


async def test_get_unknown_run_is_404(client: AsyncClient) -> None:
    response = await client.get(f"/runs/{uuid.uuid4()}")
    assert response.status_code == 404
    assert response.json() == {"detail": "Run not found"}


async def test_malformed_run_id_is_422(client: AsyncClient) -> None:
    response = await client.get("/runs/not-a-uuid")
    assert response.status_code == 422


@pytest.mark.parametrize(
    "body",
    [
        {"url": "ftp://shop.test", "goal": "g"},
        {"url": "not a url", "goal": "g"},
        {"goal": "g"},
    ],
    ids=["ftp-scheme", "not-a-url", "missing"],
)
async def test_invalid_url_is_422(client: AsyncClient, body: dict[str, str]) -> None:
    response = await client.post("/runs", json=body)
    assert response.status_code == 422


@pytest.mark.parametrize(
    "goal", ["", "   ", "x" * 2001], ids=["empty", "blank", "too-long"]
)
async def test_invalid_goal_is_422(client: AsyncClient, goal: str) -> None:
    response = await client.post(
        "/runs", json={"url": "https://shop.test", "goal": goal}
    )
    assert response.status_code == 422


async def test_list_runs_newest_first(client: AsyncClient) -> None:
    for goal in ["first", "second", "third"]:
        await create_run(client, goal)

    response = await client.get("/runs")
    assert response.status_code == 200
    assert [r["goal"] for r in response.json()] == ["third", "second", "first"]

    response = await client.get("/runs", params={"limit": 2})
    assert [r["goal"] for r in response.json()] == ["third", "second"]

    response = await client.get("/runs", params={"limit": 2, "offset": 1})
    assert [r["goal"] for r in response.json()] == ["second", "first"]


@pytest.mark.parametrize("limit", [0, 101])
async def test_list_limit_out_of_range_is_422(client: AsyncClient, limit: int) -> None:
    response = await client.get("/runs", params={"limit": limit})
    assert response.status_code == 422


async def test_health_ok(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok"}


async def test_health_reports_unreachable_database(
    client: AsyncClient, test_url: URL
) -> None:
    # Port 1 has nothing listening, so connecting fails immediately.
    override_session(test_url.set(port=1))

    response = await client.get("/health")

    assert response.status_code == 503
    assert response.json() == {"status": "error", "database": "unreachable"}
