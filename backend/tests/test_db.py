"""Models against a real Postgres, in a separate argus_test database.

Skipped when .env is missing or Postgres isn't reachable (`docker compose up -d`).
"""

import asyncio
import uuid
from collections.abc import Iterator

import pytest
from pydantic import ValidationError
from sqlalchemy import Engine, create_engine, delete, func, select, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import Session

from argus.config import get_settings
from argus.db import Base, Run, RunStep

pytestmark = pytest.mark.db

TEST_DB = "argus_test"


@pytest.fixture(scope="session")
def test_url() -> URL:
    try:
        main_url = make_url(get_settings().database_url)
    except ValidationError:
        pytest.skip("DATABASE_URL is not set")

    # CREATE DATABASE can't run inside a transaction, hence AUTOCOMMIT. We connect
    # to the main database because we can't connect to one that doesn't exist yet.
    admin = create_engine(
        main_url, isolation_level="AUTOCOMMIT", connect_args={"connect_timeout": 3}
    )
    try:
        with admin.connect() as conn:
            exists = conn.scalar(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": TEST_DB},
            )
            if not exists:
                conn.execute(text(f'CREATE DATABASE "{TEST_DB}"'))
    except OperationalError:
        pytest.skip("Postgres is not reachable")
    finally:
        admin.dispose()

    return main_url.set(database=TEST_DB)


@pytest.fixture(scope="session")
def engine(test_url: URL) -> Iterator[Engine]:
    engine = create_engine(test_url)
    # Fresh tables for every test session, so leftovers from a crashed run don't matter.
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    with Session(engine) as session:
        yield session
        session.rollback()  # In case the test left a failed transaction behind
        session.execute(delete(Run))  # Steps go too, via ON DELETE CASCADE
        session.commit()


def make_step(number: int, tool: str = "click") -> RunStep:
    return RunStep(
        step_number=number,
        tool_name=tool,
        tool_input={"ref": f"e{number}"},
        result="ok",
        is_error=False,
    )


def test_run_with_steps_round_trip(engine: Engine, session: Session) -> None:
    run = Run(url="https://shop.test", goal="Add a backpack to the cart")
    run.steps = [
        make_step(1, "click"),
        make_step(2, "type_text"),
        make_step(3, "finish"),
    ]
    session.add(run)
    session.commit()
    run_id = run.id

    # A new session, so the data really comes from the database, not a cache.
    with Session(engine) as fresh:
        loaded = fresh.get_one(Run, run_id)
        assert loaded.goal == "Add a backpack to the cart"
        assert [s.step_number for s in loaded.steps] == [1, 2, 3]
        assert [s.tool_name for s in loaded.steps] == [
            "click",
            "type_text",
            "finish",
        ]
        assert loaded.steps[1].tool_input == {"ref": "e2"}  # JSONB round-trip
        assert loaded.steps[0].created_at is not None


def test_run_defaults(session: Session) -> None:
    run = Run(url="https://shop.test", goal="g")
    session.add(run)
    session.commit()
    session.refresh(run)  # Reload, so server-side values like created_at are read

    assert isinstance(run.id, uuid.UUID)
    assert run.status == "queued"
    assert (run.step_count, run.input_tokens, run.output_tokens) == (0, 0, 0)
    assert run.reason is None
    assert run.started_at is None
    assert run.finished_at is None
    assert run.created_at.tzinfo is not None  # Timezone-aware


def test_server_defaults_apply_to_raw_sql(session: Session) -> None:
    # An insert that bypasses the ORM still gets defaults from the table itself.
    run_id = uuid.uuid4()
    session.execute(
        text("INSERT INTO runs (id, url, goal) VALUES (:id, 'u', 'g')"), {"id": run_id}
    )
    row = session.execute(
        text("SELECT status, step_count FROM runs WHERE id = :id"), {"id": run_id}
    ).one()
    assert tuple(row) == ("queued", 0)


def test_deleting_run_deletes_steps(session: Session) -> None:
    run = Run(url="https://shop.test", goal="g", steps=[make_step(1), make_step(2)])
    session.add(run)
    session.commit()
    run_id = run.id

    # A Core DELETE skips the ORM's own cascade, so this checks the database's
    # ON DELETE CASCADE rule.
    session.execute(delete(Run).where(Run.id == run_id))
    session.commit()

    count = session.scalar(
        select(func.count()).select_from(RunStep).where(RunStep.run_id == run_id)
    )
    assert count == 0


def test_invalid_status_rejected(session: Session) -> None:
    session.add(Run(url="https://shop.test", goal="g", status="skipped"))
    with pytest.raises(IntegrityError, match="ck_runs_status"):
        session.commit()


def test_async_session_reads_run(session: Session, test_url: URL) -> None:
    run = Run(url="https://shop.test", goal="async check")
    session.add(run)
    session.commit()
    run_id = run.id

    async def load_goal() -> str:
        engine = create_async_engine(test_url)
        try:
            async with engine.connect() as conn:
                result = await conn.scalar(select(Run.goal).where(Run.id == run_id))
                return str(result)
        finally:
            await engine.dispose()

    # The same psycopg URL works for the async engine the API will use.
    assert asyncio.run(load_goal()) == "async check"
