"""Shared fixtures for tests that use a real Postgres (the separate argus_test database).

Tests that use them are skipped when .env is missing or Postgres isn't reachable
(`docker compose up -d`).
"""

from collections.abc import Iterator

import pytest
from pydantic import ValidationError
from sqlalchemy import Engine, create_engine, delete, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from argus.config import get_settings
from argus.db import Base, Run

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
def clean_db(engine: Engine) -> Iterator[None]:
    """Deletes every run after the test, so tests don't see each other's rows."""
    yield
    with Session(engine) as session:
        session.execute(delete(Run))  # Steps go too, via ON DELETE CASCADE
        session.commit()
