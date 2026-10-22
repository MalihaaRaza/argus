"""Database models and engines (SQLAlchemy 2.0, Postgres via psycopg 3)."""

import uuid
from datetime import datetime
from functools import lru_cache
from typing import Any, Literal, get_args

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Engine,
    ForeignKey,
    MetaData,
    String,
    Text,
    create_engine,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from argus.config import get_settings

RunStatus = Literal["queued", "running", "passed", "failed", "error"]
RUN_STATUSES: tuple[str, ...] = get_args(RunStatus)

# Gives every constraint a predictable name (e.g. "fk_run_steps_run_id_runs").
# Without names, Postgres invents them, and Alembic can't reliably drop or alter
# a constraint it doesn't know the name of.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class Run(Base):
    """One test of a URL against a goal."""

    __tablename__ = "runs"
    # A CHECK constraint instead of a native Postgres ENUM: adding a status later
    # is a simple migration, while changing an ENUM needs hand-written ALTER TYPE.
    __table_args__ = (
        CheckConstraint(
            "status IN (" + ", ".join(f"'{s}'" for s in RUN_STATUSES) + ")",
            name="status",
        ),
    )

    # Generated in Python (at flush), so no database extension is needed.
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    url: Mapped[str] = mapped_column(Text)
    goal: Mapped[str] = mapped_column(Text)
    # Each default is set twice: `default` is used when inserting through the ORM,
    # `server_default` lives in the table itself and covers raw SQL inserts too.
    status: Mapped[str] = mapped_column(
        String(16), default="queued", server_default="queued"
    )
    reason: Mapped[str | None] = mapped_column(Text)
    step_count: Mapped[int] = mapped_column(default=0, server_default="0")
    input_tokens: Mapped[int] = mapped_column(default=0, server_default="0")
    output_tokens: Mapped[int] = mapped_column(default=0, server_default="0")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # passive_deletes=True: when a run is deleted, the database's ON DELETE CASCADE
    # removes the steps, instead of the ORM loading and deleting them one by one.
    steps: Mapped[list["RunStep"]] = relationship(
        back_populates="run",
        order_by="RunStep.id",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class RunStep(Base):
    """One tool call the agent made during a run."""

    __tablename__ = "run_steps"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), index=True
    )
    step_number: Mapped[int]
    tool_name: Mapped[str] = mapped_column(String(64))
    tool_input: Mapped[dict[str, Any]] = mapped_column(JSONB)
    result: Mapped[str] = mapped_column(Text)
    is_error: Mapped[bool]
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    run: Mapped[Run] = relationship(back_populates="steps")


# Engines are created on first use, not at import time, so importing this module
# (e.g. in tests or alembic) doesn't need a .env or a running database.
# psycopg 3 has both sync and async modes, so one DATABASE_URL serves both engines.


@lru_cache
def get_engine() -> Engine:
    """Sync engine, for the worker."""
    return create_engine(get_settings().database_url)


@lru_cache
def get_sessionmaker() -> sessionmaker[Session]:
    return sessionmaker(get_engine())


@lru_cache
def get_async_engine() -> AsyncEngine:
    """Async engine, for the API."""
    return create_async_engine(get_settings().database_url)


@lru_cache
def get_async_sessionmaker() -> async_sessionmaker[AsyncSession]:
    # Async code can't lazily reload attributes after commit (that would be hidden
    # I/O), so objects keep their loaded values instead of being expired.
    return async_sessionmaker(get_async_engine(), expire_on_commit=False)
