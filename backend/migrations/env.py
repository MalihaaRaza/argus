"""Alembic entry point: connects to DATABASE_URL and compares against Argus's models."""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from argus.config import get_settings
from argus.db import Base

# The Alembic Config object, which gives access to values in alembic.ini.
config = context.config

# Set up Python logging from the [loggers] sections of alembic.ini.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Autogenerate diffs the live database against this metadata (all Argus models).
target_metadata = Base.metadata

# Read the URL from Settings instead of alembic.ini, so .env is the single source.
# (Passing it through config.set_main_option would break on a "%" in the password,
# because the ini parser treats "%" as interpolation syntax.)
DATABASE_URL = get_settings().database_url


def run_migrations_offline() -> None:
    """Emit the migration SQL as a script (`alembic upgrade head --sql`), no DB needed."""
    context.configure(
        url=DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Connect to the database and apply migrations."""
    # NullPool: a one-off command has no reason to keep connections open.
    connectable = create_engine(DATABASE_URL, poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
