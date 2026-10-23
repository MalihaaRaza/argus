"""Point the API's dependencies at test services, shared by the API test modules.

Imported as `from api_overrides import ...` (like fakes.py): pytest puts the
tests/ folder on sys.path because it has no __init__.py.
"""

from collections.abc import AsyncIterator

import redis.asyncio as aioredis
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from argus.api import app, get_redis, get_session


def override_session(url: URL) -> None:
    """Point the API's session dependency at the database `url`."""
    # NullPool: every test runs in its own event loop, and a pooled async
    # connection can't be reused from a different loop.
    engine = create_async_engine(url, poolclass=NullPool)
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    async def get_test_session() -> AsyncIterator[AsyncSession]:
        async with sessions() as session:
            yield session

    app.dependency_overrides[get_session] = get_test_session


def override_redis(url: str) -> None:
    """Give each request its own Redis client, closed when the request ends.

    The app's shared client is bound to the event loop that first used it, and
    the test client may run each connection in a different loop.
    """

    async def get_test_redis() -> AsyncIterator[aioredis.Redis]:
        client = aioredis.Redis.from_url(url)
        try:
            yield client
        finally:
            await client.aclose()

    app.dependency_overrides[get_redis] = get_test_redis
