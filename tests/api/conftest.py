"""Fixtures for the API tests.

`client` calls the API like a browser would. The health tests only need that.

`db_client` is for the routes that read our tables. Those tables only exist in Postgres, so these
tests need TEST_DATABASE_URL (they are skipped without it). They use the same rolled back session as
the data layer tests in tests/db/integration, so they never change the database.
"""

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.api.main import app
from src.db.engine import get_session

# Reuse the database fixtures from the data layer tests (session, setup, run, ...)
from tests.db.integration.conftest import (  # noqa: F401
    TEST_DATABASE_URL,
    engine,
    run,
    segments,
    session,
    setup,
)


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def db_client(session):
    # make the routes use the test session instead of a normal one
    async def override_get_session():
        yield session

    app.dependency_overrides[get_session] = override_get_session
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()
