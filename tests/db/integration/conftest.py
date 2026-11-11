"""Fixtures for the tests that use a real Postgres database.

The tests only run if TEST_DATABASE_URL is set (in .env or as an environment variable).
Each test runs inside a transaction that is rolled back at the end, so nothing is saved in the
database. That means it is safe to run them against your local database.
"""

import os

import pytest
import pytest_asyncio
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from src.db.models import OsmEdge, TaskType
from src.db.repositories import (
    AttributeRepository,
    FeatureSetRepository,
    ModelRunRepository,
    SplitRepository,
)

load_dotenv()
TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")

# Fake segment ids that are much bigger than the real ones, so they do not clash
SEGMENT_IDS = [9000000001, 9000000002, 9000000003]


def pytest_collection_modifyitems(config, items):
    # skip the integration tests if there is no test database
    if TEST_DATABASE_URL:
        return
    for item in items:
        if "tests/db/integration" in str(item.fspath):
            item.add_marker(pytest.mark.skip(reason="TEST_DATABASE_URL is not set"))


@pytest_asyncio.fixture
async def engine():
    engine = create_async_engine(TEST_DATABASE_URL, poolclass=NullPool)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def session(engine):
    async with engine.connect() as connection:
        transaction = await connection.begin()
        # "create_savepoint" makes the session use savepoints, so the outer transaction can be rolled back
        session = AsyncSession(bind=connection, expire_on_commit=False, join_transaction_mode="create_savepoint")
        yield session
        await session.close()
        await transaction.rollback()


@pytest_asyncio.fixture
async def segments(session):
    """Adds three fake road segments (the tables have foreign keys to them)."""
    session.add_all([
        OsmEdge(fid=SEGMENT_IDS[0], u=1, v=2, highway="primary", length=100.0),
        OsmEdge(fid=SEGMENT_IDS[1], u=2, v=3, highway="residential", length=200.0),
        OsmEdge(fid=SEGMENT_IDS[2], u=3, v=4, highway="residential", length=300.0),
    ])
    await session.flush()
    return SEGMENT_IDS


@pytest_asyncio.fixture
async def setup(session, segments):
    """Creates an attribute with two classes, a label rule, a split and a feature set."""
    from src.db.models import AttributeClass, LabelRule

    attribute = await AttributeRepository(session).create(name="test_attribute", task_type=TaskType.CLASSIFICATION)
    primary = AttributeClass(attribute_id=attribute.attribute_id, name="primary", category="major")
    residential = AttributeClass(attribute_id=attribute.attribute_id, name="residential", category="minor")
    session.add_all([primary, residential, LabelRule(attribute_id=attribute.attribute_id, version=1)])
    await session.flush()

    split = await SplitRepository(session).create_split(
        "random", {SEGMENT_IDS[0]: "train", SEGMENT_IDS[1]: "val", SEGMENT_IDS[2]: "test"}
    )
    feature_set = await FeatureSetRepository(session).get_or_create("test_features", 1, ["length", "degree"])
    return {
        "attribute": attribute,
        "primary": primary,
        "residential": residential,
        "split": split,
        "feature_set": feature_set,
    }


@pytest_asyncio.fixture
async def run(session, setup):
    return await ModelRunRepository(session).start_run(
        attribute_id=setup["attribute"].attribute_id,
        label_rule_version=1,
        split_id=setup["split"].split_id,
        feature_set_id=setup["feature_set"].feature_set_id,
        architecture="test_model",
        seed=0,
        hyperparameters={"lr": 0.01},
    )
