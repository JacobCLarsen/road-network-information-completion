from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.db.config import EXPECTED_MIGRATION, settings

if settings.database_url.startswith("sqlite"):
    # SQLite is only used in the API tests, and it does not support the pool settings below
    engine = create_async_engine(settings.database_url, echo=settings.debug)
else:
    engine = create_async_engine(
        settings.database_url,
        echo=settings.debug,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,  # checks that a connection still works before using it
        pool_recycle=1800,  # replace connections after 30 minutes
    )

SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@asynccontextmanager
async def session_scope():
    """Gives a database session. Everything is committed at the end of the with-block.
    If an error happens, everything is rolled back instead, so we never save half of something.

    Use it in scripts like this:
        async with session_scope() as session:
            ...
    """
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_session():
    """Same as session_scope, but for FastAPI routes: session: AsyncSession = Depends(get_session)"""
    async with session_scope() as session:
        yield session


async def check_database(db_engine=engine, expected_migration=EXPECTED_MIGRATION):
    """Checks that we can connect to the database and that it has run the migration we expect.
    The tables are made by the migrations in the data repo, so we only check them here."""
    async with db_engine.connect() as connection:
        result = await connection.execute(
            text("SELECT name FROM public.schema_migrations WHERE name = :name"),
            {"name": expected_migration},
        )
        if result.first() is None:
            raise RuntimeError(
                f"The database has not run the migration {expected_migration}. "
                "Run the migrations in the data repo first (docker compose run --rm migrate)."
            )
