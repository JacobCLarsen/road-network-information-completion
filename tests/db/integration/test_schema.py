"""Checks that our model classes in src/db/models match the real tables in the database.
The tables are made by the migrations in the data repo, so if someone changes a migration
this test will tell us that the models have to be updated."""

import pytest
from sqlalchemy import inspect
from sqlalchemy.dialects import postgresql

from src.db.engine import check_database
from src.db.models import SCHEMA, Base


def type_name(column_type):
    # turn the type into text like "integer" or "double precision" so we can compare them
    return column_type.compile(dialect=postgresql.dialect()).lower().replace(SCHEMA + ".", "")


def read_columns(sync_connection, table):
    inspector = inspect(sync_connection)
    columns = inspector.get_columns(table.name, table.schema)
    return {column["name"]: column for column in columns}


async def test_database_has_expected_migration(engine):
    await check_database(engine)


async def test_missing_migration_gives_error(engine):
    with pytest.raises(RuntimeError):
        await check_database(engine, expected_migration="attribute_prediction/9999_not_real.sql")


@pytest.mark.parametrize("table", Base.metadata.tables.values(), ids=lambda table: table.fullname)
async def test_model_matches_database(engine, table):
    async with engine.connect() as connection:
        db_columns = await connection.run_sync(read_columns, table)

    assert db_columns, f"{table.fullname} does not exist in the database"
    assert set(table.columns.keys()) == set(db_columns.keys())

    is_view = table.info.get("is_view", False)
    for column in table.columns:
        db_column = db_columns[column.name]
        assert type_name(column.type) == type_name(db_column["type"]), f"wrong type for {column.name}"
        if not is_view:  # views do not have NOT NULL in postgres
            assert column.nullable == db_column["nullable"], f"wrong nullable for {column.name}"
