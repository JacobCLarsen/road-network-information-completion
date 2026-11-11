"""Small script to check that we can connect to the database in DATABASE_URL.

Run it from the repo root:
    python -m src.db.check_connection

It works for both the local docker database and the server database (through kubectl port-forward).
It prints which database we are connected to, if the migrations are up to date, and how many
segments and labels there are.
"""

import asyncio

from sqlalchemy import text

from src.db.config import EXPECTED_MIGRATION, settings
from src.db.engine import check_database, engine


async def main():
    # Do not print the password from the URL
    url_without_password = settings.database_url.split("@")[-1]
    print(f"Connecting to {url_without_password} ...")

    try:
        async with engine.connect() as connection:
            version = await connection.scalar(text("SELECT version()"))
            print(f"Connected: {version.split(',')[0]}")

        await check_database()
        print(f"Migrations OK (found {EXPECTED_MIGRATION})")

        async with engine.connect() as connection:
            segments = await connection.scalar(text("SELECT count(*) FROM attribute_prediction.segment"))
            labels = await connection.scalar(text("SELECT count(*) FROM attribute_prediction.segment_label"))
            print(f"Segments: {segments}, labels: {labels}")
            if labels == 0:
                print("There are no labels yet, run: python -m src.db.create_road_type_labels")
    except Exception as error:
        print(f"FAILED: {error}")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
