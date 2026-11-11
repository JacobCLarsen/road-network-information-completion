"""Database access for the API.

The connection itself lives in the shared data layer (src/db), so the API and the ML scripts use the
same settings. Tables are created by the migrations in the astep-aau/data repo, never by the API.
"""

from sqlalchemy.orm import declarative_base

from src.db.engine import get_session

# Base for the old `models` schema tables in src/api/tables. They will move to src/db/models.
Base = declarative_base()

# FastAPI dependency: one session per request, committed when the request succeeds.
get_db = get_session
