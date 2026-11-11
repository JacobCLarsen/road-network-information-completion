"""Tests for the data layer that do not need a database (the session is mocked)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.db import engine as engine_module
from src.db.create_road_type_labels import road_type_from_tag
from src.db.exceptions import NotFoundError
from src.db.repositories import AttributeRepository, PredictionRepository


def make_fake_session(monkeypatch):
    """Replaces SessionLocal with a fake one and returns the fake session."""
    session = AsyncMock()
    fake_context = MagicMock()
    fake_context.__aenter__ = AsyncMock(return_value=session)
    fake_context.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(engine_module, "SessionLocal", lambda: fake_context)
    return session


async def test_session_scope_commits(monkeypatch):
    session = make_fake_session(monkeypatch)
    async with engine_module.session_scope():
        pass
    session.commit.assert_awaited_once()
    session.rollback.assert_not_awaited()


async def test_session_scope_rolls_back_on_error(monkeypatch):
    session = make_fake_session(monkeypatch)
    with pytest.raises(ValueError):
        async with engine_module.session_scope():
            raise ValueError("something went wrong")
    session.rollback.assert_awaited_once()
    session.commit.assert_not_awaited()


async def test_update_missing_row_raises_not_found():
    session = AsyncMock()
    session.get.return_value = None
    with pytest.raises(NotFoundError):
        await AttributeRepository(session).update(42, description="x")


def test_road_type_from_tag():
    assert road_type_from_tag("residential") == "residential"
    assert road_type_from_tag("primary_link") == "primary"  # link roads are merged
    assert road_type_from_tag("service") is None  # rare type
    assert road_type_from_tag("['residential', 'unclassified']") is None  # more than one tag
    assert road_type_from_tag(None) is None


async def test_save_predictions_checks_lengths():
    with pytest.raises(ValueError):
        await PredictionRepository(AsyncMock()).save_predictions(1, [10, 20], [1])
