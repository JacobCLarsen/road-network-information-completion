import pytest

from src.db.models import MetricLevel, Partition
from src.db.repositories import EpochRepository, MetricRepository
from tests.api.conftest import TEST_DATABASE_URL

pytestmark = pytest.mark.skipif(not TEST_DATABASE_URL, reason="TEST_DATABASE_URL is not set")


async def test_list_runs(db_client, run):
    response = await db_client.get("/runs/")
    assert response.status_code == 200

    # newest run first, so our test run is the first one
    first = response.json()[0]
    assert first["run_id"] == run.run_id
    assert first["architecture"] == "test_model"
    assert first["status"] == "running"
    assert first["hyperparameters"] == {"lr": 0.01}


async def test_get_run_with_epochs_and_metrics(db_client, session, run):
    epochs = EpochRepository(session)
    # added in the wrong order on purpose, the API should sort them
    await epochs.create(run_id=run.run_id, epoch=2, train_loss=0.4, val_loss=0.5, val_macro_f1=0.6)
    await epochs.create(run_id=run.run_id, epoch=1, train_loss=0.8, val_loss=0.9, val_macro_f1=0.3)
    await MetricRepository(session).save_metrics(run.run_id, Partition.TEST, MetricLevel.FINE, {"macro_f1": 0.7})

    response = await db_client.get(f"/runs/{run.run_id}")
    assert response.status_code == 200

    data = response.json()
    assert data["run_id"] == run.run_id
    assert [epoch["epoch"] for epoch in data["epochs"]] == [1, 2]
    assert data["metrics"] == [
        {"partition": "test", "level": "fine", "metric_name": "macro_f1", "class_id": None, "value": 0.7}
    ]


async def test_get_missing_run(db_client):
    response = await db_client.get("/runs/-1")
    assert response.status_code == 404
