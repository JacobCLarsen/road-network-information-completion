# Tests for saving model runs (src/ml/save_run.py) in a real database. Like the other integration tests, everything
# runs in a transaction that is rolled back afterwards, so nothing is saved. The fixtures (session, segments, setup)
# are in conftest.py.

import pytest
from sqlalchemy import select

from src.db.models import Epoch, Metric, ModelRun, Prediction, RunStatus
from src.ml import save_run
from tests.db.integration.conftest import SEGMENT_IDS


# Checks that the same split is found again instead of saved twice, and that a different split is saved as a new one
@pytest.mark.asyncio
async def test_same_split_is_reused(session, setup):
    attribute_id = setup["attribute"].attribute_id
    assignments = {SEGMENT_IDS[0]: "train", SEGMENT_IDS[1]: "val", SEGMENT_IDS[2]: "test"}

    first = await save_run.find_or_create_split(session, attribute_id, "random (test)", assignments)
    second = await save_run.find_or_create_split(session, attribute_id, "random (test)", assignments)
    assert first == second

    # Two segments swapped: a different split
    swapped = {SEGMENT_IDS[0]: "val", SEGMENT_IDS[1]: "train", SEGMENT_IDS[2]: "test"}
    third = await save_run.find_or_create_split(session, attribute_id, "random (test)", swapped)
    assert third != first


# Checks a whole run: started as "running", then the training curve, the scores, the predictions and the best epoch
# are saved, and the run is "finished"
@pytest.mark.asyncio
async def test_run_is_saved_with_its_results(session, setup):
    attribute_id = setup["attribute"].attribute_id
    class_ids = {"primary": setup["primary"].class_id, "residential": setup["residential"].class_id}

    run_id = await save_run.create_run(session, attribute_id, setup["split"].split_id, "GraphSAGE", 3,
                                       {"dropout": 0.2})
    run = await session.get(ModelRun, run_id)
    assert run.status == RunStatus.RUNNING

    history = [{"epoch": 1, "train_loss": 1.5, "val_macro_f1": None},
               {"epoch": 2, "train_loss": 1.0, "val_macro_f1": 0.5}]
    # The same keys as evaluate() returns, with made-up scores
    scores = {"precision": 0.5, "recall": 0.5, "f1": 0.5}
    results = {"val": {"macro_f1": 0.5, "accuracy": 0.6, "grouped_macro_f1": 0.7,
                       "per_class": {"primary": scores, "residential": scores}}}
    predicted = [class_ids["primary"], class_ids["residential"], class_ids["residential"]]

    await save_run.save_results(session, run_id, class_ids, SEGMENT_IDS, predicted, results, history, best_epoch=2)

    run = await session.get(ModelRun, run_id)
    assert run.status == RunStatus.FINISHED
    assert run.hyperparameters == {"dropout": 0.2, "best_epoch": 2}

    epochs = (await session.execute(select(Epoch).where(Epoch.run_id == run_id).order_by(Epoch.epoch))).scalars().all()
    assert [(e.epoch, e.train_loss, e.val_macro_f1) for e in epochs] == [(1, 1.5, None), (2, 1.0, 0.5)]

    # 2 overall scores, 3 scores for each of the 2 road types and 1 coarse score
    metrics = (await session.execute(select(Metric).where(Metric.run_id == run_id))).scalars().all()
    assert len(metrics) == 2 + 3 * 2 + 1

    predictions = (await session.execute(select(Prediction).where(Prediction.run_id == run_id))).scalars().all()
    assert {p.segment_id: p.predicted_class_id for p in predictions} == dict(zip(SEGMENT_IDS, predicted))

