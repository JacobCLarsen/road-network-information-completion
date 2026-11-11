import pytest
from sqlalchemy.exc import IntegrityError

from src.db.exceptions import NotFoundError
from src.db.models import Epoch, MetricLevel, Partition, RunStatus, SegmentLabel, TaskType
from src.db.repositories import (
    AttributeRepository,
    EpochRepository,
    FeatureSetRepository,
    LabelRepository,
    MetricRepository,
    ModelRunRepository,
    PredictionRepository,
    SegmentRepository,
    SplitRepository,
)
from tests.db.integration.conftest import SEGMENT_IDS


# Basic CRUD (tested with the attribute table)

async def test_crud(session):
    repo = AttributeRepository(session)

    attribute = await repo.create(name="crud_test", task_type=TaskType.REGRESSION)
    assert attribute.attribute_id is not None

    assert (await repo.get(attribute.attribute_id)).name == "crud_test"
    assert [a.name for a in await repo.get_all(name="crud_test")] == ["crud_test"]

    await repo.update(attribute.attribute_id, description="new description")
    assert (await repo.get(attribute.attribute_id)).description == "new description"

    await repo.delete(attribute.attribute_id)
    assert await repo.get(attribute.attribute_id) is None


async def test_update_and_delete_missing_row(session):
    repo = AttributeRepository(session)
    with pytest.raises(NotFoundError):
        await repo.update(-1, description="x")
    with pytest.raises(NotFoundError):
        await repo.delete(-1)


async def test_duplicate_name_is_not_allowed(session):
    repo = AttributeRepository(session)
    await repo.create(name="duplicate", task_type=TaskType.REGRESSION)
    with pytest.raises(IntegrityError):
        await repo.create(name="duplicate", task_type=TaskType.REGRESSION)


# Attributes, labels, segments and splits

async def test_road_type_attribute_exists(session):
    attribute = await AttributeRepository(session).get_by_name("road_type")
    assert attribute.task_type == TaskType.CLASSIFICATION


async def test_get_classes(session, setup):
    classes = await AttributeRepository(session).get_classes(setup["attribute"].attribute_id)
    assert [c.name for c in classes] == ["primary", "residential"]


async def test_get_labels(session, setup):
    attribute_id = setup["attribute"].attribute_id
    primary_id = setup["primary"].class_id
    session.add(SegmentLabel(segment_id=SEGMENT_IDS[0], attribute_id=attribute_id, label_rule_version=1, class_id=primary_id))
    await session.flush()

    labels = await LabelRepository(session).get_labels(attribute_id, 1)
    assert labels == {SEGMENT_IDS[0]: primary_id}


async def test_save_labels_replaces_old_labels(session, setup):
    attribute_id = setup["attribute"].attribute_id
    repo = LabelRepository(session)
    await repo.save_labels(attribute_id, 1, {SEGMENT_IDS[0]: setup["primary"].class_id})
    # saving again should replace the labels, not add to them
    await repo.save_labels(attribute_id, 1, {SEGMENT_IDS[1]: setup["residential"].class_id})
    assert await repo.get_labels(attribute_id, 1) == {SEGMENT_IDS[1]: setup["residential"].class_id}


async def test_get_or_create_rule(session, setup):
    repo = LabelRepository(session)
    attribute_id = setup["attribute"].attribute_id
    existing = await repo.get_or_create_rule(attribute_id, 1)
    assert existing.version == 1
    new_rule = await repo.get_or_create_rule(attribute_id, 2, "second rule")
    assert new_rule.description == "second rule"


async def test_add_class(session):
    repo = AttributeRepository(session)
    attribute = await repo.create(name="class_test", task_type=TaskType.CLASSIFICATION)
    new_class = await repo.add_class(attribute.attribute_id, "motorway", "highways")
    assert new_class.class_id is not None
    assert [c.name for c in await repo.get_classes(attribute.attribute_id)] == ["motorway"]


async def test_get_highway_tags(session, segments):
    tags = await SegmentRepository(session).get_highway_tags()
    assert tags[SEGMENT_IDS[0]] == "primary"
    assert tags[SEGMENT_IDS[1]] == "residential"


async def test_segments(session, segments):
    segment = await SegmentRepository(session).get(SEGMENT_IDS[1])
    assert segment.highway == "residential"
    assert segment.length_m == 200.0


async def test_split(session, setup):
    partitions = await SplitRepository(session).get_partitions(setup["split"].split_id)
    assert partitions == {"train": [SEGMENT_IDS[0]], "val": [SEGMENT_IDS[1]], "test": [SEGMENT_IDS[2]]}


async def test_feature_set_is_only_created_once(session):
    repo = FeatureSetRepository(session)
    first = await repo.get_or_create("same_features", 1, ["a", "b"])
    second = await repo.get_or_create("same_features", 1, ["a", "b"])
    assert first.feature_set_id == second.feature_set_id


# Model runs and their results

async def test_start_and_finish_run(session, run):
    assert run.status == RunStatus.RUNNING
    assert run.started_at is not None
    assert run.hyperparameters == {"lr": 0.01}

    await ModelRunRepository(session).finish_run(run.run_id)
    assert run.status == RunStatus.FINISHED

    await ModelRunRepository(session).fail_run(run.run_id)
    assert run.status == RunStatus.FAILED


async def test_epochs(session, run):
    repo = EpochRepository(session)
    await repo.create(run_id=run.run_id, epoch=1, train_loss=1.0, val_loss=1.2, val_macro_f1=0.3)
    await repo.create(run_id=run.run_id, epoch=2, train_loss=0.8, val_loss=1.0, val_macro_f1=0.4)

    epochs = await repo.get_all(run_id=run.run_id)
    assert sorted(e.epoch for e in epochs) == [1, 2]
    assert (await repo.get((run.run_id, 2))).train_loss == 0.8


async def test_save_metrics(session, run, setup):
    repo = MetricRepository(session)
    await repo.save_metrics(run.run_id, Partition.TEST, MetricLevel.FINE, {"macro_f1": 0.7, "accuracy": 0.8})
    await repo.save_metrics(run.run_id, Partition.TEST, MetricLevel.FINE, {"f1": 0.9}, class_id=setup["primary"].class_id)

    metrics = await repo.get_all(run_id=run.run_id)
    scores = {m.metric_name: m.value for m in metrics}
    assert scores == {"macro_f1": 0.7, "accuracy": 0.8, "f1": 0.9}


async def test_save_predictions(session, run, setup):
    primary = setup["primary"].class_id
    residential = setup["residential"].class_id
    await PredictionRepository(session).save_predictions(
        run.run_id, SEGMENT_IDS, [primary, residential, residential],
        probabilities=[[0.9, 0.1], [0.2, 0.8], [0.4, 0.6]],
    )

    predictions = await PredictionRepository(session).get_all(run_id=run.run_id)
    assert {p.segment_id: p.predicted_class_id for p in predictions} == {
        SEGMENT_IDS[0]: primary, SEGMENT_IDS[1]: residential, SEGMENT_IDS[2]: residential,
    }


async def test_deleting_run_deletes_epochs(session, run):
    await EpochRepository(session).create(run_id=run.run_id, epoch=1, train_loss=1.0)
    await ModelRunRepository(session).delete(run.run_id)
    session.expunge_all()  # forget what the session remembers, so we really read from the database
    assert await EpochRepository(session).get_all(run_id=run.run_id) == []
    assert await session.get(Epoch, (run.run_id, 1)) is None
