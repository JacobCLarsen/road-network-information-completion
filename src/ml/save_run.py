# src/ml/save_run.py

# Saves the training runs of the models in the database (the attribute_prediction schema), so the results can be
# shown by the API and compared later. Used by the GraphSAGE and MLP run scripts (dev_run_* and final_testrun_*). The
# sweeps do not save their runs, because they train many models only to choose the settings.
#
# A script uses it like this:
#     split_id = save_split(graph, masks)                        # once, before the seeds
#     run_id = start_run("GraphSAGE", seed, split_id, settings)  # saved with status "running"
#     ... train ...                                              # if training fails: fail_run(run_id)
#     finish_run(run_id, graph, predictions, {"val": result}, history)
#
# Set SAVE_RUNS = False in src/ml/config.py to train without a database: then every function here does nothing.
#
# The models are plain (not async) code, so every function opens its own database session with asyncio.run. Training
# can take hours, so no session is kept open while a model trains.

import asyncio

from sqlalchemy import select

from src.db.create_road_type_labels import LABEL_RULE_VERSION
from src.db.engine import engine, session_scope
from src.db.models import AttributeClass, MetricLevel, Partition
from src.db.repositories import (AttributeRepository, EpochRepository, FeatureSetRepository, MetricRepository,
                                 ModelRunRepository, PredictionRepository, SplitRepository)
from src.ml import config
from src.ml.data.road_graph_builder import ROAD_CLASSES

# The input features of the models, in the order of the columns of x (see src/ml/README.md). If the features change,
# give the feature set a new version, so the old runs keep the list they were trained with.
FEATURE_NAMES = (["log_length", "curvature", "one_way", "degree",
                  "log_traversals", "has_traffic", "avg_speed", "avg_acceleration", "speed_std", "slow_share"]
                 + ["neighbour_share_" + name for name in ROAD_CLASSES])
FEATURE_SET_NAME = "geometry_traffic_neighbour_labels"
FEATURE_SET_VERSION = 1

# The names of the split masks in make_split_masks, and the partition names in the database
PARTITIONS = {"train": "train", "validation": "val", "test": "test"}


def split_method():
    # How the split was made, e.g. "random (seed 42, train 0.6, val 0.2)"
    return f"{config.STRATEGY} (seed {config.SPLIT_SEED}, train {config.TRAIN_SHARE}, val {config.VAL_SHARE})"


def split_assignments(graph, masks):
    # Returns {segment_id: "train" / "val" / "test"} for every labelled database segment.
    # A two-way road is one segment in the graph, but two segment ids in the database. graph["edge_to_segment"] gives
    # the graph segment of every database id, so both ids get the partition of their graph segment.
    partition_of_segment = {}
    for mask_name, partition in PARTITIONS.items():
        for index in masks[mask_name].nonzero().view(-1).tolist():
            partition_of_segment[index] = partition

    assignments = {}
    for edge_id, index in graph["edge_to_segment"].items():
        if index in partition_of_segment:
            assignments[edge_id] = partition_of_segment[index]
    return assignments


def prediction_rows(graph, predictions, class_ids):
    # Returns two lists in the same order: the database segment ids and the class_id predicted for each of them.
    # predictions has one road type index per graph segment, also for unlabelled segments. class_ids is
    # {road type name: class_id in the database}.
    segment_ids = []
    predicted_class_ids = []
    for edge_id, index in graph["edge_to_segment"].items():
        segment_ids.append(edge_id)
        predicted_class_ids.append(class_ids[ROAD_CLASSES[predictions[index].item()]])
    return segment_ids, predicted_class_ids


def run_in_database(work):
    # Runs work(session) in one database session and returns its result. Everything is saved together at the end,
    # or nothing if there is an error. The engine is closed afterwards, because asyncio.run makes a new event loop
    # every time, and the connections can not be reused in the next one.
    async def main():
        try:
            async with session_scope() as session:
                return await work(session)
        finally:
            await engine.dispose()

    return asyncio.run(main())


# The async functions below do the actual work with a session. They are separate from the plain functions further
# down, so the tests can give them a session that is rolled back afterwards.

async def get_road_type(session):
    # Returns the attribute_id of road_type and {road type name: class_id}
    road_type = await AttributeRepository(session).get_by_name("road_type")
    if road_type is None:
        raise ValueError("The attribute road_type is not in the database")
    result = await session.execute(select(AttributeClass).where(AttributeClass.attribute_id == road_type.attribute_id))
    class_ids = {row.name: row.class_id for row in result.scalars()}
    return road_type.attribute_id, class_ids


async def find_or_create_split(session, attribute_id, method, assignments):
    # Returns the id of a saved split with exactly these assignments, or saves a new one. Every model uses the same
    # split, so normally it is only saved the first time.
    splits = SplitRepository(session)
    for split in await splits.get_all(method=method, attribute_id=attribute_id):
        saved = {}
        partitions = await splits.get_partitions(split.split_id)
        for partition, segment_ids in partitions.items():
            for segment_id in segment_ids:
                saved[segment_id] = partition
        if saved == assignments:
            return split.split_id

    split = await splits.create_split(method, assignments, attribute_id)
    print("Saved a new split in the database:", split.split_id)
    return split.split_id


async def create_run(session, attribute_id, split_id, architecture, seed, hyperparameters):
    # Saves a new run with status "running" and returns its run_id
    feature_set = await FeatureSetRepository(session).get_or_create(FEATURE_SET_NAME, FEATURE_SET_VERSION,
                                                                    FEATURE_NAMES)
    run = await ModelRunRepository(session).start_run(attribute_id, LABEL_RULE_VERSION, split_id,
                                                      feature_set.feature_set_id, architecture, seed, hyperparameters)
    return run.run_id


async def save_results(session, run_id, class_ids, segment_ids, predicted_class_ids, results, history, best_epoch):
    # Saves the training curve, the scores and the predictions of a run, and marks it as finished
    runs = ModelRunRepository(session)

    # The best epoch is only known after training, so it is added to the saved settings now. A new dict is made,
    # because the database only notices that the column changed when it gets a new value.
    if best_epoch is not None:
        run = await runs.get(run_id)
        run.hyperparameters = {**(run.hyperparameters or {}), "best_epoch": best_epoch}

    epochs = EpochRepository(session)
    for row in history or []:
        await epochs.create(run_id=run_id, epoch=row["epoch"], train_loss=row["train_loss"],
                            val_macro_f1=row["val_macro_f1"])

    metrics = MetricRepository(session)
    for partition, result in results.items():
        await metrics.save_metrics(run_id, Partition(partition), MetricLevel.FINE,
                                   {"macro_f1": result["macro_f1"], "accuracy": result["accuracy"]})
        # One precision, recall and F1 per road type
        for name, scores in result["per_class"].items():
            await metrics.save_metrics(run_id, Partition(partition), MetricLevel.FINE,
                                       {"precision": scores["precision"], "recall": scores["recall"],
                                        "f1": scores["f1"]},
                                       class_id=class_ids[name])
        # The macro-F1 over the coarse categories (highways, main roads, minor roads)
        if "grouped_macro_f1" in result:
            await metrics.save_metrics(run_id, Partition(partition), MetricLevel.COARSE,
                                       {"macro_f1": result["grouped_macro_f1"]})

    await PredictionRepository(session).save_predictions(run_id, segment_ids, predicted_class_ids)
    await runs.finish_run(run_id)


# The plain functions the run scripts call

def save_split(graph, masks):
    """Saves the split (or finds the same one saved before) and returns its split_id."""
    if not config.SAVE_RUNS:
        return None

    async def work(session):
        attribute_id, _ = await get_road_type(session)
        return await find_or_create_split(session, attribute_id, split_method(), split_assignments(graph, masks))

    return run_in_database(work)


def start_run(architecture, seed, split_id, hyperparameters):
    """Saves a new run with status "running" before training starts, and returns its run_id."""
    if not config.SAVE_RUNS:
        return None

    async def work(session):
        attribute_id, _ = await get_road_type(session)
        return await create_run(session, attribute_id, split_id, architecture, seed, hyperparameters)

    run_id = run_in_database(work)
    print("Saving this run in the database as run", run_id)
    return run_id


def finish_run(run_id, graph, predictions, results, history=None, best_epoch=None):
    """Saves the results of a run after training and marks it as finished.

    predictions: one road type index per graph segment (for every segment, also the unlabelled ones)
    results:     {"val": evaluate(...)} and/or {"test": evaluate(...)}, evaluated with class_names=ROAD_CLASSES
    history:     the training curve, a list of {"epoch", "train_loss", "val_macro_f1"}
    best_epoch:  the epoch the predictions come from, added to the saved settings
    """
    if not config.SAVE_RUNS:
        return

    async def work(session):
        _, class_ids = await get_road_type(session)
        segment_ids, predicted_class_ids = prediction_rows(graph, predictions, class_ids)
        await save_results(session, run_id, class_ids, segment_ids, predicted_class_ids, results, history, best_epoch)

    run_in_database(work)
    print("Saved run", run_id, "in the database")


def fail_run(run_id):
    """Marks a run as failed, e.g. when training crashed or was stopped."""
    if not config.SAVE_RUNS:
        return

    async def work(session):
        await ModelRunRepository(session).fail_run(run_id)

    run_in_database(work)
    print("Marked run", run_id, "as failed in the database")
