# Tests for the parts of src/ml/save_run.py that do not need a database. Saving to a real database is tested in
# tests/db/integration/test_save_run.py.

import pytest
import torch

from src.ml import config, save_run
from src.ml.data.road_graph_builder import ROAD_CLASSES


# A graph of 3 segments. Segment 0 is a two-way road, so it has two database ids (11 and 12). Segment 2 has no label,
# so it is in none of the masks.
@pytest.fixture
def graph_and_masks():
    graph = {"edge_to_segment": {11: 0, 12: 0, 20: 1, 30: 2}}
    masks = {
        "train": torch.tensor([True, False, False]),
        "validation": torch.tensor([False, True, False]),
        "test": torch.tensor([False, False, False]),
    }
    return graph, masks


# Checks that both ids of a two-way road get the partition of their segment, and that unlabelled segments are left
# out (the database only stores the labelled segments of a split)
def test_split_assignments_per_database_id(graph_and_masks):
    graph, masks = graph_and_masks

    assert save_run.split_assignments(graph, masks) == {11: "train", 12: "train", 20: "val"}


# Checks that every database id gets the class_id of the road type predicted for its segment, also the unlabelled
# segment 30
def test_prediction_rows_per_database_id(graph_and_masks):
    graph, _ = graph_and_masks
    # Made-up class ids: 100 for the first road type, 101 for the second, ...
    class_ids = {name: 100 + i for i, name in enumerate(ROAD_CLASSES)}
    predictions = torch.tensor([0, 6, 2])

    segment_ids, predicted_class_ids = save_run.prediction_rows(graph, predictions, class_ids)

    assert segment_ids == [11, 12, 20, 30]
    assert predicted_class_ids == [100, 100, 106, 102]


# Checks that one feature name is saved for every column of x: the 10 map and traffic features and one neighbour
# label share per road type
def test_one_feature_name_per_column():
    assert len(save_run.FEATURE_NAMES) == 10 + len(ROAD_CLASSES)
    assert len(set(save_run.FEATURE_NAMES)) == len(save_run.FEATURE_NAMES)


# Checks that with SAVE_RUNS = False nothing touches the database, so the models can be trained without one
def test_nothing_is_saved_when_save_runs_is_off(monkeypatch, graph_and_masks):
    graph, masks = graph_and_masks
    monkeypatch.setattr(config, "SAVE_RUNS", False)

    def no_database(work):
        raise AssertionError("the database must not be used")

    monkeypatch.setattr(save_run, "run_in_database", no_database)

    assert save_run.save_split(graph, masks) is None
    assert save_run.start_run("GraphSAGE", 0, None, {}) is None
    save_run.finish_run(None, graph, torch.tensor([0, 0, 0]), {})
    save_run.fail_run(None)
