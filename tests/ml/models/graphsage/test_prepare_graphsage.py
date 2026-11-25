# Tests for prepare_data() in train_graphsage.py, without loading the Beijing road network from the database.
# load_segments, RoadGraphBuilder and make_split_masks are replaced by stand-ins, so only prepare_data's own steps are
# tested: which mask each step gets, and that the returned graph has the new features.

from unittest.mock import Mock

import pytest
import torch
import src.ml.models.graphsage.train_graphsage as train_module
from src.ml import config
from src.ml.class_weights import compute_class_weights
from src.ml.data.road_graph_builder import ROAD_CLASSES
from src.ml.evaluation import labels_in_data


# Six segments: 0-1 train, 2 validation, 3-4 test, 5 unlabelled. Road type 7 only occurs in the test set, so it shows
# whether test labels reach the class weights.
@pytest.fixture
def stand_ins(monkeypatch):
    graph = {
        "x": torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]),
        "y": torch.tensor([0, 1, 2, 7, 7, -1]),
        "mask": torch.tensor([True, True, True, True, True, False]),
    }
    masks = {
        "train": torch.tensor([True, True, False, False, False, False]),
        "validation": torch.tensor([False, False, True, False, False, False]),
        "test": torch.tensor([False, False, False, True, True, False]),
    }
    # Empty placeholders instead of real road segments: prepare_data only passes them on
    segments = [object() for _ in range(6)]

    builder = Mock()
    builder.get_static_components.return_value = graph
    # The features after add_neighbour_label_features: x gets new columns. The stand-in sets builder.x to this when
    # add_neighbour_label_features is called, like the real builder does.
    x_with_neighbour_features = torch.arange(18, dtype=torch.float).reshape(6, 3)
    builder.add_neighbour_label_features.side_effect = (
        lambda train_mask: setattr(builder, "x", x_with_neighbour_features))

    load_segments = Mock(return_value=segments)
    builder_class = Mock(return_value=builder)
    split_function = Mock(return_value=masks)

    monkeypatch.setattr(train_module, "load_segments", load_segments)
    monkeypatch.setattr(train_module, "RoadGraphBuilder", builder_class)
    monkeypatch.setattr(train_module, "make_split_masks", split_function)

    return {"graph": graph, "masks": masks, "segments": segments, "builder": builder,
            "load_segments": load_segments, "builder_class": builder_class, "split_function": split_function,
            "x_with_neighbour_features": x_with_neighbour_features}


# Checks that the segments come from the database loader, go into the graph builder, and that the split is made from the
# built graph
def test_loads_segments_and_builds_the_split(stand_ins):
    train_module.prepare_data()

    stand_ins["load_segments"].assert_called_once_with()
    stand_ins["builder_class"].assert_called_once_with(stand_ins["segments"])
    stand_ins["split_function"].assert_called_once_with(stand_ins["graph"])


# Checks that standardising and the neighbour label features use the training mask only. With the validation or test
# mask, the features would contain information from the segments the model is scored on.
def test_feature_steps_use_the_training_mask_only(stand_ins):
    train_module.prepare_data()

    builder = stand_ins["builder"]
    train_mask = stand_ins["masks"]["train"]
    builder.standardize_features.assert_called_once_with(stand_ins["segments"], train_mask)
    builder.add_neighbour_label_features.assert_called_once_with(train_mask)


# Checks that the returned graph has the x with the neighbour label features. add_neighbour_label_features makes a new
# x instead of changing the old one, so without graph["x"] = builder.x the model would never see the new columns.
def test_returned_graph_has_the_neighbour_label_features(stand_ins):
    graph, *_ = train_module.prepare_data()

    assert graph is stand_ins["graph"]
    assert torch.equal(graph["x"], stand_ins["x_with_neighbour_features"])
    assert graph["x"].shape[1] == 3


# Checks that the masks come back unchanged and in the order the run scripts unpack them: train, validation, test
def test_returns_the_masks_in_order(stand_ins):
    _, train_mask, val_mask, test_mask, _, _ = train_module.prepare_data()

    masks = stand_ins["masks"]
    assert torch.equal(train_mask, masks["train"])
    assert torch.equal(val_mask, masks["validation"])
    assert torch.equal(test_mask, masks["test"])


# Checks that the class weights are counted from the training labels only, and that labels has every road type that
# occurs in the labelled data
def test_class_weights_use_training_labels_only(stand_ins):
    _, _, _, _, class_weights, labels = train_module.prepare_data()

    graph = stand_ins["graph"]
    train_mask = stand_ins["masks"]["train"]
    expected = compute_class_weights(graph["y"][train_mask], len(ROAD_CLASSES), config.CLASS_WEIGHT_POWER)
    assert torch.equal(class_weights, expected)
    # Road type 7 only occurs in the test set, so it must get no weight
    assert class_weights[7] == 0

    assert labels == labels_in_data(graph["y"], graph["mask"])
