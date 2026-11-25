# tests/ml/models/mlp/test_prepare_mlp.py

"""Test MLP data preparation without loading the full Beijing dataset."""

"""Imports"""
import torch
import src.ml.models.mlp.train_mlp as train_module
from unittest.mock import Mock
from src.ml import config
from src.ml.class_weights import compute_class_weights
from src.ml.data.road_graph_builder import ROAD_CLASSES
from src.ml.evaluation import labels_in_data

"""Check that data preparation preserves the splits and calculates class weights using training labels only."""
def test_prepare_data_keeps_splits_and_uses_training_labels_only(monkeypatch):
    # monkeypatch is a pytest fixture that temporarily replaces functions or classes.
    # This allows us to test prepare_data() without loading the Beijing dataset.

    # Create six artificial road segments using torch.tensor(), which converts lists into PyTorch tensors.
    # x contains one arbitrary feature per segment, y contains road-type labels (-1 means unlabeled),
    # and mask indicates which segments have usable labels.
    # Road type 7 only appears in the test set, allowing us to detect label leakage.
    graph = {
        "x": torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]),
        "y": torch.tensor([0, 1, 2, 7, 7, -1]),
        "mask": torch.tensor([True, True, True, True, True, False]),
    }

    # Each position represents a segment: training uses 0-1, validation uses 2,
    # and testing uses 3-4. Segment 5 is unlabeled and excluded from all partitions.
    # True means the segment belongs to the corresponding partition.
    masks = {
        "train": torch.tensor([True, True, False, False, False, False]),
        "validation": torch.tensor([False, False, True, False, False, False]),
        "test": torch.tensor([False, False, False, True, True, False]),
    }

    # Create six empty placeholder objects instead of real road segments.
    # object() creates an empty object, while _ ignores the counter from range(6).
    segments = [object() for _ in range(6)]

    # Mock() creates substitute objects whose methods return predefined values.
    # The file loader returns our placeholder segments, and the graph builder returns our artificial graph.
    loader = Mock()
    loader.get_segments.return_value = segments

    builder = Mock()
    builder.get_static_components.return_value = graph

    # Make the mocked constructors return our objects and the split function return our fixed masks.
    loader_class = Mock(return_value = loader)
    builder_class = Mock(return_value = builder)
    split_function = Mock(return_value = masks)

    # Replace the real dependencies inside train_mlp.py during this test.
    # monkeypatch.setattr() automatically restores them afterward.
    monkeypatch.setattr(train_module, "ShapefileMapLoader", loader_class)
    monkeypatch.setattr(train_module, "RoadGraphBuilder", builder_class)
    monkeypatch.setattr(train_module, "make_split_masks", split_function)

    # Execute the real prepare_data() function using the mocked dependencies.
    prepared_graph, train_mask, val_mask, test_mask, weights, labels = train_module.prepare_data()

    # assert_called_once_with() verifies that each mocked function was called exactly once
    # with the expected arguments.
    loader_class.assert_called_once_with(config.DATA_DIR)
    builder_class.assert_called_once_with(segments)
    split_function.assert_called_once_with(graph)

    # Standardization must use the training mask so validation and test data cannot influence its statistics.
    builder.standardize_features.assert_called_once_with(segments, masks["train"])

    # Verify that the original graph and the correct partition masks are returned.
    # torch.equal() compares tensor values, while 'is' checks that both variables reference the same object.
    assert prepared_graph is graph
    assert torch.equal(train_mask, masks["train"])
    assert torch.equal(val_mask, masks["validation"])
    assert torch.equal(test_mask, masks["test"])

    # Calculate the expected class weights using only training labels.
    # Including validation or test labels would produce different weights.
    expected_weights = compute_class_weights(
        graph["y"][train_mask],
        len(ROAD_CLASSES),
        config.CLASS_WEIGHT_POWER,
    )
    assert torch.equal(weights, expected_weights)

    # Road type 7 only occurs in the test set, so its training weight must be zero.
    assert weights[7] == 0

    # labels_in_data() returns the road types with usable labels for evaluation.
    expected_labels = labels_in_data(graph["y"], graph["mask"])
    assert labels == expected_labels
