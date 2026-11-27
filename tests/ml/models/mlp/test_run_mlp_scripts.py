# tests/ml/models/mlp/test_run_mlp_scripts.py

"""Test the MLP validation and final-test scripts without running real training."""

"""Imports"""
import runpy
import pytest
import torch
import src.ml.evaluation as evaluation_module
import src.ml.models.mlp.train_mlp as train_module
from src.ml import config

"""Check that each MLP runner evaluates the correct data partition."""
# pytest.mark.parametrize() runs this test with two different sets of arguments.
# The development script must evaluate validation data, while the final script must evaluate test data.
@pytest.mark.parametrize(
    ("module_name", "evaluation_partition", "partition_name"),
    [
        ("src.ml.models.mlp.dev_run_mlp", "validation", "Validation"),
        ("src.ml.models.mlp.final_testrun_mlp", "test", "Test"),
    ],
)
def test_run_script_scores_the_correct_partition(
    monkeypatch, capsys, fake_database, module_name, evaluation_partition, partition_name,
):
    # monkeypatch temporarily replaces functions and settings during the test.
    # capsys captures printed output, while pytest supplies the remaining arguments from the cases above.

    # Create nine artificial road segments with two arbitrary features each.
    # torch.ones((9, 2)) creates a tensor with nine rows and two columns, all containing 1.0.
    # y contains the road-type labels for each segment.
    # Validation labels [0, 1, 7] differ from test labels [7, 1, 0], allowing us to detect incorrect partition usage.
    graph = {
        "x": torch.ones((9, 2)),
        "y": torch.tensor([0, 1, 7, 0, 1, 7, 7, 1, 0]),
    }

    # Divide the nine segments into three fixed partitions:
    # Training: segments 0-2, validation: segments 3-5, and testing: segments 6-8.
    # True means the segment belongs to the corresponding partition.
    train_mask = torch.tensor([True] * 3 + [False] * 6)
    val_mask = torch.tensor([False] * 3 + [True] * 3 + [False] * 3)
    test_mask = torch.tensor([False] * 6 + [True] * 3)

    # Define the road types and class weights normally returned by prepare_data().
    # Record the seeds used during training and the labels and predictions passed to evaluation.
    labels = [0, 1, 7]
    weights = torch.ones(8)
    seeds_seen = []
    evaluated_labels = []
    evaluated_predictions = []

    # Use two predefined seeds instead of the full set of model seeds.
    # Replace prepare_data() with a lambda that returns our artificial graph and fixed partitions.
    # This avoids loading and processing the Beijing dataset.
    monkeypatch.setattr(config, "MODEL_SEEDS", [0, 1])
    monkeypatch.setattr(train_module, "prepare_data", lambda: (
        graph, train_mask, val_mask, test_mask, weights, labels
    ))

    # Replace the real train_mlp() function with predictable behavior.
    # Instead of training a model, fake_train() checks its inputs and returns perfect predictions.
    def fake_train(prepared_graph, training, validation, class_weights, used_labels, seed, history=None):
        # Verify that training receives the correct graph, masks, class weights, and road-type labels.
        # The test mask must never be passed to training or used for selecting the best epoch.
        assert prepared_graph is graph
        assert torch.equal(training, train_mask)
        assert torch.equal(validation, val_mask)
        assert class_weights is weights
        assert used_labels == labels

        # Record the current seed and return the true labels as artificial predictions.
        # clone() creates a separate tensor, while 3 represents a fixed best epoch.
        seeds_seen.append(seed)
        return graph["y"].clone(), 3

    # Temporarily replace train_mlp() with fake_train() to avoid running actual model training.
    monkeypatch.setattr(train_module, "train_mlp", fake_train)

    # Store the original evaluate() function so we can still calculate the real evaluation metrics.
    # record_evaluation() acts as a spy: it records the inputs before calling the original function.
    real_evaluate = evaluation_module.evaluate

    def record_evaluation(true_labels, predictions, used_labels, **kwargs):
        # Record the labels and predictions passed to evaluation.
        # **kwargs collects additional keyword arguments, such as class_names.
        evaluated_labels.append(true_labels.clone())
        evaluated_predictions.append(predictions.clone())

        # Call the real evaluation function so the runner can calculate and display its results.
        return real_evaluate(true_labels, predictions, used_labels, **kwargs)

    # Temporarily replace evaluate() with our spy to check which partition each script evaluates.
    monkeypatch.setattr(evaluation_module, "evaluate", record_evaluation)

    # runpy.run_module() executes the selected Python module as if it were run directly.
    # run_name="__main__" ensures that its main execution block runs.
    # Since data preparation and training are mocked, no real dataset or model training is needed.
    runpy.run_module(module_name, run_name="__main__")

    # Select the partition that the current script is supposed to evaluate.
    # The development script must use validation data, while the final script must use test data.
    target_mask = val_mask if evaluation_partition == "validation" else test_mask

    # Verify that training was executed once for each predefined seed.
    # Both runs must use the same training and validation partitions.
    assert seeds_seen == [0, 1]

    # Verify that evaluation was performed once per seed using the correct partition.
    # all() returns True only if every recorded tensor matches the expected labels or predictions.
    assert len(evaluated_labels) == 2
    assert all(torch.equal(y, graph["y"][target_mask]) for y in evaluated_labels)
    assert all(torch.equal(y, graph["y"][target_mask]) for y in evaluated_predictions)

    # capsys.readouterr().out retrieves the text printed during the test.
    # Verify that the summary identifies the correct partition as Validation or Test.
    assert partition_name in capsys.readouterr().out

    # The run is saved in the database (fake_database replaces the real one): the split once, then one MLP run per
    # seed with the scores of the partition this script evaluates, and nothing failed.
    assert len(fake_database["splits"]) == 1
    assert [(started[0], started[1]) for started in fake_database["started"]] == [("MLP", 0), ("MLP", 1)]
    partition = "val" if evaluation_partition == "validation" else "test"
    assert [list(finished[1]) for finished in fake_database["finished"]] == [[partition], [partition]]
    assert fake_database["failed"] == []
