# Tests for dev_run_graphsage.py and final_testrun_graphsage.py, without real data or training. prepare_data and
# train_graphsage are replaced by stand-ins, and the scripts are run as if started from the command line.

import runpy

import pytest
import torch
import src.ml.evaluation as evaluation_module
import src.ml.models.graphsage.train_graphsage as train_module
from src.ml import config


# Checks that the dev run scores the validation set and the final run the test set, once per seed, and that neither
# passes the test mask to training. The validation labels [0, 1, 7] and test labels [7, 1, 0] differ, so scoring the
# wrong set is caught.
@pytest.mark.parametrize(
    ("module_name", "scored_set", "set_name"),
    [
        ("src.ml.models.graphsage.dev_run_graphsage", "validation", "Validation"),
        ("src.ml.models.graphsage.final_testrun_graphsage", "test", "Test"),
    ],
)
def test_run_script_scores_the_correct_set(monkeypatch, capsys, module_name, scored_set, set_name):
    graph = {"x": torch.ones((9, 2)), "y": torch.tensor([0, 1, 7, 0, 1, 7, 7, 1, 0])}
    train_mask = torch.tensor([True] * 3 + [False] * 6)
    val_mask = torch.tensor([False] * 3 + [True] * 3 + [False] * 3)
    test_mask = torch.tensor([False] * 6 + [True] * 3)
    labels = [0, 1, 7]
    weights = torch.ones(8)

    monkeypatch.setattr(config, "MODEL_SEEDS", [0, 1])
    monkeypatch.setattr(train_module, "prepare_data",
                        lambda: (graph, train_mask, val_mask, test_mask, weights, labels))

    seeds_seen = []

    # Instead of training: check what training gets, then predict every road type correctly
    def fake_train(prepared_graph, training, validation, class_weights, used_labels, seed):
        assert prepared_graph is graph
        assert torch.equal(training, train_mask)
        # The test mask must never reach training, not even in the final run
        assert torch.equal(validation, val_mask)
        assert class_weights is weights
        assert used_labels == labels
        seeds_seen.append(seed)
        return graph["y"].clone(), 3

    monkeypatch.setattr(train_module, "train_graphsage", fake_train)

    # A spy on evaluate: write down what it scores, then compute the real scores so the summary can be printed
    scored_labels = []
    real_evaluate = evaluation_module.evaluate

    def spy_evaluate(true_labels, predictions, used_labels, **kwargs):
        scored_labels.append(true_labels.clone())
        return real_evaluate(true_labels, predictions, used_labels, **kwargs)

    monkeypatch.setattr(evaluation_module, "evaluate", spy_evaluate)

    runpy.run_module(module_name, run_name="__main__")

    target_mask = val_mask if scored_set == "validation" else test_mask
    assert seeds_seen == [0, 1]
    assert len(scored_labels) == 2
    assert all(torch.equal(true_labels, graph["y"][target_mask]) for true_labels in scored_labels)

    output = capsys.readouterr().out
    # The summary names the set it scored, and each seed line shows the best epoch the stand-in returned
    assert set_name in output
    assert "best epoch 3" in output
