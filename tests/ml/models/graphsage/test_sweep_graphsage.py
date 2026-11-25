# Tests for sweep_graphsage.py, without real data or training. The sweep's main part is under
# if __name__ == "__main__", so it is run with runpy, as if started from the command line. runpy runs the file again
# from the top, so SWEEP, SWEEP_SEEDS and RESULTS_FILE always have their values from the file: the tests change the
# working folder instead, so the results file is written to a temporary folder.

import csv
import runpy

import pytest
import torch
import src.ml.models.graphsage.sweep_graphsage as sweep_module
import src.ml.models.graphsage.train_graphsage as train_module
from src.ml.models.graphsage import graphsage_config as sage_config

SWEEP_MODULE = "src.ml.models.graphsage.sweep_graphsage"


# 12 segments: 0-5 train, 6-8 validation, 9-11 test. The test segments have road type 7 only, so any use of them shows.
@pytest.fixture
def sweep_graph():
    graph = {
        "x": torch.ones((12, 2)),
        "y": torch.tensor([0, 0, 0, 1, 1, 1, 0, 1, 0, 7, 7, 7]),
        "mask": torch.ones(12, dtype=torch.bool),
    }
    train_mask = torch.tensor([True] * 6 + [False] * 6)
    val_mask = torch.tensor([False] * 6 + [True] * 3 + [False] * 3)
    test_mask = torch.tensor([False] * 9 + [True] * 3)
    return graph, train_mask, val_mask, test_mask, torch.ones(8), [0, 1, 7]


# The settings the sweep changes, so the tests can check they are put back afterwards
def current_settings():
    return {name: getattr(sage_config, name) for name in sweep_module.SWEEP}


# Checks the score format: mean only when there is no standard deviation (one seed), otherwise mean ± std
def test_format_score_with_and_without_standard_deviation():
    assert sweep_module._format_score({"mean": 0.5, "std": None}) == "0.5000"
    assert sweep_module._format_score({"mean": 0.5, "std": 0.0125}) == "0.5000 ± 0.0125"


# Runs the whole sweep with a stand-in for training that only predicts the validation set correctly when weight decay
# is 5e-4. Checks that the sweep:
# - only trains on the training mask and chooses on the validation mask, never the test mask
# - trains every new combination once per seed in SWEEP_SEEDS, and never a combination twice
# - finds weight decay 5e-4 as the best value
# - saves every combination to the CSV, best first
# - puts graphsage_config's settings back afterwards
def test_sweep_finds_best_setting_and_saves_sorted_results(monkeypatch, tmp_path, capsys, sweep_graph):
    graph, train_mask, val_mask, test_mask, weights, labels = sweep_graph
    original = current_settings()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train_module, "prepare_data",
                        lambda: (graph, train_mask, val_mask, test_mask, weights, labels))

    train_calls = []

    def fake_train(prepared_graph, training, validation, class_weights, used_labels, seed):
        assert prepared_graph is graph
        assert torch.equal(training, train_mask)
        assert torch.equal(validation, val_mask)
        assert class_weights is weights
        assert used_labels == labels
        # The sweep sets the settings in graphsage_config before training, so train_graphsage reads them there
        train_calls.append((tuple(getattr(sage_config, name) for name in sweep_module.SWEEP), seed))

        predictions = torch.zeros_like(graph["y"])
        if sage_config.WEIGHT_DECAY == 5e-4:
            predictions[val_mask] = graph["y"][val_mask]
        return predictions, seed + 10

    monkeypatch.setattr(train_module, "train_graphsage", fake_train)

    runpy.run_module(SWEEP_MODULE, run_name="__main__")

    # Every combination is trained with every sweep seed, in order, and no combination twice
    combinations = list(dict.fromkeys(values for values, _ in train_calls))
    assert len(train_calls) == len(combinations) * len(sweep_module.SWEEP_SEEDS)
    for values in combinations:
        assert [seed for trained, seed in train_calls if trained == values] == sweep_module.SWEEP_SEEDS

    # One round tries every value of every setting, but the current best value of a setting was already trained in
    # the step before, so each setting after the first adds one combination fewer than it has values
    values_per_setting = [len(values) for values in sweep_module.SWEEP.values()]
    assert len(combinations) == values_per_setting[0] + sum(count - 1 for count in values_per_setting[1:])

    # The settings are put back, so the sweep does not change later runs in the same Python session
    assert current_settings() == original

    output = capsys.readouterr().out
    assert "  WEIGHT_DECAY = 0.0005" in output.split("Best settings found")[1]

    # The CSV has one row per combination, best first, with the best epochs of every seed
    with (tmp_path / sweep_module.RESULTS_FILE).open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == len(combinations)
    assert float(rows[0]["WEIGHT_DECAY"]) == 5e-4
    means = [float(row["macro_f1_mean"]) for row in rows]
    assert means == sorted(means, reverse=True)
    assert rows[0]["best_epochs"] == " ".join(str(seed + 10) for seed in sweep_module.SWEEP_SEEDS)


# Checks that the settings are put back also when training crashes halfway, so a failed sweep does not leave
# graphsage_config with the values of the combination it was trying
def test_sweep_restores_settings_when_training_fails(monkeypatch, tmp_path, sweep_graph):
    graph, train_mask, val_mask, test_mask, weights, labels = sweep_graph
    original = current_settings()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train_module, "prepare_data",
                        lambda: (graph, train_mask, val_mask, test_mask, weights, labels))

    def fail_training(*args):
        raise RuntimeError("Training failed")

    monkeypatch.setattr(train_module, "train_graphsage", fail_training)

    with pytest.raises(RuntimeError, match="Training failed"):
        runpy.run_module(SWEEP_MODULE, run_name="__main__")

    assert current_settings() == original
    assert not (tmp_path / sweep_module.RESULTS_FILE).exists()


# Checks that _train_combination sets the settings it is given before training, scores on the validation set and
# summarises over the seeds
def test_train_combination_applies_settings_and_scores_validation(monkeypatch, sweep_graph):
    graph, train_mask, val_mask, _, weights, _ = sweep_graph
    # Only the road types in the validation set, so correct predictions give a macro-F1 of exactly 1
    labels = [0, 1]
    monkeypatch.setattr(sweep_module, "SWEEP", {"DROPOUT": [0.3], "WORKING_SPACE": [7]})
    monkeypatch.setattr(sweep_module, "SWEEP_SEEDS", [4, 5])
    # monkeypatch puts these back after the test, also though _train_combination changes them
    monkeypatch.setattr(sage_config, "DROPOUT", sage_config.DROPOUT)
    monkeypatch.setattr(sage_config, "WORKING_SPACE", sage_config.WORKING_SPACE)

    seen = []

    def fake_train(prepared_graph, training, validation, class_weights, used_labels, seed):
        seen.append((sage_config.DROPOUT, sage_config.WORKING_SPACE, seed))
        predictions = torch.zeros_like(graph["y"])
        predictions[val_mask] = graph["y"][val_mask]
        return predictions, 1

    monkeypatch.setattr(sweep_module, "train_graphsage", fake_train)

    row = sweep_module._train_combination((graph, train_mask, val_mask, weights, labels), (0.3, 7))

    assert seen == [(0.3, 7, 4), (0.3, 7, 5)]
    assert row["values"] == (0.3, 7)
    assert row["best_epochs"] == [1, 1]
    assert row["summary"]["macro_f1"]["mean"] == 1.0
