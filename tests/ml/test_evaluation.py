import pytest
import torch
from src.ml.evaluation import evaluate, grouped_macro_f1, labels_in_data, majority_class_predictions, summarize_runs


def test_labels_in_data_skips_unlabelled_and_missing_types():
    y = torch.tensor([2, 0, 2, -1, 5])
    mask = torch.tensor([True, True, True, False, True])
    assert labels_in_data(y, mask) == [0, 2, 5]


def test_perfect_predictions():
    y = torch.tensor([0, 1, 1, 2])
    result = evaluate(y, y, labels=[0, 1, 2])
    assert result["macro_f1"] == 1.0
    assert result["accuracy"] == 1.0


def test_scores_by_hand():
    # Road type 0: 2 of 2 found, 1 wrong guess   -> precision 2/3, recall 1   -> F1 0.8
    # Road type 1: 1 of 2 found, no wrong guesses -> precision 1,   recall 1/2 -> F1 2/3
    y_true = torch.tensor([0, 0, 1, 1])
    y_pred = torch.tensor([0, 0, 0, 1])
    result = evaluate(y_true, y_pred, labels=[0, 1])

    assert result["accuracy"] == 0.75
    assert result["per_class"][0]["precision"] == pytest.approx(2 / 3)
    assert result["per_class"][0]["recall"] == 1.0
    assert result["per_class"][0]["f1"] == pytest.approx(0.8)
    assert result["per_class"][1]["precision"] == 1.0
    assert result["per_class"][1]["recall"] == 0.5
    assert result["per_class"][1]["f1"] == pytest.approx(2 / 3)
    assert result["per_class"][1]["support"] == 2
    assert result["macro_f1"] == pytest.approx((0.8 + 2 / 3) / 2)
    assert result["confusion_matrix"].tolist() == [[2, 0], [1, 1]]


def test_wrong_guess_of_other_type_does_not_add_it_to_macro_f1():
    # Road type 7 is predicted once but is not in labels, so macro-F1 still averages over types 0 and 1 only
    y_true = torch.tensor([0, 0, 1, 1])
    y_pred = torch.tensor([0, 7, 1, 1])
    result = evaluate(y_true, y_pred, labels=[0, 1])
    assert list(result["per_class"].keys()) == [0, 1]


def test_grouped_macro_f1_ignores_mix_ups_within_a_category():
    names = ["motorway", "trunk", "primary", "secondary", "residential", "road"]
    # Every mistake stays inside its category (motorway/trunk, primary/secondary), so grouped macro-F1 is perfect
    y_true = [0, 1, 2, 3, 4]
    y_pred = [1, 0, 3, 2, 4]
    assert grouped_macro_f1(y_true, y_pred, names) == 1.0

    # Only highways, main roads and minor roads count: a "road" segment (in no category) is left out
    assert grouped_macro_f1(y_true + [5], y_pred + [0], names) == 1.0

    # A residential road predicted as a motorway is a mistake across categories, so the score drops
    assert grouped_macro_f1(y_true, [1, 0, 3, 2, 0], names) < 1.0


def test_majority_class_baseline_predicts_most_frequent_training_type():
    y_train = torch.tensor([2, 2, 0, 2, 1])
    assert majority_class_predictions(y_train, 3).tolist() == [2, 2, 2]


def test_majority_class_baseline_has_low_macro_f1_despite_reasonable_accuracy():
    # The accuracy paradox from the report: always guessing the frequent type gets 3 of 4 right, but finds no type 1
    y_true = torch.tensor([0, 0, 0, 1])
    result = evaluate(y_true, majority_class_predictions(torch.tensor([0, 0, 1]), 4), labels=[0, 1])
    assert result["accuracy"] == 0.75
    assert result["per_class"][1]["f1"] == 0.0
    # Type 0: precision 3/4, recall 1 -> F1 6/7. Type 1: F1 0. So macro-F1 is 3/7, well below the accuracy
    assert result["macro_f1"] == pytest.approx(3 / 7)


def test_summarize_runs_gives_mean_and_std_over_seeds():
    y_true = torch.tensor([0, 0, 1, 1])
    runs = [
        evaluate(y_true, y_true, labels=[0, 1]),
        evaluate(y_true, torch.tensor([0, 0, 0, 1]), labels=[0, 1]),
    ]
    summary = summarize_runs(runs)
    assert summary["accuracy"]["mean"] == pytest.approx((1.0 + 0.75) / 2)
    assert summary["accuracy"]["std"] == pytest.approx(0.125)
    assert summary["per_class_f1"][1]["mean"] == pytest.approx((1.0 + 2 / 3) / 2)
    assert "grouped_macro_f1" not in summary
