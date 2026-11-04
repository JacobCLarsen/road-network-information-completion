import statistics

import pytest
import torch
from src.ml.evaluation import (evaluate, format_confusion_matrix, format_summary, grouped_evaluation, grouped_macro_f1,
                               labels_in_data, majority_class_predictions, summarize_runs)


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


def test_grouped_evaluation_per_category_and_confusion_matrix():
    names = ["motorway", "trunk", "primary", "secondary", "residential"]
    # A secondary road predicted as residential: main roads lose a segment to minor roads
    y_true = [0, 1, 2, 3, 4]
    y_pred = [1, 0, 2, 4, 4]
    grouped = grouped_evaluation(y_true, y_pred, names)

    assert list(grouped["per_category"].keys()) == ["highways", "main_roads", "minor_roads"]
    assert grouped["per_category"]["highways"]["f1"] == 1.0
    assert grouped["per_category"]["main_roads"]["recall"] == 0.5
    assert grouped["per_category"]["minor_roads"]["precision"] == 0.5
    assert grouped["confusion_matrix"].tolist() == [[2, 0, 0], [0, 1, 1], [0, 0, 1]]
    assert grouped["macro_f1"] == grouped_macro_f1(y_true, y_pred, names)


def test_evaluate_includes_grouped_scores_with_class_names():
    names = ["motorway", "trunk", "primary"]
    y = torch.tensor([0, 1, 2])
    result = evaluate(y, y, labels=[0, 1, 2], class_names=names)
    assert result["grouped_macro_f1"] == result["grouped"]["macro_f1"]
    assert list(result["per_class"].keys()) == names


def test_evaluate_warns_about_road_types_without_segments():
    y = torch.tensor([0, 0, 1])
    with pytest.warns(UserWarning, match="lowers macro-F1"):
        result = evaluate(y, y, labels=[0, 1, 2])
    assert result["missing_labels"] == [2]


def test_summarize_runs_mean_and_std():
    y_true = torch.tensor([0, 0, 1, 1])
    runs = [evaluate(y_true, y_true, labels=[0, 1]),
            evaluate(y_true, torch.tensor([0, 0, 0, 1]), labels=[0, 1])]
    summary = summarize_runs(runs)

    assert summary["runs"] == 2
    assert summary["accuracy"]["mean"] == pytest.approx(0.875)
    assert summary["accuracy"]["std"] == pytest.approx(statistics.stdev([1.0, 0.75]))
    assert summary["per_class"][1]["recall"]["mean"] == pytest.approx(0.75)
    assert summary["per_class"][1]["support"] == 2
    assert summary["confusion_matrix"].tolist() == [[2, 0], [0.5, 1.5]]


def test_summarize_runs_with_one_run_has_no_std():
    y = torch.tensor([0, 1])
    summary = summarize_runs([evaluate(y, y, labels=[0, 1], class_names=["motorway", "residential"])])
    assert summary["macro_f1"] == {"mean": 1.0, "std": None}
    # Formatting a single run must not fail on the missing standard deviation
    assert "macro-F1: 1.0000" in format_summary(summary)


def test_format_summary_lists_scores_and_matrices():
    names = ["motorway", "trunk", "residential"]
    y = torch.tensor([0, 1, 2])
    summary = summarize_runs([evaluate(y, y, labels=[0, 1, 2], class_names=names)] * 2)
    text = format_summary(summary, set_name="Test")

    assert "Test macro-F1: 1.0000 ± 0.0000" in text
    assert "Test grouped macro-F1" in text
    assert "Grouped confusion matrix" in text
    assert "minor_roads" in text


def test_format_confusion_matrix():
    text = format_confusion_matrix([[2, 0], [1, 1]], ["motorway", "trunk"])
    lines = text.splitlines()
    assert lines[0].split()[-2:] == ["motorway", "trunk"]
    assert lines[1].split() == ["motorway", "2", "0"]
    assert lines[2].split() == ["trunk", "1", "1"]


def test_majority_class_predictions_uses_most_frequent_training_label():
    train_labels = torch.tensor([6, 6, 0, 6, 2])
    predictions = majority_class_predictions(train_labels, 4)
    assert predictions.tolist() == [6, 6, 6, 6]


def test_majority_class_baseline_scores():
    # Always predicting road type 1: accuracy is its share (3 of 4), but road type 0 gets F1 0.
    # Road type 1 has precision 3/4 and recall 1, so F1 6/7
    y_true = torch.tensor([1, 1, 1, 0])
    predictions = majority_class_predictions(torch.tensor([1, 1, 0]), len(y_true))
    result = evaluate(y_true, predictions, labels=[0, 1])
    assert result["accuracy"] == 0.75
    assert result["per_class"][0]["f1"] == 0.0
    assert result["macro_f1"] == pytest.approx((6 / 7) / 2)
