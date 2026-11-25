# src/ml/evaluation.py

import statistics
import warnings

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

# The coarse road categories used for grouped evaluation (report Table 13), following the OSM road hierarchy.
# Link types are not listed: they are merged into their parent type when the labels are made (src/db/create_road_type_labels.py).
# Road types that are in no category are left out of grouped evaluation.
ROAD_GROUPS = {
    "highways": ["motorway", "trunk"],
    "main_roads": ["primary", "secondary", "tertiary", "unclassified"],
    "minor_roads": [ "residential", "living_street"],
}


def labels_in_data(y, mask):
    """
    Returns the road type indices that occur among the labelled segments (mask = True), sorted.

    Use this as the labels argument of evaluate(), so every model is scored over the same road types. Road types in
    the class list that have no segments at all (e.g. "service" in the Beijing data) are left out, because their F1
    would always be 0 and only pull macro-F1 down.
    """
    return sorted(y[mask].unique().tolist())


def majority_class_predictions(train_labels, num_segments):
    """
    Majority-class baseline (report section "Accuracy Paradox"): predicts the most frequent road type of the TRAINING
    segments for every segment, ignoring the input. Score it with evaluate() on the same segments as the models.

    It is a lower bound: its accuracy is the share of the most frequent road type, while its F1 is 0 for every other
    road type, so its macro-F1 is very low. A model that does not clearly beat it on macro-F1 has learned little.

    train_labels: The labels of the training segments only, so the majority class is never taken from validation or
                  test labels.
    num_segments: How many predictions to return, e.g. the number of segments in the graph.

    Returns a tensor of length num_segments with the same road type index everywhere.
    """
    # bincount counts how often each road type occurs, argmax picks the most frequent (the lowest index on a tie)
    majority_class = torch.bincount(train_labels).argmax()
    return torch.full((num_segments,), majority_class.item(), dtype=torch.long)


def _scores_per_label(y_true, y_pred, labels, names):
    """
    Precision, recall, F1 and support of each label, as a dictionary keyed by the names (same order as labels).
    Also returns the F1 values as an array, so the caller can average them into macro-F1.
    """
    # zero_division=0: a label that is never predicted gets precision (and F1) 0 instead of a warning
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None,
                                                                     zero_division=0)
    scores = {}
    for i, name in enumerate(names):
        scores[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
    return scores, f1


def grouped_evaluation(y_true, y_pred, class_names, groups=ROAD_GROUPS):
    """
    Scores over coarse road categories instead of single road types (report section "Grouped Evaluation").

    Both the true and the predicted road type are mapped to their category before scoring, so mixing up two road
    types in the same category (e.g. secondary and primary) is not counted as an error. Segments whose true road type
    is in no category are left out. A prediction of a road type that is in no category counts as wrong (it is then
    also missing from the confusion matrix, so a row can add up to less than its support).

    Returns a dictionary with:
        macro_f1:         Mean F1 over the categories, each category counting equally
        per_category:     Precision, recall, F1 and number of segments (support) of each category
        confusion_matrix: Row i, column j = number of segments of category i predicted as category j, in the order
                          of groups

    If grouped macro-F1 is high while the fine-grained macro-F1 is low, the model tells the categories apart but
    struggles within them. The per-category scores and confusion matrix show which categories that holds for.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # For each road type index, the index of its category, or -1 if it is in no category
    category_of_name = {}
    for category, names in enumerate(groups.values()):
        for name in names:
            category_of_name[name] = category
    category_of_label = np.array([category_of_name.get(name, -1) for name in class_names])

    true_category = category_of_label[y_true]
    pred_category = category_of_label[y_pred]
    keep = true_category != -1
    true_category = true_category[keep]
    pred_category = pred_category[keep]

    categories = list(range(len(groups)))
    per_category, f1 = _scores_per_label(true_category, pred_category, categories, list(groups.keys()))
    return {
        "macro_f1": float(f1.mean()),
        "per_category": per_category,
        "confusion_matrix": confusion_matrix(true_category, pred_category, labels=categories),
    }


def grouped_macro_f1(y_true, y_pred, class_names, groups=ROAD_GROUPS):
    """
    Macro-F1 over coarse road categories instead of single road types. Only the macro-F1 of grouped_evaluation().
    """
    return grouped_evaluation(y_true, y_pred, class_names, groups)["macro_f1"]


def evaluate(y_true, y_pred, labels, class_names=None):
    """
    Scores predicted road types against the true ones. Shared by all models, so their results can be compared.

    y_true:      True road type index of each evaluated segment (tensor or array)
    y_pred:      Predicted road type index of each evaluated segment, in the same order
    labels:      The road type indices that count in macro-F1 and the confusion matrix. Always pass the same list for
                 every model. If it were left to scikit-learn, it would use the road types that happen to occur in
                 y_true or y_pred, so a model that wrongly predicts a rare type once would be averaged over one more
                 road type than a model that never predicts it.
    class_names: Optional list of road type names (e.g. ROAD_CLASSES). Used as keys of per_class, and needed for
                 the grouped scores.

    Only pass the segments you evaluate on (e.g. the test segments), never unlabelled segments (y = -1).

    Returns a dictionary with:
        macro_f1:         Mean F1 over the road types in labels, each road type counting equally (main metric)
        accuracy:         Share of segments whose road type is predicted correctly
        per_class:        Precision, recall, F1 and number of segments (support) of each road type in labels, in the
                          order of labels
        confusion_matrix: Row i, column j = number of segments of road type labels[i] predicted as labels[j]
        missing_labels:   Road types in labels with no segments among the evaluated ones. Their F1 is always 0 and
                          pulls macro-F1 down, which can happen with a spatial split. A warning is also shown.
        grouped_macro_f1: Macro-F1 over the coarse categories in ROAD_GROUPS (only if class_names is given)
        grouped:          All scores of grouped_evaluation() (only if class_names is given)
    """
    # scikit-learn works on numpy arrays, so convert tensors first
    if hasattr(y_true, "cpu"):
        y_true = y_true.cpu().numpy()
    if hasattr(y_pred, "cpu"):
        y_pred = y_pred.cpu().numpy()

    names = [class_names[label] for label in labels] if class_names is not None else list(labels)
    per_class, f1 = _scores_per_label(y_true, y_pred, labels, names)

    missing_labels = [name for name, scores in per_class.items() if scores["support"] == 0]
    if missing_labels:
        # stacklevel=2 points the warning at the caller, so it is shown once per calling line instead of every epoch
        warnings.warn(f"No segments of road type(s) {missing_labels} among the evaluated segments. Their F1 is 0 "
                      f"and lowers macro-F1.", stacklevel=2)

    result = {
        "macro_f1": float(f1.mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
        "missing_labels": missing_labels,
    }
    if class_names is not None:
        grouped = grouped_evaluation(y_true, y_pred, class_names)
        result["grouped_macro_f1"] = grouped["macro_f1"]
        result["grouped"] = grouped
    return result


def _mean_and_std(values):
    """
    Mean and standard deviation of a list of values. The standard deviation is None for a single value, since the
    spread of one run is unknown (statistics.stdev would raise an error).
    """
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else None,
    }


def _summarize_scores(score_dicts):
    """
    Combines per-class (or per-category) scores of several runs. Precision, recall and F1 become mean and standard
    deviation. Support is the same in every run (same split), so it is kept as a single number.
    """
    summary = {}
    for name in score_dicts[0]:
        summary[name] = {
            metric: _mean_and_std([scores[name][metric] for scores in score_dicts])
            for metric in ("precision", "recall", "f1")
        }
        summary[name]["support"] = score_dicts[0][name]["support"]
    return summary


def summarize_runs(results):
    """
    Combines the evaluate() results of several runs (e.g. one per seed in config.MODEL_SEEDS) into mean and standard
    deviation, so every model reports its results in the same way (report: Evaluation Protocol).

    results: List of dictionaries returned by evaluate(), all scored on the same segments with the same labels.

    Returns a dictionary with:
        runs:             Number of runs
        macro_f1, accuracy, grouped_macro_f1:
                          {"mean": ..., "std": ...}. std is None if there is only one run.
        per_class:        For each road type, mean and std of precision, recall and F1, plus its support
        confusion_matrix: Mean confusion matrix over the runs
        grouped:          per_category (as per_class) and the mean confusion_matrix of the categories
                          (grouped_macro_f1 and grouped only if the results contain grouped scores)
    """
    if not results:
        raise ValueError("summarize_runs needs at least one evaluate() result")

    summary = {
        "runs": len(results),
        "macro_f1": _mean_and_std([result["macro_f1"] for result in results]),
        "accuracy": _mean_and_std([result["accuracy"] for result in results]),
        "per_class": _summarize_scores([result["per_class"] for result in results]),
        "confusion_matrix": np.mean([result["confusion_matrix"] for result in results], axis=0),
    }
    if "grouped" in results[0]:
        summary["grouped_macro_f1"] = _mean_and_std([result["grouped_macro_f1"] for result in results])
        summary["grouped"] = {
            "per_category": _summarize_scores([result["grouped"]["per_category"] for result in results]),
            "confusion_matrix": np.mean([result["grouped"]["confusion_matrix"] for result in results], axis=0),
        }
    return summary


def _format_mean_std(score, decimals=4):
    # "0.8123 ± 0.0120", or only the mean if there is no standard deviation (a single run)
    if score["std"] is None:
        return f"{score['mean']:.{decimals}f}"
    return f"{score['mean']:.{decimals}f} ± {score['std']:.{decimals}f}"


def format_confusion_matrix(matrix, names):
    """
    The confusion matrix as a text table with road type (or category) names. Rows are the true types, columns the
    predicted ones. names must be in the order of the matrix, e.g. list(result["per_class"].keys()).
    Works for the counts of one run and for the mean matrix of summarize_runs() (shown with one decimal).
    """
    matrix = np.asarray(matrix)
    whole_numbers = np.all(matrix == np.round(matrix))
    cells = [[f"{value:.0f}" if whole_numbers else f"{value:.1f}" for value in row] for row in matrix]

    names = [str(name) for name in names]
    name_width = max(len("true \\ predicted"), *(len(name) for name in names))
    column_width = max(*(len(name) for name in names), *(len(cell) for row in cells for cell in row))

    lines = ["true \\ predicted".ljust(name_width) + " " + " ".join(name.rjust(column_width) for name in names)]
    for name, row in zip(names, cells):
        lines.append(name.ljust(name_width) + " " + " ".join(cell.rjust(column_width) for cell in row))
    return "\n".join(lines)


def _format_score_table(scores, first_column):
    # One row per road type (or category) with precision, recall, F1 and support
    name_width = max(len(first_column), *(len(str(name)) for name in scores))
    lines = [f"{first_column.ljust(name_width)}  {'precision':>15}  {'recall':>15}  {'F1':>15}  {'support':>7}"]
    for name, score in scores.items():
        lines.append(f"{str(name).ljust(name_width)}  {_format_mean_std(score['precision'], 3):>15}  "
                     f"{_format_mean_std(score['recall'], 3):>15}  {_format_mean_std(score['f1'], 3):>15}  "
                     f"{score['support']:>7}")
    return "\n".join(lines)


def format_summary(summary, set_name="Validation"):
    """
    The result of summarize_runs() as readable text: the main scores, the per-class table, the confusion matrix and,
    if present, the grouped scores. set_name says which segments were scored (e.g. "Validation" or "Test").
    """
    lines = [f"Results over {summary['runs']} run(s) (mean ± standard deviation):",
             f"{set_name} macro-F1: {_format_mean_std(summary['macro_f1'])}"]
    if "grouped_macro_f1" in summary:
        lines.append(f"{set_name} grouped macro-F1: {_format_mean_std(summary['grouped_macro_f1'])}")
    lines.append(f"{set_name} accuracy: {_format_mean_std(summary['accuracy'])}")

    lines += ["", _format_score_table(summary["per_class"], "road type"),
              "", "Confusion matrix (mean over runs):",
              format_confusion_matrix(summary["confusion_matrix"], list(summary["per_class"].keys()))]

    if "grouped" in summary:
        per_category = summary["grouped"]["per_category"]
        lines += ["", _format_score_table(per_category, "category"),
                  "", "Grouped confusion matrix (mean over runs):",
                  format_confusion_matrix(summary["grouped"]["confusion_matrix"], list(per_category.keys()))]
    return "\n".join(lines)
