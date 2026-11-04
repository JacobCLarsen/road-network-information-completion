# src/ml/evaluation.py

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

# The coarse road categories used for grouped evaluation (report Table 12), following the OSM road hierarchy.
# The report merges link types into their parent type before evaluating. They are listed here as well, under the
# category of their parent, so grouped evaluation also works while link types are still separate classes.
# Road types that are in no category (e.g. "road" or "busway") are left out of grouped evaluation.
ROAD_GROUPS = {
    "highways": ["motorway", "trunk", "motorway_link", "trunk_link"],
    "main_roads": ["primary", "secondary", "tertiary", "primary_link", "secondary_link", "tertiary_link"],
    "minor_roads": ["unclassified", "residential", "living_street"],
}


def labels_in_data(y, mask):
    """
    Returns the road type indices that occur among the labelled segments (mask = True), sorted.

    Use this as the labels argument of evaluate(), so every model is scored over the same road types. Road types in
    the class list that have no segments at all (e.g. "service" in the Beijing data) are left out, because their F1
    would always be 0 and only pull macro-F1 down.
    """
    return sorted(y[mask].unique().tolist())


def grouped_macro_f1(y_true, y_pred, class_names, groups=ROAD_GROUPS):
    """
    Macro-F1 over coarse road categories instead of single road types (report section "Grouped Evaluation").

    Both the true and the predicted road type are mapped to their category before scoring, so mixing up two road
    types in the same category (e.g. secondary and primary) is not counted as an error. Segments whose true road type
    is in no category are left out. A prediction of a road type that is in no category counts as wrong.
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

    categories = list(range(len(groups)))
    return float(f1_score(true_category[keep], pred_category[keep], labels=categories, average="macro",
                          zero_division=0))


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
                 grouped_macro_f1.

    Only pass the segments you evaluate on (e.g. the test segments), never unlabelled segments (y = -1).

    Returns a dictionary with:
        macro_f1:         Mean F1 over the road types in labels, each road type counting equally (main metric)
        accuracy:         Share of segments whose road type is predicted correctly
        per_class:        Precision, recall, F1 and number of segments (support) of each road type in labels
        confusion_matrix: Row i, column j = number of segments of road type labels[i] predicted as labels[j]
        grouped_macro_f1: Macro-F1 over the coarse categories in ROAD_GROUPS (only if class_names is given)
    """
    # scikit-learn works on numpy arrays, so convert tensors first
    if hasattr(y_true, "cpu"):
        y_true = y_true.cpu().numpy()
    if hasattr(y_pred, "cpu"):
        y_pred = y_pred.cpu().numpy()

    # zero_division=0: a road type that is never predicted gets precision (and F1) 0 instead of a warning
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None,
                                                                     zero_division=0)

    per_class = {}
    for i, label in enumerate(labels):
        name = class_names[label] if class_names is not None else label
        per_class[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    result = {
        "macro_f1": float(f1.mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
    }
    if class_names is not None:
        result["grouped_macro_f1"] = grouped_macro_f1(y_true, y_pred, class_names)
    return result
