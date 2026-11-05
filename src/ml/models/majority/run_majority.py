# src/ml/models/majority/run_majority.py

# Majority-class baseline (report: Evaluation Protocol). Always predicts the most frequent road type of the training
# segments and scores that on the validation and test sets, so the candidate models can be compared against it.
#
# Unlike the other models it has no random starting weights and no settings to tune, so it is run once instead of once
# per seed (every seed would give the same result), and the test set can be scored right away: nothing is chosen on it.

from src.ml import config
from src.ml.data.file_loader import ShapefileMapLoader
from src.ml.data.road_graph_builder import RoadGraphBuilder, ROAD_CLASSES
from src.ml.data.split_graph import make_split_masks
from src.ml.evaluation import evaluate, format_summary, labels_in_data, majority_class_predictions, summarize_runs


if __name__ == "__main__":
    # The same graph, split and road types as the other models, so the scores can be compared. The features are not
    # standardised, because the baseline never looks at them.
    segments = ShapefileMapLoader(config.DATA_DIR).get_segments()
    graph = RoadGraphBuilder(segments).get_static_components()
    masks = make_split_masks(graph)
    labels = labels_in_data(graph["y"], graph["mask"])

    # The majority class is taken from the training segments only
    predictions = majority_class_predictions(graph["y"][masks["train"]], len(graph["y"]))
    print("Majority class:", ROAD_CLASSES[predictions[0].item()])

    for set_name, mask in (("Validation", masks["validation"]), ("Test", masks["test"])):
        result = evaluate(graph["y"][mask], predictions[mask], labels, class_names=ROAD_CLASSES)
        print()
        print(format_summary(summarize_runs([result]), set_name))
