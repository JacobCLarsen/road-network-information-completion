# src/ml/models/graphsage/final_testrun_graphsage.py

# Final test run: trains GraphSAGE and scores it on the TEST set. Run this only ONCE, at the very end, after all
# settings are chosen on the validation set with dev_run_graphsage.py. If the test score is used to change anything,
# it no longer shows how well the model does on unseen data (report: Evaluation Protocol).
#
# The model is trained again instead of loaded from a file. The seeds are fixed, so these are the same models as in
# the development run. Like the development run, it uses every seed and reports the mean and standard deviation.

import statistics

from src.ml import config
from src.ml.data.road_graph_builder import ROAD_CLASSES
from src.ml.evaluation import evaluate
from src.ml.models.graphsage.train_graphsage import prepare_data, train_graphsage


if __name__ == "__main__":
    # The data is the same for every seed, so it is prepared once
    graph, train_mask, val_mask, test_mask, class_weights, labels = prepare_data()

    # One list per score, with one value per seed
    macro_f1s = []
    grouped_f1s = []
    accuracies = []
    # The F1 of each road type, one value per seed, e.g. {"motorway": [0.72, 0.70, ...], ...}
    f1_per_type = {}

    for seed in config.MODEL_SEEDS:
        # Training and the choice of epoch only use the training and validation sets, exactly as in the development run
        predictions, best_epoch = train_graphsage(graph, train_mask, val_mask, class_weights, labels, seed)

        # Score this seed's predictions on the test segments with the shared evaluation function
        result = evaluate(graph["y"][test_mask], predictions[test_mask], labels, class_names=ROAD_CLASSES)

        # Keep only the numbers, so the mean and standard deviation can be computed after the loop
        macro_f1s.append(result["macro_f1"])
        grouped_f1s.append(result["grouped_macro_f1"])
        accuracies.append(result["accuracy"])
        for name, type_scores in result["per_class"].items():
            # setdefault creates an empty list the first time a road type is seen
            f1_per_type.setdefault(name, []).append(type_scores["f1"])

        print(f"Seed {seed}: best epoch {best_epoch}, test macro-F1 {result['macro_f1']:.4f}, grouped macro-F1 {result['grouped_macro_f1']:.4f}, accuracy {result['accuracy']:.4f}")

    # Mean and standard deviation over the seeds, as in the development run, so the two can be compared
    print(f"Results over {len(config.MODEL_SEEDS)} seeds (mean ± standard deviation):")
    print(f"Test macro-F1: {statistics.mean(macro_f1s):.4f} ± {statistics.stdev(macro_f1s):.4f}")
    print(f"Test grouped macro-F1: {statistics.mean(grouped_f1s):.4f} ± {statistics.stdev(grouped_f1s):.4f}")
    print(f"Test accuracy: {statistics.mean(accuracies):.4f} ± {statistics.stdev(accuracies):.4f}")
    for name, f1s in f1_per_type.items():
        print(f"{name}: F1 {statistics.mean(f1s):.3f} ± {statistics.stdev(f1s):.3f}")
