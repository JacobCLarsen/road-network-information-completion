# src/ml/models/graphsage/dev_run_graphsage.py

# Development run: trains GraphSAGE and scores it on the VALIDATION set. Use this while developing and tuning.
# The test set is never used here; see final_testrun_graphsage.py.

import statistics

from src.ml import config
from src.ml.data.road_graph_builder import ROAD_CLASSES
from src.ml.evaluation import evaluate
from src.ml.models.graphsage.train_graphsage import prepare_data, train_graphsage


if __name__ == "__main__":
    # The data is the same for every seed, so it is prepared once.
    # The _ is the test mask. It is not given a name, because it must not be used during development.
    graph, train_mask, val_mask, _, class_weights, labels = prepare_data()

    # One list per score, with one value per seed. Every seed gives different random starting weights and a different
    # dropout pattern (which values are dropped; how many is set by DROPOUT in graphsage_config.py), so the scores
    # differ a little from run to run.
    macro_f1s = []
    grouped_f1s = []
    accuracies = []
    # The F1 of each road type, one value per seed, e.g. {"motorway": [0.72, 0.70, ...], ...}
    f1_per_type = {}

    for seed in config.MODEL_SEEDS:
        predictions, best_epoch = train_graphsage(graph, train_mask, val_mask, class_weights, labels, seed)

        # Score this seed's predictions on the validation segments with the shared evaluation function
        result = evaluate(graph["y"][val_mask], predictions[val_mask], labels, class_names=ROAD_CLASSES)

        # Keep only the numbers, so the mean and standard deviation can be computed after the loop
        macro_f1s.append(result["macro_f1"])
        grouped_f1s.append(result["grouped_macro_f1"])
        accuracies.append(result["accuracy"])
        for name, type_scores in result["per_class"].items():
            # setdefault creates an empty list the first time a road type is seen
            f1_per_type.setdefault(name, []).append(type_scores["f1"])

        print(f"Seed {seed}: best epoch {best_epoch}, validation macro-F1 {result['macro_f1']:.4f}, grouped macro-F1 {result['grouped_macro_f1']:.4f}, accuracy {result['accuracy']:.4f}")

    # Mean and standard deviation over the seeds. The standard deviation shows how much the result depends on the seed:
    # a difference between two settings only counts if it is clearly bigger than this.
    # Macro-F1 is the main metric. Grouped macro-F1 scores only the coarse categories (highways, main roads, minor roads),
    # and accuracy is shown for comparison with the majority-class baseline.
    print(f"Results over {len(config.MODEL_SEEDS)} seeds (mean ± standard deviation):")
    print(f"Validation macro-F1: {statistics.mean(macro_f1s):.4f} ± {statistics.stdev(macro_f1s):.4f}")
    print(f"Validation grouped macro-F1: {statistics.mean(grouped_f1s):.4f} ± {statistics.stdev(grouped_f1s):.4f}")
    print(f"Validation accuracy: {statistics.mean(accuracies):.4f} ± {statistics.stdev(accuracies):.4f}")
    for name, f1s in f1_per_type.items():
        print(f"{name}: F1 {statistics.mean(f1s):.3f} ± {statistics.stdev(f1s):.3f}")
