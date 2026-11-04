# src/ml/models/graphsage/dev_run_graphsage.py

# Development run: trains GraphSAGE and scores it on the VALIDATION set. Use this while developing and tuning.
# The test set is never used here; see final_testrun_graphsage.py.

from src.ml import config
from src.ml.data.road_graph_builder import ROAD_CLASSES
from src.ml.evaluation import evaluate, format_summary, summarize_runs
from src.ml.models.graphsage.train_graphsage import prepare_data, train_graphsage


if __name__ == "__main__":
    # The data is the same for every seed, so it is prepared once.
    # The _ is the test mask. It is not given a name, because it must not be used during development.
    graph, train_mask, val_mask, _, class_weights, labels = prepare_data()

    # One evaluate() result per seed. Every seed gives different random starting weights and a different dropout
    # pattern (which values are dropped; how many is set by DROPOUT in graphsage_config.py), so the scores differ a
    # little from run to run.
    results = []

    for seed in config.MODEL_SEEDS:
        predictions, best_epoch = train_graphsage(graph, train_mask, val_mask, class_weights, labels, seed)

        # Score this seed's predictions on the validation segments with the shared evaluation function
        result = evaluate(graph["y"][val_mask], predictions[val_mask], labels, class_names=ROAD_CLASSES)
        results.append(result)

        print(f"Seed {seed}: best epoch {best_epoch}, validation macro-F1 {result['macro_f1']:.4f}, grouped macro-F1 {result['grouped_macro_f1']:.4f}, accuracy {result['accuracy']:.4f}")

    # Mean and standard deviation over the seeds. The standard deviation shows how much the result depends on the seed:
    # a difference between two settings only counts if it is clearly bigger than this.
    # Macro-F1 is the main metric. Grouped macro-F1 scores only the coarse categories (highways, main roads, minor roads),
    # and accuracy is shown for comparison with the majority-class baseline. Per-class scores and the confusion matrices
    # show which road types the model mixes up.
    print(format_summary(summarize_runs(results), "Validation"))
