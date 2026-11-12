# src/ml/models/mlp/dev_run_mlp.py

# Development run: trains the MLP and scores it on the VALIDATION set. Use this while developing and tuning.
# The test set is never used here; see final_testrun_mlp.py.

from src.ml import config
from src.ml.data.road_graph_builder import ROAD_CLASSES
from src.ml.evaluation import evaluate, format_summary, summarize_runs
from src.ml.models.mlp.train_mlp import prepare_data, train_mlp


if __name__ == "__main__":
    # The data is the same for every seed, so it is prepared once.
    # The _ is the test mask. It is not given a name, because it must not be used during development.
    graph, train_mask, val_mask, _, class_weights, labels = prepare_data()

    # One evaluate() result per seed. Every seed gives different random starting weights and a different dropout
    # pattern, so the scores differ a little from run to run.
    results = []

    for seed in config.MODEL_SEEDS:
        predictions, best_epoch = train_mlp(graph, train_mask, val_mask, class_weights, labels, seed)

        # Score this seed's predictions on the validation segments with the shared evaluation function
        result = evaluate(graph["y"][val_mask], predictions[val_mask], labels, class_names=ROAD_CLASSES)
        results.append(result)

        print(f"Seed {seed}: best epoch {best_epoch}, validation macro-F1 {result['macro_f1']:.4f}, grouped macro-F1 {result['grouped_macro_f1']:.4f}, accuracy {result['accuracy']:.4f}")

    # Mean and standard deviation over the seeds, in the same format as the other models
    print(format_summary(summarize_runs(results), "Validation"))
