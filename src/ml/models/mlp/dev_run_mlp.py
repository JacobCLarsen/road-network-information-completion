# src/ml/models/mlp/dev_run_mlp.py

# Development run: trains the MLP and scores it on the VALIDATION set. Use this while developing and tuning.
# The test set is never used here; see final_testrun_mlp.py.

from src.ml import config
from src.ml.data.road_graph_builder import ROAD_CLASSES
from src.ml.evaluation import evaluate, format_summary, summarize_runs
from src.ml import save_run
from src.ml.models.mlp.train_mlp import prepare_data, saved_settings, train_mlp


if __name__ == "__main__":
    # The data is the same for every seed, so it is prepared once.
    # The test mask must not be used for training or scoring during development. It is only saved with the split.
    graph, train_mask, val_mask, test_mask, class_weights, labels = prepare_data()

    # Save the split in the database (or find the same one saved before), so every run points to the segments it was
    # trained and scored on. The test mask is only used for this, never for training or scoring.
    split_id = save_run.save_split(graph, {"train": train_mask, "validation": val_mask, "test": test_mask})

    # One evaluate() result per seed. Every seed gives different random starting weights and a different dropout
    # pattern, so the scores differ a little from run to run.
    results = []

    for seed in config.MODEL_SEEDS:
        # Saved with status "running" first, so a run that crashes or is stopped can be seen in the database
        run_id = save_run.start_run("MLP", seed, split_id, saved_settings())
        try:
            # The training curve, filled in by train_mlp: one row per epoch
            history = []
            predictions, best_epoch = train_mlp(graph, train_mask, val_mask, class_weights, labels, seed, history=history)

            # Score this seed's predictions on the validation segments with the shared evaluation function
            result = evaluate(graph["y"][val_mask], predictions[val_mask], labels, class_names=ROAD_CLASSES)
            results.append(result)

            # Save the settings, the training curve, the scores and the predictions of every segment
            save_run.finish_run(run_id, graph, predictions, {"val": result}, history, best_epoch)
        except (Exception, KeyboardInterrupt):
            # KeyboardInterrupt is Ctrl+C, which is not an Exception
            save_run.fail_run(run_id)
            raise

        print(f"Seed {seed}: best epoch {best_epoch}, validation macro-F1 {result['macro_f1']:.4f}, grouped macro-F1 {result['grouped_macro_f1']:.4f}, accuracy {result['accuracy']:.4f}")

    # Mean and standard deviation over the seeds, in the same format as the other models
    print(format_summary(summarize_runs(results), "Validation"))
