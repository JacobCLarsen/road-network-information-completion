# src/ml/models/graphsage/final_testrun_graphsage.py

# Final test run: trains GraphSAGE and scores it on the TEST set. Run this only ONCE, at the very end, after all
# settings are chosen on the validation set with dev_run_graphsage.py. If the test score is used to change anything,
# it no longer shows how well the model does on unseen data (report: Evaluation Protocol).
#
# The model is trained again instead of loaded from a file. The seeds are fixed, so these are the same models as in
# the development run. Like the development run, it uses every seed and reports the mean and standard deviation.

from src.ml import config
from src.ml.data.road_graph_builder import ROAD_CLASSES
from src.ml.evaluation import evaluate, format_summary, summarize_runs
from src.ml import save_run
from src.ml.models.graphsage.train_graphsage import prepare_data, saved_settings, train_graphsage


if __name__ == "__main__":
    # The data is the same for every seed, so it is prepared once
    graph, train_mask, val_mask, test_mask, class_weights, labels = prepare_data()

    # Save the split in the database (or find the same one saved before), so every run points to the segments it was
    # trained and scored on. The test mask is only used for this, never for training or scoring.
    split_id = save_run.save_split(graph, {"train": train_mask, "validation": val_mask, "test": test_mask})

    # One evaluate() result per seed
    results = []

    # Training and the choice of epoch only use the training and validation sets, exactly as in the development run
    for seed in config.MODEL_SEEDS:
        # Saved with status "running" first, so a run that crashes or is stopped can be seen in the database
        run_id = save_run.start_run("GraphSAGE", seed, split_id, saved_settings())
        try:
            # The training curve, filled in by train_graphsage: one row per epoch
            history = []
            predictions, best_epoch = train_graphsage(graph, train_mask, val_mask, class_weights, labels, seed, history=history)

            # Score this seed's predictions on the test segments with the shared evaluation function
            result = evaluate(graph["y"][test_mask], predictions[test_mask], labels, class_names=ROAD_CLASSES)
            results.append(result)

            # Save the settings, the training curve, the scores and the predictions of every segment
            save_run.finish_run(run_id, graph, predictions, {"test": result}, history, best_epoch)
        except (Exception, KeyboardInterrupt):
            # KeyboardInterrupt is Ctrl+C, which is not an Exception
            save_run.fail_run(run_id)
            raise

        print(f"Seed {seed}: best epoch {best_epoch}, test macro-F1 {result['macro_f1']:.4f}, grouped macro-F1 {result['grouped_macro_f1']:.4f}, accuracy {result['accuracy']:.4f}")

    # Mean and standard deviation over the seeds, in the same format as the development run, so the two can be compared
    print(format_summary(summarize_runs(results), "Test"))
