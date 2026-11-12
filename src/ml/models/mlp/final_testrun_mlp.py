# src/ml/models/mlp/final_testrun_mlp.py

# Final test run: trains the MLP and scores it on the TEST set. Run this only ONCE, at the very end, after all settings
# are chosen on the validation set with dev_run_mlp.py. If the test score is used to change anything, it no longer
# shows how well the model does on unseen data (report: Evaluation Protocol).
#
# The model is trained again instead of loaded from a file. The seeds are fixed, so these are the same models as in
# the development run. Like the development run, it uses every seed and reports the mean and standard deviation.

from src.ml import config
from src.ml.data.road_graph_builder import ROAD_CLASSES
from src.ml.evaluation import evaluate, format_summary, summarize_runs
from src.ml.models.mlp.train_mlp import prepare_data, train_mlp


if __name__ == "__main__":
    # The data is the same for every seed, so it is prepared once
    graph, train_mask, val_mask, test_mask, class_weights, labels = prepare_data()

    # One evaluate() result per seed
    results = []

    for seed in config.MODEL_SEEDS:
        # Training and the choice of epoch only use the training and validation sets, exactly as in the development run
        predictions, best_epoch = train_mlp(graph, train_mask, val_mask, class_weights, labels, seed)

        # Score this seed's predictions on the test segments with the shared evaluation function
        result = evaluate(graph["y"][test_mask], predictions[test_mask], labels, class_names=ROAD_CLASSES)
        results.append(result)

        print(f"Seed {seed}: best epoch {best_epoch}, test macro-F1 {result['macro_f1']:.4f}, grouped macro-F1 {result['grouped_macro_f1']:.4f}, accuracy {result['accuracy']:.4f}")

    # Mean and standard deviation over the seeds, in the same format as the development run, so the two can be compared
    print(format_summary(summarize_runs(results), "Test"))
