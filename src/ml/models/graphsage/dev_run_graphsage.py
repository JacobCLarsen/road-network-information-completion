# src/ml/models/graphsage/dev_run_graphsage.py

# Development run: trains GraphSAGE and scores it on the VALIDATION set. Use this while developing and tuning.
# The test set is never used here; see final_testrun_graphsage.py.

from src.ml import config
from src.ml.data.road_graph_builder import ROAD_CLASSES
from src.ml.evaluation import evaluate
from src.ml.models.graphsage.train_graphsage import prepare_data, train_graphsage


if __name__ == "__main__":
    # The _ is the test mask. It is not given a name, because it must not be used during development.
    graph, train_mask, val_mask, _, class_weights, labels = prepare_data()

    predictions, best_epoch, best_f1 = train_graphsage(graph, train_mask, val_mask, class_weights, labels, config.MODEL_SEED)

    # Score the predictions on the validation segments with the shared evaluation function
    result = evaluate(graph["y"][val_mask], predictions[val_mask], labels, class_names=ROAD_CLASSES)

    # Macro-F1 is the main metric. Grouped macro-F1 scores only the coarse categories (highways, main roads, minor roads),
    # and accuracy is shown for comparison with the majority-class baseline.
    print(f"Validation macro-F1: {result['macro_f1']:.4f}")
    print(f"Validation grouped macro-F1: {result['grouped_macro_f1']:.4f}")
    print(f"Validation accuracy: {result['accuracy']:.4f}")
    for name, scores in result["per_class"].items():
        print(f"{name}: precision {scores['precision']:.3f}, recall {scores['recall']:.3f}, F1 {scores['f1']:.3f}")
