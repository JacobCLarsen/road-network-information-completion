# src/ml/models/graphsage/final_testrun_graphsage.py

# Final test run: trains GraphSAGE and scores it on the TEST set. Run this only ONCE, at the very end, after all
# settings are chosen on the validation set with dev_run_graphsage.py. If the test score is used to change anything,
# it no longer shows how well the model does on unseen data (report: Evaluation Protocol).
#
# The model is trained again instead of loaded from a file. The seeds are fixed, so it is the same model as in the
# development run.

from src.ml import config
from src.ml.data.road_graph_builder import ROAD_CLASSES
from src.ml.evaluation import evaluate
from src.ml.models.graphsage.train_graphsage import prepare_data, train_graphsage


if __name__ == "__main__":
    graph, train_mask, val_mask, test_mask, class_weights, labels = prepare_data()

    # Training and the choice of epoch only use the training and validation sets, exactly as in the development run
    predictions, best_epoch = train_graphsage(graph, train_mask, val_mask, class_weights, labels, config.MODEL_SEEDS[0])

    # Score the predictions on the test segments with the shared evaluation function
    result = evaluate(graph["y"][test_mask], predictions[test_mask], labels, class_names=ROAD_CLASSES)

    print(f"Test macro-F1: {result['macro_f1']:.4f}")
    print(f"Test grouped macro-F1: {result['grouped_macro_f1']:.4f}")
    print(f"Test accuracy: {result['accuracy']:.4f}")
    for name, scores in result["per_class"].items():
        print(f"{name}: precision {scores['precision']:.3f}, recall {scores['recall']:.3f}, F1 {scores['f1']:.3f}")
