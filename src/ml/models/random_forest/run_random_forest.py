# src/ml/models/random_forest/run_random_forest.py

"""Imports"""
from src.ml import config
from src.ml.models.random_forest import config as forest_config
from src.ml.data.file_loader import ShapefileMapLoader
from src.ml.data.road_graph_builder import ROAD_CLASSES, RoadGraphBuilder
from src.ml.data.split import train_val_test_split
from src.ml.evaluation import evaluate, labels_in_data
from src.ml.models.random_forest.random_forest import train_random_forest

"""Train and evaluate Random Forest on the Beijing road-network dataset."""

# Run the training and evaluation pipeline when this file is executed directly.
if __name__ == "__main__":
    # Load the road segments from the Beijing dataset.
    loader = ShapefileMapLoader(config.DATA_DIR)
    segments = loader.get_segments()

    # Build the road-network graph and obtain its features, labels, and connections.
    graph_builder = RoadGraphBuilder(segments)
    graph = graph_builder.get_static_components()

    # Use the shared split so all models train and validate on the same segments.
    # The third returned value is the test mask, which we ignore for now using "_".
    training_mask, validation_mask, _ = train_val_test_split(
        graph["mask"],
        config.SPLIT_SEED,
        config.TRAIN_SHARE,
        config.VAL_SHARE,
    )

    # Standardize the input features using the mean and standard deviation of training segments only.
    # The training mask identifies these segments, preventing validation and test data from influencing the statistics.
    graph_builder.standardize_features(segments, training_mask)

    # The Graph Builder stores features and labels as PyTorch tensors (multi-dimensional array).
    # Convert them to NumPy arrays, which are the standard input format for scikit-learn.
    training_features = graph["x"][training_mask].numpy()
    training_labels = graph["y"][training_mask].numpy()
    validation_features = graph["x"][validation_mask].numpy()
    validation_labels = graph["y"][validation_mask].numpy()

    # Train the classifier using only the labeled training segments and use the shared random seed and the Random Forest-specific parameters.
    classifier = train_random_forest(
        training_features,
        training_labels,
        random_state = config.MODEL_SEED,
        n_estimators = forest_config.N_ESTIMATORS,
        max_depth = forest_config.MAX_DEPTH,
        class_weight = forest_config.CLASS_WEIGHT,
    )

    # Predict the withheld validation labels and calculate the shared metrics.
    predictions = classifier.predict(validation_features)

    # Find all road types with usable labels in the dataset.
    # Use the same classes when evaluating every model, so their macro-F1 scores are comparable.
    class_indices = labels_in_data(graph["y"], graph["mask"]) 

    # Compare the predicted road types with the actual validation labels and calculate the evaluation metrics.
    results = evaluate(validation_labels, predictions, class_indices, class_names = ROAD_CLASSES)

    # Display the overall validation metrics.
    print(f"Validation macro-F1: {results['macro_f1']:.4f}")
    print(f"Validation grouped macro-F1: {results['grouped_macro_f1']:.4f}")
    print(f"Validation accuracy: {results['accuracy']:.4f}")

    # Display precision, recall, and F1-score for each road type.
    for road_type, scores in results["per_class"].items():
        print(
            f"{road_type}: "
            f"precision {scores['precision']:.3f}, "
            f"recall {scores['recall']:.3f}, " 
            f"F1 {scores['f1']:.3f}"
        )

    # Display the confusion matrix and the order of its road-type classes.
    print("Confusion matrix class order:", [ROAD_CLASSES[index] for index in class_indices])
    print("Validation confusion matrix (rows=true, columns=predicted):")
    print(results["confusion_matrix"])
