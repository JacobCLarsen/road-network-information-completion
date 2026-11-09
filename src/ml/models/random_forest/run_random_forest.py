# src/ml/models/random_forest/run_random_forest.py

print("Starting imports...", flush = True)

"""Imports"""
from src.ml import config
from src.ml.models.random_forest import config as forest_config
from src.ml.data.file_loader import ShapefileMapLoader
from src.ml.data.road_graph_builder import ROAD_CLASSES, RoadGraphBuilder
from src.ml.data.split_graph import make_split_masks
from src.ml.evaluation import evaluate, labels_in_data, summarize_runs, format_summary
from src.ml.models.random_forest.random_forest import train_random_forest

"""Train and evaluate Random Forest on the Beijing road-network dataset."""

print("Imports completed.", flush = True)

# Run the training and evaluation pipeline when this file is executed directly.
if __name__ == "__main__":
    print("Loading road-network dataset...") # DEBUG TRACKING

    # Load the road segments from the Beijing dataset.
    loader = ShapefileMapLoader(config.DATA_DIR)
    segments = loader.get_segments()

    print("Building road-network graph...") # DEBUG TRACKING

    # Build the road-network graph and obtain its features, labels, and connections.
    graph_builder = RoadGraphBuilder(segments)
    graph = graph_builder.get_static_components()

    # Use the shared split so all models train and validate on the same segments.
    # The test mask is not used during development.
    masks = make_split_masks(graph)

    training_mask = masks["train"]
    validation_mask = masks["validation"]

    # Standardize the input features using the mean and standard deviation of training segments only.
    # The training mask identifies these segments, preventing validation and test data from influencing the statistics.
    graph_builder.standardize_features(segments, training_mask)

    # The Graph Builder stores features and labels as PyTorch tensors (multi-dimensional array).
    # Convert them to NumPy arrays, which are the standard input format for scikit-learn.
    training_features = graph["x"][training_mask].numpy()
    training_labels = graph["y"][training_mask].numpy()
    validation_features = graph["x"][validation_mask].numpy()
    validation_labels = graph["y"][validation_mask].numpy()

    print("Training Random Forest...") # DEBUG TRACKING

    # Train the classifier using only the labeled training segments and use the shared random seed and the Random Forest-specific parameters.
    classifier = train_random_forest(
        training_features,
        training_labels,
        random_state = config.MODEL_SEEDS[0],
        n_estimators = forest_config.N_ESTIMATORS,
        max_depth = forest_config.MAX_DEPTH,
        class_weight = forest_config.CLASS_WEIGHT,
    )

    print("Evaluating Random Forest...\n") # DEBUG TRACKING

    # Predict the withheld validation labels and calculate the shared metrics.
    predictions = classifier.predict(validation_features)

    # Find all road types with usable labels in the dataset.
    # Use the same classes when evaluating every model, so their macro-F1 scores are comparable.
    class_indices = labels_in_data(graph["y"], graph["mask"]) 

    # Compare the predicted road types with the actual validation labels and calculate the evaluation metrics.
    results = evaluate(validation_labels, predictions, class_indices, class_names = ROAD_CLASSES)

    # Summarize the evaluation results. Currently only one training run is performed.
    summary = summarize_runs([results])

    # Format and display the validation metrics using the shared evaluation functionality.
    print(format_summary(summary, set_name = "Validation"))
