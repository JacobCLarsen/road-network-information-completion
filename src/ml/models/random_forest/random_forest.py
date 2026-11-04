# src/ml/models/random_forest/random_forest.py

"""Imports"""
#Training function for the road-type Random Forest classifier.
from sklearn.ensemble import RandomForestClassifier

"""Random Forest classifier for road-type prediction"""
def train_random_forest(
    training_features,   # Input features of the road segments used for training. (e.g. log_length, curvature, one_way, degree)
    training_labels,     # Corresponding road-type labels for the training segments.
    *,                   # All following parameters must be provided by name.
    random_state,        # Random seed used to make training reproducible.
    n_estimators = 100,  # Number of decision trees in the Random Forest.
    max_depth = None,    # Maximum depth of each tree (None means no predefined limit)
    class_weight = None, # Optional class weights to account for imbalanced road types.
):
    """Train and return a Random Forest classifier using the provided training data."""
    # Create the classifier with the specified model parameters.
    classifier = RandomForestClassifier(
        n_estimators = n_estimators,
        max_depth = max_depth,
        class_weight = class_weight,
        random_state = random_state,
    )

    # Train the classifier using the road-segment features and their labels.
    classifier.fit(training_features, training_labels)

    # Return the trained classifier for later prediction and evaluation.
    return classifier
