# tests/ml/models/test_random_forest.py

""" imports """
import pytest
from src.ml.models.random_forest.random_forest import train_random_forest

"""Tests for the Random Forest road-type classifier."""

# Artificial road segments: [log(length), curvature, one_way, degree].
# Class 6 = residential, class 0 = motorway.
TRAINING_FEATURES = [
    [3.0, 0.15, 0, 2],
    [3.2, 0.12, 0, 3],
    [3.5, 0.10, 0, 2],
    [3.8, 0.09, 0, 3],
    [5.8, 0.01, 1, 6],
    [6.0, 0.02, 1, 7],
    [6.2, 0.01, 1, 5],
    [6.5, 0.02, 1, 6],
]

# Corresponding road-type labels (first four residential, last four motorway).
TRAINING_LABELS = [6, 6, 6, 6, 0, 0, 0, 0]

def test_random_forest_trains_with_given_parameters():
    # Verify that training creates the requested forest using four input features.
    classifier = train_random_forest(
        TRAINING_FEATURES,
        TRAINING_LABELS,
        random_state = 42,
        n_estimators = 10,
        max_depth = 2,
        class_weight=None,
    )

    assert len(classifier.estimators_) == 10
    assert classifier.n_features_in_ == 4
    assert classifier.max_depth == 2

def test_random_forest_predicts_road_types():
    # Test predictions on artificial segments not included in the training data.
    classifier = train_random_forest(
        TRAINING_FEATURES,
        TRAINING_LABELS,
        random_state = 42,
        n_estimators = 50,
        max_depth = None,
        class_weight = None,
    )

    validation_features = [
        [3.1, 0.14, 0, 2],  # Similar to residential training segments.
        [6.1, 0.02, 1, 6],  # Similar to motorway training segments.
    ]

    # Predict the road types of the artificial validation segments.
    predictions = classifier.predict(validation_features)

    # Verify that the predictions match the expected road-type labels.
    assert predictions.tolist() == [6, 0]

# Verify that Random Forest rejects training data where the number of features and labels does not match.
def test_random_forest_rejects_mismatched_training_data():
    # pytest.raises() checks that the code inside the block raises the specified exception (ValueError).
    # The test passes if the expected error occurs and fails if no error is raised.
    with pytest.raises(ValueError):
        # Attempt to train Random Forest using eight feature vectors but only seven labels.
        train_random_forest(
            TRAINING_FEATURES,
            TRAINING_LABELS[:-1],  # [:-1] removes the last label, creating a mismatch with the eight feature vectors.
            random_state = 42,
            n_estimators = 10,
            max_depth = None,
            class_weight = None,
        )
