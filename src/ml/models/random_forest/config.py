# src/ml/models/random_forest/config.py

"""Random Forest model configuration."""

# Number of decision trees used by the Random Forest.
N_ESTIMATORS = 100

# Maximum depth of each decision tree (None means no predefined limit).
MAX_DEPTH = None

# Class weighting used during training (None means no additional class weighting).
CLASS_WEIGHT = None
