# src/ml/config.py

# Settings shared by all models. Every model must use the same values, so their results can be compared fairly.
# Settings that belong to one model (and are tuned on the validation set) are in that model's own config.py,
# e.g. src/ml/models/graphsage/config.py.

# Folder with the road network shapefile (edges.shp and its companion files)
DATA_DIR = "data/bj_data/map"

# Shares of the labelled segments used for training and validation. The test share is the rest
# (1 - TRAIN_SHARE - VAL_SHARE).
TRAIN_SHARE = 0.6
VAL_SHARE = 0.2

# How the labelled segments are split: "random" or "spatial".
SPLIT_STRATEGY = "random"

# The same seed always gives the same split
SPLIT_SEED = 42
# The same seed always gives the same starting weights for the model. Kept separate from SPLIT_SEED, so the model can
# be run with several seeds on the same split.
MODEL_SEED = 42

# Number of training epochs. Each epoch runs the model on the whole graph once and changes the weights once.
EPOCHS = 50
