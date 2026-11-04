# src/ml/config.py

# Settings shared by all models. Every model must use the same values, so their results can be compared fairly.
# Settings that belong to one model (and are tuned on the validation set) are in that model's own config file,
# e.g. src/ml/models/graphsage/graphsage_config.py.

# Folder with the road network shapefile (edges.shp and its companion files)
DATA_DIR = "data/bj_data/map"

# Shares of the labelled segments used for training and validation. The test share is the rest
# (1 - TRAIN_SHARE - VAL_SHARE).
TRAIN_SHARE = 0.6
VAL_SHARE = 0.2

# Graph Splitting Strategy: "random" or "spatial"
STRATEGY = "random"

# The same seed always gives the same split
SPLIT_SEED = 42

# Every model is trained once per seed, and results are printed as mean and standard deviation over these runs.
# The same seed always gives the same starting weights. Kept separate from SPLIT_SEED, so every seed uses the same split.
# The seeds are 0, 1, 2, ... up to AMOUNT_OF_SEEDS - 1. Change AMOUNT_OF_SEEDS to use more or fewer seeds.
# They are fixed numbers, not random ones, so every run gives the same results.
AMOUNT_OF_SEEDS = 5
# range(5) gives 0, 1, 2, 3, 4, and list() turns that into [0, 1, 2, 3, 4]
MODEL_SEEDS = list(range(AMOUNT_OF_SEEDS))

# Number of training epochs. Each epoch runs the model on the whole graph once and changes the weights once.
EPOCHS = 50
