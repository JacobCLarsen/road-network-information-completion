# src/ml/config.py

# Settings shared by all models. Every model must use the same values, so their results can be compared fairly.
# Settings that belong to one model (and are tuned on the validation set) are in that model's own config file,
# e.g. src/ml/models/graphsage/graphsage_config.py.

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
AMOUNT_OF_SEEDS = 3
# range(5) gives 0, 1, 2, 3, 4, and list() turns that into [0, 1, 2, 3, 4]
MODEL_SEEDS = list(range(AMOUNT_OF_SEEDS))

# Number of training epochs. Each epoch runs the model on the whole graph once and changes the weights once.
EPOCHS = 750

# How strongly the class weights in the loss favour rare road types (see src/ml/class_weights.py). 1 gives the full
# inverse-frequency weights, 0.5 their square root, 0 no weighting. With 1, GraphSAGE predicted rare types far too
# often (e.g. many residential segments as living_street); 0.5 raised its validation macro-F1 from 0.370 to 0.402.
CLASS_WEIGHT_POWER = 0.5
