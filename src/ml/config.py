# src/ml/config.py

# ---------------------------------------------------------------------------------------------------------------------
# Shared by all models. Every model must use the same values, so their results can be compared fairly.
# ---------------------------------------------------------------------------------------------------------------------

# Folder with the road network shapefile (edges.shp and its companion files)
DATA_DIR = "data/bj_data/map"

# The labelled segments are split into training, validation and test. The test share is the rest (here 20%).
TRAIN_SHARE = 0.6
VAL_SHARE = 0.2
# The same seed always gives the same split
SPLIT_SEED = 42

# Number of training epochs. Each epoch runs the model on the whole graph once and changes the weights once.
EPOCHS = 50

# ---------------------------------------------------------------------------------------------------------------------
# GraphSAGE. These are tuned on the validation set, so they may differ from the values other models use.
# ---------------------------------------------------------------------------------------------------------------------

# Hidden size: after the first layer, each road segment is described by this many learned numbers, which the second
# layer uses to score the road types. Bigger can capture more patterns but is slower and overfits more easily.
# 64 is a starting point.
GRAPHSAGE_WORKING_SPACE = 64

# Dropout is only active during training. It sets this share of values to zero, so the model learns patterns instead
# of memorising the training roads.
GRAPHSAGE_DROPOUT = 0.5

# How big each weight change is: too big and training jumps around, too small and it learns very slowly.
# 0.01 is a common starting point.
GRAPHSAGE_LEARNING_RATE = 0.01
