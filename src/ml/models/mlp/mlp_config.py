# src/ml/models/mlp/mlp_config.py

# MLP settings. These are tuned on the validation set, so they may differ from the values other models use.
# Settings that must be the same for every model are in src/ml/config.py.

# Hidden size: after the first layer, each road segment is described by this many learned numbers, which the second
# layer uses to score the road types. Bigger can capture more patterns but is slower and overfits more easily.
# Starts at the same value as GraphSAGE, so the only difference between the two is that the MLP ignores the neighbours.
WORKING_SPACE = 64

# Dropout is only active during training. It sets this share of values to zero, so the model learns patterns instead
# of memorising the training roads.
DROPOUT = 0.5

# How big each weight change is: too big and training jumps around, too small and it learns very slowly.
# 0.01 is a common starting point.
LEARNING_RATE = 0.01
