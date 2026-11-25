# src/ml/models/graphsage/graphsage_config.py

# GraphSAGE settings. These are tuned on the validation set, so they may differ from the values other models use.
# Settings that must be the same for every model are in src/ml/config.py.

# Hidden size: after the first layer, each road segment is described by this many learned numbers, which the second
# layer uses to score the road types. Bigger can capture more patterns but is slower and overfits more easily.
# 64 is a starting point.
WORKING_SPACE = 64

# Dropout is only active during training. It sets this share of values to zero, so the model learns patterns instead
# of memorising the training roads.
DROPOUT = 0.5

# How big each weight change is: too big and training jumps around, too small and it learns very slowly.
# 0.01 is a common starting point.
LEARNING_RATE = 0.01

# Group hiding (the "label trick", Wang et al. 2021, https://arxiv.org/abs/2103.13355).
# Every training segment is then predicted once per epoch without its own label.
#
# More groups: fewer labels hidden at a time (closer to the real situation), but slower, because each group is one
#              run of the model, so an epoch takes GROUPS runs.
# Fewer groups: faster, but more labels are hidden, so the shares in training look less like when predicting.
# Must be at least 2: with 1 group every training label would be hidden, and the shares would always be 0.
GROUPS = 10
