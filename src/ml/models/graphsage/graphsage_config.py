# src/ml/models/graphsage/graphsage_config.py

# GraphSAGE settings. These are tuned on the validation set, so they may differ from the values other models use.
# Settings that must be the same for every model are in src/ml/config.py.
# The values below come from one round of sweep_graphsage.py at config.EPOCHS (500) epochs with seeds 0, 1 and 2,
# tuning one setting at a time in the order learning rate, weight decay, dropout, hidden size. The scores in the
# "Sweep:" notes are validation macro-F1 (mean ± standard deviation over the seeds).
# Best combination: validation macro-F1 0.4425 ± 0.0003, accuracy 0.6023 (the previous values, with dropout 0.6,
# scored 0.4347 ± 0.0006). Regularising more (weight decay, more dropout, a smaller hidden size) always scored worse,
# and the best epochs were mostly between 430 and 500, so the model underfits and was still improving at 500 epochs.
# Earlier sweeps at 50 epochs and 1 seed are no longer used: they were too short and could not tell noise from real
# differences.

# Hidden size: after the first layer, each road segment is described by this many learned numbers, which the second
# layer uses to score the road types. Bigger can capture more patterns but is slower and overfits more easily.
# Sweep: 256 (0.4425) clearly beat 128 (0.4364), at learning rate 0.01, weight decay 0, dropout 0.2. 256 was the largest
# value tried, so a bigger hidden size might score even higher.
WORKING_SPACE = 256

# Dropout is only active during training. It sets this share of values to zero, so the model learns patterns instead
# of memorising the training roads.
# Sweep: 0.2 (0.4425) beat 0.4 (0.4371) and 0.6 (0.4347), at learning rate 0.01, weight decay 0, hidden size 256.
# 0.2 was the smallest value tried, so less dropout might score even higher.
DROPOUT = 0.2

# How big each weight change is: too big and training jumps around, too small and it learns very slowly.
# Sweep: 0.01 (0.4347) beat 0.005 (0.4302), at weight decay 0, dropout 0.6, hidden size 256. 0.001 and 0.05 were not
# tried at 500 epochs.
LEARNING_RATE = 0.01

# Weight decay pulls every weight a little towards zero at each step, so the model keeps its weights small and
# overfits less. 0 means no weight decay.
# Sweep: 0 (0.4347) beat 5e-4 (0.4225) and 5e-3 (0.3691), at learning rate 0.01, dropout 0.6, hidden size 256.
WEIGHT_DECAY = 0

# How often the validation set is checked during training: every this many epochs (and after the last epoch). The
# best epoch is chosen from the checked epochs only. Checking runs the model on the whole graph a second time, so
# checking less often makes training faster. 1 checks after every epoch.
VALIDATE_EVERY = 1

# Group hiding (the "label trick", Wang et al. 2021, https://arxiv.org/abs/2103.13355).
# Every training segment is then predicted once per epoch without its own label.
#
# More groups: fewer labels hidden at a time (closer to the real situation), but slower, because each group is one
#              run of the model, so an epoch takes GROUPS runs.
# Fewer groups: faster, but more labels are hidden, so the shares in training look less like when predicting.
# Must be at least 2: with 1 group every training label would be hidden, and the shares would always be 0.
GROUPS = 10
