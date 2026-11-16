# src/ml/models/graphsage/graphsage_config.py

# GraphSAGE settings. These are tuned on the validation set, so they may differ from the values other models use.
# Settings that must be the same for every model are in src/ml/config.py.
# The values below come from sweeps with sweep_graphsage.py at 50 epochs and 1 seed. Those sweeps are not considered
# valid any more (too few epochs, no way to tell noise from real differences), so the values are being re-swept at
# config.EPOCHS (500) epochs with 3 seeds. Update the "Sweep:" notes below as the new results come in.
# The settings affect each other: combining the best value of each separate sweep (learning rate 0.01, dropout 0.2,
# hidden size 64) scored worse than the combination below, so always check a new combination with a dev run.

# Hidden size: after the first layer, each road segment is described by this many learned numbers, which the second
# layer uses to score the road types. Bigger can capture more patterns but is slower and overfits more easily.
# Sweep: 32, 64, 128 and 256 all scored within noise of each other (at learning rate 0.05, dropout 0.6).
WORKING_SPACE = 256

# Dropout is only active during training. It sets this share of values to zero, so the model learns patterns instead
# of memorising the training roads.
# Sweep: 0.2 to 0.6 were all within noise of each other (at learning rate 0.05, hidden size 256).
DROPOUT = 0.6

# How big each weight change is: too big and training jumps around, too small and it learns very slowly.
# Sweep: 0.01 scored clearly best of 0.001, 0.005, 0.01 and 0.05 (at dropout 0.6, hidden size 256).
LEARNING_RATE = 0.01

# Weight decay pulls every weight a little towards zero at each step, so the model keeps its weights small and
# overfits less. 0 means no weight decay.
# Sweep: not swept yet.
WEIGHT_DECAY = 0

# How often the validation set is checked during training: every this many epochs (and after the last epoch). The
# best epoch is chosen from the checked epochs only. Checking runs the model on the whole graph a second time, so
# checking less often makes training faster. 1 checks after every epoch.
VALIDATE_EVERY = 5
