# src/ml/class_weights.py

import torch


def compute_class_weights(train_labels, num_classes):
    # Returns one weight per road type for the loss, so mistakes on rare road types cost more than mistakes on common
    # ones. Shared by all models, so they are trained with the same weights and their results can be compared fairly.

    # The weight of a road type is inversely proportional to how often it occurs:
    #     number of training segments / (number of road types that occur * segments of this road type)
    # A road type with an average number of segments gets weight 1. Road types with no training segments get weight 0,
    # since there is nothing to learn from them (and dividing by 0 would give infinity).

    # train_labels: The labels of the TRAINING segments only. Validation and test labels must never be passed in, so
    #               nothing is learned from them.
    # num_classes:  The number of road types, so every road type gets a weight, also those with no training segments.

    # How many training segments each road type has. bincount counts how often each label occurs, e.g. the labels
    # [0, 0, 2] give [2, 0, 1]. minlength makes the list one entry per road type, also when the last types never occur.
    # .float() turns the counts into decimal numbers, so the division below gives decimals instead of whole numbers.
    counts = torch.bincount(train_labels, minlength=num_classes).float()

    # Start with weight 0 for every road type
    class_weights = torch.zeros(num_classes)
    # True for the road types with at least one training segment, e.g. [2, 0, 1] gives [True, False, True]
    present = counts > 0
    # Fill in the weights for those road types only: number of training segments / (number of road types that occur
    # * segments of this road type). present.sum() counts the True values. The other road types keep weight 0.
    class_weights[present] = len(train_labels) / (present.sum() * counts[present])
    return class_weights
