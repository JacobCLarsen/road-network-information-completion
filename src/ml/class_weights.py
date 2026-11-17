# src/ml/class_weights.py

"""Imports"""
import torch

"""Calculate loss weights from training labels, with adjustable weighting strength."""
def compute_class_weights(train_labels, num_classes, power=1.0):
    # Returns one weight per road type for the loss, so mistakes on rare road types cost more than mistakes on common
    # ones. Shared by all models, so they are trained with the same weights and their results can be compared fairly.
    
    # The weight of a road type is proportional to how often it occurs:
    #     number of training segments / (number of road types that occur * segments of this road type)
    # A road type with an average number of segments gets weight 1. Road types with no training segments get weight 0,
    # since there is nothing to learn from them (and dividing by 0 would give infinity).
    
    # train_labels: The labels of the TRAINING segments only. Validation and test labels must never be passed in, so
    #               nothing is learned from them.
    # num_classes:  The number of road types, so every road type gets a weight, also those with no training segments.

    # bincount() counts how many training segments have each label.
    # For example, [0, 0, 2] becomes [2, 0, 1]. minlength ensures
    # that even road types with no training segments are included.
    counts = torch.bincount(train_labels, minlength=num_classes).float()

    # Road types absent from training keep a weight of zero. The mask
    # 'present' prevents dividing by zero for those missing road types.
    class_weights = torch.zeros(num_classes)
    present = counts > 0

    # The original inverse-frequency formula gives larger weights to
    # rare road types: N / (K * count), where N is the number of training
    # segments and K is the number of road types present in training.
    class_weights[present] = len(train_labels) / (present.sum() * counts[present])

    # Power 1.0 keeps the original weights; 0.5 takes their square roots;
    # 0.0 gives every present road type weight 1 (no class weighting).
    # Only change present classes, so missing classes stay at weight zero.
    class_weights[present] = class_weights[present] ** power

    return class_weights
