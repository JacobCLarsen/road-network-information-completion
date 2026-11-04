# src/ml/data/split.py

import torch


def train_val_test_split(mask, seed, train_share, val_share):
    """
    TEMPORARY random split, to be replaced by the group's split later. Anything that returns the same three masks
    can take its place.

    Only labelled segments (mask = True) are split. Returns train_mask, val_mask and test_mask: one True/False value
    per segment, with no segment in more than one mask. The same seed always gives the same split.
    """
    # Positions of the labelled segments, in random order
    labelled = mask.nonzero().view(-1)
    generator = torch.Generator().manual_seed(seed)
    shuffled = labelled[torch.randperm(len(labelled), generator=generator)]

    # Cut the shuffled list into train, validation and test (the rest)
    n_train = int(train_share * len(shuffled))
    n_val = int(val_share * len(shuffled))
    parts = [shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]]

    # Turn each part into a True/False mask over all segments
    masks = []
    for part in parts:
        part_mask = torch.zeros(len(mask), dtype=torch.bool)
        part_mask[part] = True
        masks.append(part_mask)

    train_mask, val_mask, test_mask = masks
    return train_mask, val_mask, test_mask
