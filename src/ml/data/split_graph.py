# src/ml/data/split_graph.py
"""
Train/validation/test split of the road line graph, done with masks over the complete graph.

The model always sees the whole graph (all segments, connections and features). The masks only decide whose
label is used for training, for model selection (validation) and for the final evaluation (test).

By default, everything comes from the config, so every model uses the same split:
    TRAIN_SHARE, VAL_SHARE:  the sizes of the training and validation sets (the test set gets the rest)
    SPLIT_STRATEGY:          "random" or "spatial"
    SPLIT_SEED:              the seed, so the same split comes out every time

Two strategies:
    "random":  Random split of the labeled segments, stratified by road type (every type is split on its own).
    "spatial": The map is divided into a grid and whole cells go to one split, so test segments lie in areas
               the model has not seen. The shares are then approximate, because a whole cell moves at once.

Segments without a label (graph["mask"] is False) are in none of the masks.

Usage:
    from src.ml.data.split_graph import make_split_masks

    graph = RoadGraphBuilder(segments).get_static_components()
    masks = make_split_masks(graph)    # uses SPLIT_STRATEGY, SPLIT_SEED, TRAIN_SHARE and VAL_SHARE from the config

    # To use other values than the config, e.g. in tests, pass them in:
    masks = make_split_masks(graph, seed=1, strategy="spatial", train_share=0.6, val_share=0.2)
"""

import torch

from src.ml.config import SPLIT_SEED, SPLIT_STRATEGY, TRAIN_SHARE, VAL_SHARE

SPLIT_NAMES = ("train", "validation", "test")


def _new_masks(num_nodes):
    return {name: torch.zeros(num_nodes, dtype=torch.bool) for name in SPLIT_NAMES}


def _split_counts(count, train_share, val_share):
    """Number of items for train, validation and test out of `count`. Train gets at least one item, test the rest."""
    if count == 0:
        return 0, 0, 0
    n_train = min(max(1, int(round(count * train_share))), count)
    n_val = min(int(round(count * val_share)), count - n_train)
    return n_train, n_val, count - n_train - n_val


def _random_split(y, known, train_share, val_share, seed, stratify):
    generator = torch.Generator().manual_seed(seed)
    masks = _new_masks(y.shape[0])

    if stratify:
        groups = [(known & (y == c)).nonzero().flatten() for c in y[known].unique(sorted=True)]
    else:
        groups = [known.nonzero().flatten()]

    for indices in groups:
        indices = indices[torch.randperm(len(indices), generator=generator)]
        n_train, n_val, _ = _split_counts(len(indices), train_share, val_share)
        masks["train"][indices[:n_train]] = True
        masks["validation"][indices[n_train:n_train + n_val]] = True
        masks["test"][indices[n_train + n_val:]] = True

    return masks


def _spatial_split(geometry, known, train_share, val_share, test_share, seed, cells_per_side):
    """
    Lays a cells_per_side x cells_per_side grid over the map and gives each whole cell to one split.
    Cells are handed out in random order to the split that is furthest below its target size.
    """
    masks = _new_masks(known.shape[0])
    if not known.any():
        return masks

    # Position of each segment: the mean (longitude, latitude) of its points
    centroids = torch.tensor(
        [(sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)) for pts in geometry],
        dtype=torch.float64,
    )
    low = centroids.min(dim=0).values
    span = (centroids.max(dim=0).values - low).clamp(min=1e-12)
    cell_xy = ((centroids - low) / span * cells_per_side).floor().long().clamp(0, cells_per_side - 1)
    cell_id = cell_xy[:, 0] * cells_per_side + cell_xy[:, 1]

    # The labeled segments in each cell
    cells = [((cell_id == cid) & known).nonzero().flatten() for cid in cell_id[known].unique()]

    total = int(known.sum())
    targets = {"train": train_share * total, "validation": val_share * total, "test": test_share * total}
    assigned = {name: 0 for name in SPLIT_NAMES}

    generator = torch.Generator().manual_seed(seed)
    for pos in torch.randperm(len(cells), generator=generator).tolist():
        members = cells[pos]
        best = max(
            (name for name in SPLIT_NAMES if targets[name] > 0),
            key=lambda name: (targets[name] - assigned[name]) / targets[name],
        )
        masks[best][members] = True
        assigned[best] += len(members)

    return masks


def make_split_masks(graph, seed=SPLIT_SEED, strategy=SPLIT_STRATEGY, stratify=True, cells_per_side=20,
                     train_share=TRAIN_SHARE, val_share=VAL_SHARE):
    """
    Returns boolean masks {"train", "validation", "test"}, one value per segment of the full graph.

    graph:           RoadGraphBuilder, or the dictionary from get_static_components().
    seed:            Defaults to SPLIT_SEED from the config. The same seed always gives the same split.
    strategy:        "random" or "spatial". Defaults to SPLIT_STRATEGY from the config.
    stratify:        Random strategy only: split every road type separately.
    cells_per_side:  Spatial strategy only: size of the grid. More cells gives a split closer to random.
    train_share,
    val_share:       Default to TRAIN_SHARE and VAL_SHARE from the config. The test share is the remainder.
    """
    if train_share < 0 or val_share < 0 or train_share + val_share > 1 + 1e-9:
        raise ValueError(f"Invalid shares: train={train_share}, val={val_share}. Both must be >= 0 and sum to at most 1.")
    test_share = max(0.0, 1.0 - train_share - val_share)

    static = graph.get_static_components() if hasattr(graph, "get_static_components") else graph
    y = static["y"]
    known = static["mask"].bool()

    if strategy == "random":
        masks = _random_split(y, known, train_share, val_share, seed, stratify)
    elif strategy == "spatial":
        masks = _spatial_split(static["geometry"], known, train_share, val_share, test_share, seed, cells_per_side)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Use 'random' or 'spatial'.")

    _check_masks(masks, known)
    return masks


def _check_masks(masks, known):
    """The splits must not overlap, and together they must cover exactly the labeled segments."""
    overlap = (masks["train"].int() + masks["validation"].int() + masks["test"].int()) > 1
    if overlap.any():
        raise AssertionError("A segment is in more than one split.")

    union = masks["train"] | masks["validation"] | masks["test"]
    if not torch.equal(union, known):
        raise AssertionError("The splits do not cover exactly the labeled segments.")
