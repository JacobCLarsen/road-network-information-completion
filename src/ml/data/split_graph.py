# src/ml/data/split_graph.py

# Splits the labeled road segments into a training, a validation and a test set. Shared by all models, so they are
# trained and scored on exactly the same segments and their results can be compared fairly.
#
# The split is done with masks over the complete graph: the model always sees every segment, connection and feature.
# The masks only decide whose label is used for training, for model selection (validation) and for the final
# evaluation (test). Segments without a label (graph["mask"] is False) are in none of the masks.
#
# Two strategies (STRATEGY in the config):
# "random":  Random split of the labeled segments, stratified by road type (every road type is split on its own).
# "spatial": The map is divided into a grid and whole cells go to one split, so test segments lie in areas the
#            model has not seen. The shares are then approximate, because a whole cell moves at once.

import torch

from src.ml.config import TRAIN_SHARE, VAL_SHARE, STRATEGY, SPLIT_SEED

# The names of the splits, used as keys of the masks dictionary
SPLIT_NAMES = ("train", "validation", "test")

# Spatial strategy: the map is divided into CELLS_PER_SIDE x CELLS_PER_SIDE grid cells.
# More cells gives a split that is closer to a random split.
CELLS_PER_SIDE = 20

# Allowed rounding error when checking that train_share + val_share is at most 1 (e.g. 0.7 + 0.3 = 1.0000000000000002)
SHARE_TOLERANCE = 1e-9

# Smallest width or height of the map used when placing segments in the grid. Without it, a map where every segment
# has the same longitude (or latitude) would give a division by zero.
MIN_SPAN = 1e-12


def _new_masks(num_segments):
    """
    Returns one mask per split, all False. Each mask has one value per segment of the graph.
    """
    return {name: torch.zeros(num_segments, dtype=torch.bool) for name in SPLIT_NAMES}


def _split_counts(count, train_share, val_share):
    """
    Returns how many of `count` items go to train, validation and test.
    Train gets at least one item (if there are any), and test gets whatever is left.
    """
    if count == 0:
        return 0, 0, 0

    n_train = min(max(1, int(round(count * train_share))), count)
    n_val = min(int(round(count * val_share)), count - n_train)
    n_test = count - n_train - n_val
    return n_train, n_val, n_test


def _random_split(y, has_label, train_share, val_share, seed, stratify):
    """
    Shuffles the labeled segments and cuts them into train, validation and test.

    stratify: If True, every road type is shuffled and cut on its own, so each split gets the same share of every
              road type. Otherwise all labeled segments are shuffled together.
    """
    generator = torch.Generator().manual_seed(seed)
    masks = _new_masks(y.shape[0])

    # The groups of segment indices to split. With stratify, one group per road type (in order of label).
    if stratify:
        groups = []
        for road_type in y[has_label].unique(sorted=True):
            groups.append((has_label & (y == road_type)).nonzero().flatten())
    else:
        groups = [has_label.nonzero().flatten()]

    for indices in groups:
        # Shuffle the group, then the first part goes to train, the next to validation and the rest to test
        indices = indices[torch.randperm(len(indices), generator=generator)]
        n_train, n_val, _ = _split_counts(len(indices), train_share, val_share)

        masks["train"][indices[:n_train]] = True
        masks["validation"][indices[n_train:n_train + n_val]] = True
        masks["test"][indices[n_train + n_val:]] = True

    return masks


def _segment_centroids(geometry):
    """
    Returns the position of each segment as a tensor with one (longitude, latitude) row per segment.
    The position is the mean of the segment's points.
    """
    centroids = []
    for points in geometry:
        lon_sum = 0
        lat_sum = 0
        for point in points:
            lon_sum = lon_sum + point[0]
            lat_sum = lat_sum + point[1]
        centroids.append((lon_sum / len(points), lat_sum / len(points)))

    return torch.tensor(centroids, dtype=torch.float64)


def _grid_cell_ids(centroids, cells_per_side):
    """
    Lays a cells_per_side x cells_per_side grid over the bounding box of all segments and returns the number of the
    grid cell each segment lies in.
    """
    # Scale every position to [0, 1] within the bounding box of the map
    low = centroids.min(dim=0).values
    span = (centroids.max(dim=0).values - low).clamp(min=MIN_SPAN)
    scaled = (centroids - low) / span

    # Column and row of the cell. A segment exactly on the far edge (scaled = 1) is put in the last cell.
    cell_xy = (scaled * cells_per_side).floor().long().clamp(0, cells_per_side - 1)

    # Turn (column, row) into a single cell number
    return cell_xy[:, 0] * cells_per_side + cell_xy[:, 1]


def _spatial_split(geometry, has_label, train_share, val_share, test_share, seed, cells_per_side):
    """
    Gives each whole grid cell to one split, so neighbouring segments end up in the same split.
    Cells are handed out in random order, each to the split that is furthest below its target size.
    """
    masks = _new_masks(has_label.shape[0])
    if not has_label.any():
        return masks

    cell_id = _grid_cell_ids(_segment_centroids(geometry), cells_per_side)

    # The labeled segments in each cell. Cells with no labeled segments are skipped.
    cells = []
    for cell in cell_id[has_label].unique():
        cells.append(((cell_id == cell) & has_label).nonzero().flatten())

    # How many labeled segments each split should get, and how many it has been given so far
    total = int(has_label.sum())
    targets = {"train": train_share * total, "validation": val_share * total, "test": test_share * total}
    assigned = {name: 0 for name in SPLIT_NAMES}

    generator = torch.Generator().manual_seed(seed)
    for position in torch.randperm(len(cells), generator=generator).tolist():
        members = cells[position]

        # Pick the split that is missing the largest share of its target. Splits with target 0 never get a cell.
        best = max(
            (name for name in SPLIT_NAMES if targets[name] > 0),
            key=lambda name: (targets[name] - assigned[name]) / targets[name],
        )

        masks[best][members] = True
        assigned[best] = assigned[best] + len(members)

    return masks


def _check_masks(masks, has_label):
    """
    Raises an AssertionError if the splits overlap, or if together they do not cover exactly the labeled segments.
    """
    times_used = masks["train"].int() + masks["validation"].int() + masks["test"].int()
    if (times_used > 1).any():
        raise AssertionError("A segment is in more than one split.")

    union = masks["train"] | masks["validation"] | masks["test"]
    if not torch.equal(union, has_label):
        raise AssertionError("The splits do not cover exactly the labeled segments.")


def make_split_masks(graph, seed=SPLIT_SEED, strategy=STRATEGY, stratify=True, cells_per_side=CELLS_PER_SIDE,
                     train_share=TRAIN_SHARE, val_share=VAL_SHARE):
    """
    Returns boolean masks {"train", "validation", "test"}, one value per segment of the full graph.

    By default, everything comes from the config, so every model uses the same split. Models should call it without
    arguments; the arguments are for trying other settings or for tests:
        masks = make_split_masks(graph)
        masks = make_split_masks(graph, strategy="spatial")
        masks = make_split_masks(graph, seed=1, strategy="spatial", train_share=0.6, val_share=0.2)

    graph:           RoadGraphBuilder, or the dictionary from get_static_components().
    seed:            The same seed always gives the same split. Defaults to SPLIT_SEED from the config.
    strategy:        "random" or "spatial". Defaults to STRATEGY from the config.
    stratify:        Random strategy only: split every road type separately.
    cells_per_side:  Spatial strategy only: size of the grid. More cells gives a split closer to random.
    train_share,
    val_share:       Default to TRAIN_SHARE and VAL_SHARE from the config. The test share is the rest.
    """
    if train_share < 0 or val_share < 0 or train_share + val_share > 1 + SHARE_TOLERANCE:
        raise ValueError(f"Invalid shares: train={train_share}, val={val_share}. Both must be >= 0 and sum to at most 1.")
    test_share = max(0.0, 1.0 - train_share - val_share)

    # Accept both a RoadGraphBuilder and the dictionary it returns
    if hasattr(graph, "get_static_components"):
        static = graph.get_static_components()
    else:
        static = graph

    y = static["y"]
    has_label = static["mask"].bool()

    if strategy == "random":
        masks = _random_split(y, has_label, train_share, val_share, seed, stratify)
    elif strategy == "spatial":
        masks = _spatial_split(static["geometry"], has_label, train_share, val_share, test_share, seed,
                               cells_per_side)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Use 'random' or 'spatial'.")

    _check_masks(masks, has_label)
    return masks
