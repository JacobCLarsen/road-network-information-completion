# src/ml/data/split_graph.py
"""
Train/validation/test split for the road line graph, done with masks instead of separate subgraphs.

The model always sees the complete graph (all segments, all connections, all features). The split only
decides which segment labels are used for the loss (train), for model selection (validation) and for the
final evaluation (test). This matches the real situation, where a segment with a missing road type is
surrounded by a complete network.

Two strategies are available:
    "random":  Random split of the labeled segments (stratified by road type by default).
               Gives an optimistic estimate, because missing tags in real data are not random.
    "spatial": Whole map cells go to the same split, so validation/test segments are located in areas
               where few of the neighbors are training segments. Gives a more realistic estimate.

Usage:
    builder = RoadGraphBuilder(segments)
    graph = builder.get_static_components()
    masks = make_split_masks(graph, strategy="random", seed=42)
    describe_split(graph, masks)

    out = model(graph["x"], graph["edge_index"])
    loss = F.cross_entropy(out[masks["train"]], graph["y"][masks["train"]])
"""

import math
from typing import Any, Dict, List, Optional

import torch

SPLIT_NAMES = ("train", "validation", "test")


def _validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    for name, ratio in {"train": train_ratio, "val": val_ratio, "test": test_ratio}.items():
        if ratio < 0:
            raise ValueError(f"{name}_ratio must be non-negative. Got {ratio}.")

    total = train_ratio + val_ratio + test_ratio
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            f"Split ratios must sum to 1.0. Got train={train_ratio}, val={val_ratio}, "
            f"test={test_ratio} (total={total})."
        )


def _get_static_components(graph) -> Dict[str, Any]:
    if hasattr(graph, "get_static_components"):
        static = graph.get_static_components()
    elif isinstance(graph, dict):
        static = graph
    else:
        raise TypeError("graph must be a RoadGraphBuilder or a dictionary with the static components.")

    for key in ("x", "y", "mask"):
        if key not in static:
            raise ValueError(f"The graph is missing the required component '{key}'.")
    return static


def _new_masks(num_nodes: int) -> Dict[str, torch.Tensor]:
    return {name: torch.zeros(num_nodes, dtype=torch.bool) for name in SPLIT_NAMES}


def _split_counts(count: int, train_ratio: float, val_ratio: float):
    """
    Number of items for train, validation and test out of `count` items. Test gets the remainder.
    If there are items at all, train gets at least one, so a road type never ends up missing from training
    just because it is rare.
    """
    if count == 0:
        return 0, 0, 0

    n_train = max(1, int(round(count * train_ratio)))
    n_train = min(n_train, count)
    n_val = min(int(round(count * val_ratio)), count - n_train)
    n_test = count - n_train - n_val
    return n_train, n_val, n_test


def _random_split(y: torch.Tensor, known: torch.Tensor, train_ratio: float, val_ratio: float,
                  seed: int, stratify: bool) -> Dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    masks = _new_masks(y.shape[0])

    if stratify:
        groups = [(known & (y == c)).nonzero().flatten() for c in y[known].unique(sorted=True)]
    else:
        groups = [known.nonzero().flatten()]

    for indices in groups:
        indices = indices[torch.randperm(len(indices), generator=generator)]
        n_train, n_val, _ = _split_counts(len(indices), train_ratio, val_ratio)
        masks["train"][indices[:n_train]] = True
        masks["validation"][indices[n_train:n_train + n_val]] = True
        masks["test"][indices[n_train + n_val:]] = True

    return masks


def _segment_centroids(geometry: List[list]) -> torch.Tensor:
    """Mean (longitude, latitude) of the points of each segment."""
    centroids = []
    for points in geometry:
        lon = sum(p[0] for p in points) / len(points)
        lat = sum(p[1] for p in points) / len(points)
        centroids.append((lon, lat))
    return torch.tensor(centroids, dtype=torch.float64).reshape(len(geometry), 2)


def _spatial_split(geometry: List[list], known: torch.Tensor, train_ratio: float, val_ratio: float,
                   test_ratio: float, seed: int, cells_per_side: int) -> Dict[str, torch.Tensor]:
    """
    Divides the map into a grid of cells_per_side x cells_per_side cells and gives each whole cell to one
    split. Cells are handed out in random order to the split that is furthest from its target size, so the
    final ratios are close to (but not exactly) the requested ones.
    """
    num_nodes = known.shape[0]
    masks = _new_masks(num_nodes)
    if not known.any():
        return masks

    centroids = _segment_centroids(geometry)
    low = centroids.min(dim=0).values
    span = (centroids.max(dim=0).values - low).clamp(min=1e-12)

    cell_xy = ((centroids - low) / span * cells_per_side).floor().long().clamp(0, cells_per_side - 1)
    cell_id = cell_xy[:, 0] * cells_per_side + cell_xy[:, 1]

    # Labeled segments per cell
    cells = []
    for cid in cell_id[known].unique():
        members = ((cell_id == cid) & known).nonzero().flatten()
        cells.append(members)

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(cells), generator=generator).tolist()

    total_known = int(known.sum())
    targets = {"train": train_ratio * total_known,
               "validation": val_ratio * total_known,
               "test": test_ratio * total_known}
    assigned = {name: 0 for name in SPLIT_NAMES}

    for pos in order:
        members = cells[pos]
        # Pick the split with the largest remaining share of its target (splits with target 0 are skipped)
        best = max(
            (name for name in SPLIT_NAMES if targets[name] > 0),
            key=lambda name: (targets[name] - assigned[name]) / targets[name],
        )
        masks[best][members] = True
        assigned[best] += len(members)

    return masks


def make_split_masks(graph, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15,
                     seed: int = 42, strategy: str = "random", stratify: bool = True,
                     cells_per_side: int = 20) -> Dict[str, torch.Tensor]:
    """
    Returns boolean masks {"train", "validation", "test"} over the nodes (road segments) of the full graph.

    Only segments that have a label (graph["mask"] is True) are part of the split. Segments without a
    label are in none of the masks: they are the ones that would be predicted in production.

    graph:           RoadGraphBuilder, or the dictionary from get_static_components().
    strategy:        "random" or "spatial" (see the module docstring).
    stratify:        Random strategy only. Splits every road type separately, so rare types appear in all splits.
    cells_per_side:  Spatial strategy only. Size of the grid. More cells means smaller blocks, and the
                     split moves closer to a random split.
    """
    _validate_split_ratios(train_ratio, val_ratio, test_ratio)
    static = _get_static_components(graph)

    y = static["y"]
    known = static["mask"].bool()
    if y.shape[0] != known.shape[0]:
        raise ValueError("'y' and 'mask' must have one entry per segment.")

    if strategy == "random":
        masks = _random_split(y, known, train_ratio, val_ratio, seed, stratify)
    elif strategy == "spatial":
        if "geometry" not in static:
            raise ValueError("The spatial strategy needs the 'geometry' component.")
        masks = _spatial_split(static["geometry"], known, train_ratio, val_ratio, test_ratio,
                               seed, cells_per_side)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Use 'random' or 'spatial'.")

    _check_masks(masks, known)
    return masks


def _check_masks(masks: Dict[str, torch.Tensor], known: torch.Tensor) -> None:
    """The splits must not overlap, and together they must cover exactly the labeled segments."""
    overlap = (masks["train"].int() + masks["validation"].int() + masks["test"].int()) > 1
    if overlap.any():
        raise AssertionError("A segment is in more than one split.")

    union = masks["train"] | masks["validation"] | masks["test"]
    if not torch.equal(union, known):
        raise AssertionError("The splits do not cover exactly the labeled segments.")


def describe_split(graph, masks: Dict[str, torch.Tensor], classes: Optional[List[str]] = None) -> None:
    """
    Prints the size of each split and the number of segments per road type in each split. Road types
    that are missing from the training split are flagged, since the model cannot learn them.
    """
    static = _get_static_components(graph)
    y = static["y"]

    total_known = int(static["mask"].sum())
    print(f"Segments: {y.shape[0]}  |  with label: {total_known}  |  without label: {y.shape[0] - total_known}")
    for name in SPLIT_NAMES:
        count = int(masks[name].sum())
        share = count / total_known if total_known else 0.0
        print(f"  {name:<11} {count:>8}  ({share:.1%})")

    print("\nSegments per road type (train / validation / test):")
    for c in y[static["mask"].bool()].unique(sorted=True).tolist():
        label = classes[c] if classes is not None else str(c)
        counts = [int(((y == c) & masks[name]).sum()) for name in SPLIT_NAMES]
        warning = "   <-- not in training" if counts[0] == 0 else ""
        print(f"  {label:<16} {counts[0]:>8} / {counts[1]:>8} / {counts[2]:>8}{warning}")


def save_masks(masks: Dict[str, torch.Tensor], path: str) -> None:
    """Saves the masks so every experiment can use the exact same split."""
    torch.save(masks, path)


def load_masks(path: str) -> Dict[str, torch.Tensor]:
    return torch.load(path)


def split_graph_data(graph, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15,
                     seed: int = 42, strategy: str = "random", stratify: bool = True,
                     cells_per_side: int = 20) -> Dict[str, Any]:
    """
    Convenience wrapper that keeps the old function name. Returns the unchanged full graph together
    with the split masks:

        {"graph": <static components>, "masks": {"train": ..., "validation": ..., "test": ...}}
    """
    static = _get_static_components(graph)
    masks = make_split_masks(static, train_ratio, val_ratio, test_ratio, seed, strategy, stratify,
                             cells_per_side)
    return {"graph": static, "masks": masks}