import math
from typing import Any, Dict, Iterable, List, Tuple

import torch


def _validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            f"Split ratios must sum to 1.0. Got train={train_ratio}, val={val_ratio}, "
            f"test={test_ratio} (total={total})."
        )

    for name, ratio in {"train": train_ratio, "val": val_ratio, "test": test_ratio}.items():
        if ratio < 0:
            raise ValueError(f"{name}_ratio must be non-negative. Got {ratio}.")


def _split_indices(num_nodes: int, train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    if num_nodes == 0:
        return {"train": [], "validation": [], "test": []}

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_nodes, generator=generator).tolist()

    train_count = int(round(num_nodes * train_ratio))
    val_count = int(round(num_nodes * val_ratio))
    test_count = num_nodes - train_count - val_count

    if train_count == 0 and num_nodes > 0:
        train_count = 1
        if train_count + val_count + test_count > num_nodes:
            val_count -= 1
    if val_count < 0:
        val_count = 0
    if test_count < 0:
        test_count = 0

    # Keep the counts consistent with the requested ratios even after clamping.
    while train_count + val_count + test_count != num_nodes:
        if train_count + val_count + test_count < num_nodes:
            test_count += 1
        else:
            if val_count > 0:
                val_count -= 1
            elif train_count > 1:
                train_count -= 1
            else:
                test_count -= 1

    train_idx = indices[:train_count]
    val_idx = indices[train_count:train_count + val_count]
    test_idx = indices[train_count + val_count:train_count + val_count + test_count]
    return {"train": train_idx, "validation": val_idx, "test": test_idx}


def _subset_static_graph(static_components: Dict[str, Any], keep_indices: Iterable[int], all_labels_masked: bool = True):
    keep_idx = sorted(int(i) for i in keep_indices)
    if not keep_idx:
        subset_edge_index = static_components["edge_index"].new_empty((2, 0))
        return {
            "x": static_components["x"][[]],
            "y": static_components["y"][[]],
            "mask": static_components["mask"][[]],
            "edge_index": subset_edge_index,
            "seg_key": [],
            "geometry": [],
            "edge_to_segment": {},
        }

    num_nodes = static_components["x"].shape[0]
    local_index = torch.full((num_nodes,), -1, dtype=torch.long)
    for local_pos, original_pos in enumerate(keep_idx):
        local_index[original_pos] = local_pos

    subset_x = static_components["x"][keep_idx]
    subset_y = static_components["y"][keep_idx]
    subset_mask = torch.ones_like(subset_y, dtype=torch.bool) if all_labels_masked else static_components["mask"][keep_idx]

    subset_seg_key = [static_components["seg_key"][idx] for idx in keep_idx]
    subset_geometry = [static_components["geometry"][idx] for idx in keep_idx]

    edge_index = static_components["edge_index"]
    if edge_index.numel() > 0:
        valid_edges = (local_index[edge_index[0]] >= 0) & (local_index[edge_index[1]] >= 0)
        subset_edge_index = torch.stack([
            local_index[edge_index[0][valid_edges]],
            local_index[edge_index[1][valid_edges]],
        ], dim=0)
    else:
        subset_edge_index = edge_index.clone()

    subset_edge_to_segment = {}
    for edge_id, original_index in static_components["edge_to_segment"].items():
        if original_index in keep_idx:
            subset_edge_to_segment[edge_id] = keep_idx.index(original_index)

    return {
        "x": subset_x,
        "y": subset_y,
        "mask": subset_mask,
        "edge_index": subset_edge_index,
        "seg_key": subset_seg_key,
        "geometry": subset_geometry,
        "edge_to_segment": subset_edge_to_segment,
    }


def split_graph_data(graph, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15,
                    seed: int = 42, all_labels_masked: bool = True):
    """Split a road graph into train/validation/test subsets.

    The returned dictionaries match the static graph structure emitted by RoadGraphBuilder and are ready
    to be passed into the rest of the training pipeline. Each subset keeps the same fields as the source
    graph but with a local node indexing for the graph edges.

    If all_labels_masked is True, every label in the training subset is marked as present because the road
    labels are treated as available in the dataset. This keeps the mask compatible with the existing trainer
    expectations while still preserving the true labels.
    """
    _validate_split_ratios(train_ratio, val_ratio, test_ratio)

    if hasattr(graph, "get_static_components"):
        static_components = graph.get_static_components()
    elif isinstance(graph, dict):
        static_components = graph
    else:
        raise TypeError(
            "graph must be a RoadGraphBuilder instance or a dictionary with the graph's static components."
        )

    if "x" not in static_components or "y" not in static_components:
        raise ValueError("The graph is missing the required graph tensors 'x' and 'y'.")

    num_nodes = static_components["x"].shape[0]
    split_indices = _split_indices(num_nodes, train_ratio, val_ratio, test_ratio, seed)

    return {
        subset_name: _subset_static_graph(static_components, indices, all_labels_masked=all_labels_masked)
        for subset_name, indices in split_indices.items()
    }
