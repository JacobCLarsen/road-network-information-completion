# experiments/model_comparison/compare_models.py

"""
Simple comparison of road type models. Every model gets the same features and the same train/test split,
and is scored with the same metrics.

Run from the repository root:
    python -m experiments.model_comparison.compare_models data/map
    python -m experiments.model_comparison.compare_models data/map --seeds 5
"""

import argparse
import csv
import os
import statistics
import time
import warnings

# PyTorch Geometric uses torch.jit.script, which new PyTorch versions mark as deprecated. The warning is harmless.
warnings.filterwarnings("ignore", message=".*torch.jit.script.*", category=FutureWarning)

import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.nn import GATConv, GCNConv, LabelPropagation, SAGEConv

from src.ml.data.file_loader import ShapefileMapLoader
from src.ml.data.road_graph_builder import ROAD_CLASSES, RoadGraphBuilder

TRAIN_FRACTION = 0.7
HIDDEN_DIM = 64
DROPOUT = 0.3
LEARNING_RATE = 0.01
RESULTS_PATH = os.path.join("artifacts", "results", "model_comparison.csv")

# Road types with fewer labelled segments than this are left out of training and testing.
# No model can learn a class from a handful of examples, and they make macro-F1 very noisy.
MIN_CLASS_SIZE = 10


def remove_rare_classes(graph):
    """
    Marks segments of rare road types as unlabelled, so they are not used for training or testing.
    They stay in the graph as neighbours. Returns the new mask and the names of the removed classes.
    """
    labels = graph["y"][graph["mask"]]
    counts = torch.bincount(labels, minlength=len(ROAD_CLASSES))

    mask = graph["mask"].clone()
    removed = []
    for class_index in range(len(ROAD_CLASSES)):
        if 0 < counts[class_index] < MIN_CLASS_SIZE:
            mask[graph["y"] == class_index] = False
            removed.append(f"{ROAD_CLASSES[class_index]} ({int(counts[class_index])})")
    return mask, removed


def split_labelled_segments(mask, seed):
    """
    Splits the labelled segments randomly into a train and a test set.
    Unlabelled segments are in neither set, but they stay in the graph so the graph models can use them.
    """
    generator = torch.Generator().manual_seed(seed)
    labelled = mask.nonzero().view(-1)
    shuffled = labelled[torch.randperm(len(labelled), generator=generator)]
    n_train = int(TRAIN_FRACTION * len(shuffled))

    train_mask = torch.zeros(len(mask), dtype=torch.bool)
    test_mask = torch.zeros(len(mask), dtype=torch.bool)
    train_mask[shuffled[:n_train]] = True
    test_mask[shuffled[n_train:]] = True
    return train_mask, test_mask


def standardize(x, train_mask):
    """
    Scales every feature to mean 0 and standard deviation 1, using only the training segments.
    """
    mean = x[train_mask].mean(dim=0)
    std = x[train_mask].std(dim=0)
    std[std == 0] = 1
    return (x - mean) / std


# ----- Models -----
# Every run_* function trains on the training segments and returns the predicted class of every test segment.

def run_random_forest(data, seed):
    model = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1, class_weight="balanced")
    return run_sklearn_model(model, data)


def run_gradient_boosting(data, seed):
    model = HistGradientBoostingClassifier(random_state=seed, class_weight="balanced")
    return run_sklearn_model(model, data)


def run_sklearn_model(model, data):
    x = data["x"].numpy()
    y = data["y"].numpy()
    train = data["train_mask"].numpy()
    test = data["test_mask"].numpy()
    model.fit(x[train], y[train])
    return torch.tensor(model.predict(x[test]))


def run_label_propagation(data, seed):
    # Unlabelled segments have y = -1, which cannot be one-hot encoded. Their value is never used,
    # because only the training segments (given by mask) are used as known labels.
    known_labels = data["y"].clamp(min=0)
    model = LabelPropagation(num_layers=50, alpha=0.9)
    out = model(known_labels, data["edge_index"], mask=data["train_mask"])
    return out[data["test_mask"]].argmax(dim=1)


class GNN(torch.nn.Module):
    """
    Two graph layers: features -> hidden -> one score per road class.
    """

    def __init__(self, conv, in_dim, out_dim, heads=1):
        super().__init__()
        if heads > 1:
            # GAT: the first layer has several attention heads, whose outputs are concatenated
            self.conv1 = conv(in_dim, HIDDEN_DIM, heads=heads)
            self.conv2 = conv(HIDDEN_DIM * heads, out_dim, heads=1)
        else:
            self.conv1 = conv(in_dim, HIDDEN_DIM)
            self.conv2 = conv(HIDDEN_DIM, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=DROPOUT, training=self.training)
        return self.conv2(x, edge_index)


def run_gnn(conv, data, seed, epochs, heads=1, use_labels=False):
    torch.manual_seed(seed)
    x = data["x"]
    y = data["y"]
    edge_index = data["edge_index"]
    train_mask = data["train_mask"]
    test_mask = data["test_mask"]

    n_classes = len(ROAD_CLASSES)
    train_idx = train_mask.nonzero().view(-1)

    # Class weights, so rare road types count more in the loss
    counts = torch.bincount(y[train_mask], minlength=n_classes).float()
    class_weights = counts.sum() / (n_classes * counts.clamp(min=1))

    def model_input(known_idx):
        """Features, plus a one-hot of the road types treated as known if use_labels is on."""
        if not use_labels:
            return x
        onehot = torch.zeros(x.size(0), n_classes)
        onehot[known_idx, y[known_idx]] = 1
        return torch.cat([x, onehot], dim=1)

    in_dim = x.size(1) + (n_classes if use_labels else 0)
    model = GNN(conv, in_dim, n_classes, heads=heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        if use_labels:
            # Half the training labels are input, the other half are predicted,
            # so the model cannot just copy a segment's own label.
            perm = train_idx[torch.randperm(len(train_idx))]
            known, target = perm[: len(perm) // 2], perm[len(perm) // 2 :]
        else:
            known, target = train_idx, train_idx

        out = model(model_input(known), edge_index)
        loss = F.cross_entropy(out[target], y[target], weight=class_weights)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"    epoch {epoch + 1}/{epochs}, loss {loss.item():.3f}", flush=True)

    model.eval()
    with torch.no_grad():
        # At evaluation, all training labels are known. Test labels are never used as input.
        out = model(model_input(train_idx), edge_index)
    return out[test_mask].argmax(dim=1)


def get_models(epochs):
    return {
        "Random Forest": run_random_forest,
        "Gradient Boosted Trees": run_gradient_boosting,
        "Label Propagation": run_label_propagation,
        "GCN": lambda data, seed: run_gnn(GCNConv, data, seed, epochs),
        "GraphSAGE": lambda data, seed: run_gnn(SAGEConv, data, seed, epochs),
        "GraphSAGE + labels": lambda data, seed: run_gnn(SAGEConv, data, seed, epochs, use_labels=True),
        "GAT": lambda data, seed: run_gnn(GATConv, data, seed, epochs, heads=4),
    }


# ----- Comparison -----

def compare(graph, seeds, epochs):
    """
    Runs every model once per seed. Returns {model name: {"accuracy": [...], "macro_f1": [...], "time_s": [...]}}.
    """
    models = get_models(epochs)
    results = {}
    for name in models:
        results[name] = {"accuracy": [], "macro_f1": [], "time_s": []}

    for seed in range(seeds):
        train_mask, test_mask = split_labelled_segments(graph["mask"], seed)
        data = {
            "x": standardize(graph["x"], train_mask),
            "y": graph["y"],
            "edge_index": graph["edge_index"],
            "train_mask": train_mask,
            "test_mask": test_mask,
        }
        y_test = graph["y"][test_mask].numpy()

        for name, run in models.items():
            print(f"seed {seed}: training {name}...", flush=True)
            start = time.time()
            predictions = run(data, seed).numpy()
            elapsed = time.time() - start

            results[name]["accuracy"].append(accuracy_score(y_test, predictions))
            results[name]["macro_f1"].append(f1_score(y_test, predictions, average="macro", zero_division=0))
            results[name]["time_s"].append(elapsed)
            print(f"seed {seed}: {name} done in {elapsed:.1f}s", flush=True)

    return results


def mean_and_std(values):
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def print_table(results):
    print()
    print(f"{'Model':<25}{'Accuracy':>20}{'Macro-F1':>20}{'Time (s)':>12}")
    for name, scores in results.items():
        acc_mean, acc_std = mean_and_std(scores["accuracy"])
        f1_mean, f1_std = mean_and_std(scores["macro_f1"])
        time_mean, _ = mean_and_std(scores["time_s"])
        print(f"{name:<25}{acc_mean:>10.3f} +/- {acc_std:.3f}{f1_mean:>10.3f} +/- {f1_std:.3f}{time_mean:>12.1f}")


def save_csv(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std", "time_s_mean"])
        for name, scores in results.items():
            acc_mean, acc_std = mean_and_std(scores["accuracy"])
            f1_mean, f1_std = mean_and_std(scores["macro_f1"])
            time_mean, _ = mean_and_std(scores["time_s"])
            writer.writerow([name, f"{acc_mean:.4f}", f"{acc_std:.4f}", f"{f1_mean:.4f}", f"{f1_std:.4f}",
                             f"{time_mean:.2f}"])


def main():
    parser = argparse.ArgumentParser(description="Compare road type models on one shared split.")
    parser.add_argument("map_dir", help="Folder containing the edges shapefile (edges.shp)")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random splits to average over")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs for the graph neural networks")
    args = parser.parse_args()

    segments = ShapefileMapLoader(args.map_dir).get_segments()
    graph = RoadGraphBuilder(segments).get_static_components()
    graph["mask"], removed = remove_rare_classes(graph)
    print(f"{len(segments)} segments, {int(graph['mask'].sum())} labelled, "
          f"{graph['edge_index'].size(1)} connections")
    if removed:
        print(f"Left out road types with fewer than {MIN_CLASS_SIZE} segments: {', '.join(removed)}")

    results = compare(graph, args.seeds, args.epochs)
    print_table(results)
    save_csv(results, RESULTS_PATH)
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
