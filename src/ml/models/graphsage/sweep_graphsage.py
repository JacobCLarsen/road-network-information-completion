# src/ml/models/graphsage/sweep_graphsage.py

# Hyperparameter sweep: runs the development run (dev_run_graphsage.py) once for every combination of the settings in
# SWEEP, and prints one table with the validation scores of each combination. Use this while tuning, instead of
# changing graphsage_config.py by hand between runs.
# Like the development run, it only scores on the VALIDATION set. The test set is never used here.

import contextlib
import csv
import io
import itertools
import os

from src.ml import config
from src.ml.evaluation import evaluate, summarize_runs
from src.ml.models.graphsage import graphsage_config as sage_config
from src.ml.models.graphsage.train_graphsage import prepare_data, train_graphsage

# The settings to try. Every key must be the name of a setting in graphsage_config.py. Every combination of the values
# is trained with all seeds in config.MODEL_SEEDS, so the number of training runs is the product of the list lengths
# times the number of seeds: keep the lists short. Settings that are not listed keep their value from
# graphsage_config.py. Tune one setting at a time: sweep it, put the best value in graphsage_config.py, then sweep the
# next one.
SWEEP = {
    "WORKING_SPACE": [32, 64, 128, 256],
}

# The table is also saved here, so the results are kept after the terminal is closed. The artifacts folder is git-ignored.
RESULTS_FILE = "artifacts/graphsage_sweep.csv"


def _format_score(score):
    # "0.4612 ± 0.0070", the same format as the development run
    return f"{score['mean']:.4f} ± {score['std']:.4f}" if score["std"] is not None else f"{score['mean']:.4f}"


if __name__ == "__main__":
    # Check the names first, so a typo stops the sweep at once instead of silently training with the default value
    for name in SWEEP:
        if not hasattr(sage_config, name):
            raise ValueError(f"{name} is not a setting in graphsage_config.py")

    # The data is the same for every combination and seed, so it is prepared once.
    # The _ is the test mask. It is not given a name, because it must not be used during tuning.
    graph, train_mask, val_mask, _, class_weights, labels = prepare_data()

    # The original values, so graphsage_config.py's settings can be put back after the sweep
    original = {name: getattr(sage_config, name) for name in SWEEP}

    names = list(SWEEP)
    combinations = list(itertools.product(*SWEEP.values()))
    rows = []

    try:
        for number, values in enumerate(combinations, start=1):
            # train_graphsage reads the settings from graphsage_config when it is called, so changing them here
            # changes the model it trains
            for name, value in zip(names, values):
                setattr(sage_config, name, value)

            setting_text = ", ".join(f"{name} = {value}" for name, value in zip(names, values))
            print(f"[{number}/{len(combinations)}] {setting_text}")

            results = []
            best_epochs = []
            for seed in config.MODEL_SEEDS:
                # Hide the two lines per epoch that train_graphsage prints, so the sweep output stays readable
                with contextlib.redirect_stdout(io.StringIO()):
                    predictions, best_epoch = train_graphsage(graph, train_mask, val_mask, class_weights, labels, seed)

                result = evaluate(graph["y"][val_mask], predictions[val_mask], labels)
                results.append(result)
                best_epochs.append(best_epoch)
                print(f"  Seed {seed}: best epoch {best_epoch}, validation macro-F1 {result['macro_f1']:.4f}")

            summary = summarize_runs(results)
            rows.append({"values": values, "summary": summary, "best_epochs": best_epochs})
            print(f"  Validation macro-F1: {_format_score(summary['macro_f1'])}")
    finally:
        # Put the original settings back, also if the sweep was stopped with Ctrl+C or crashed
        for name, value in original.items():
            setattr(sage_config, name, value)

    # Best macro-F1 first. A combination is only really better than another if the difference is clearly bigger than
    # their standard deviations.
    rows.sort(key=lambda row: row["summary"]["macro_f1"]["mean"], reverse=True)

    headers = names + ["macro-F1 (mean ± std)", "accuracy (mean ± std)", "best epochs"]
    table = [[str(value) for value in row["values"]]
             + [_format_score(row["summary"]["macro_f1"]), _format_score(row["summary"]["accuracy"]),
                ", ".join(str(epoch) for epoch in row["best_epochs"])]
             for row in rows]

    widths = [max(len(header), *(len(line[i]) for line in table)) for i, header in enumerate(headers)]
    print()
    print(f"Sweep results on the validation set ({len(config.MODEL_SEEDS)} seeds each, best first):")
    print("  ".join(header.ljust(width) for header, width in zip(headers, widths)))
    for line in table:
        print("  ".join(cell.ljust(width) for cell, width in zip(line, widths)))
    print("Best epochs close to config.EPOCHS mean the model was still improving when training stopped.")

    # Save the table, with mean and standard deviation in separate columns so it is easy to use in a spreadsheet
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(names + ["macro_f1_mean", "macro_f1_std", "accuracy_mean", "accuracy_std", "best_epochs"])
        for row in rows:
            summary = row["summary"]
            writer.writerow(list(row["values"])
                            + [summary["macro_f1"]["mean"], summary["macro_f1"]["std"],
                               summary["accuracy"]["mean"], summary["accuracy"]["std"],
                               " ".join(str(epoch) for epoch in row["best_epochs"])])
    print(f"Saved to {RESULTS_FILE}")
