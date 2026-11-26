# src/ml/models/graphsage/sweep_graphsage.py

# Hyperparameter sweep: retests every GraphSAGE setting in SWEEP and prints which values scored best on the
# validation set. Use this while tuning, instead of changing graphsage_config.py by hand between runs.
# Like the development run, it only scores on the VALIDATION set. The test set is never used here.
#
# How it searches: trying every combination of every setting would take far too long (3 * 3 * 3 * 3 = 81
# combinations, each trained with every seed for config.EPOCHS epochs). Instead it tunes one setting at a time, in the
# order of SWEEP: it tries every value of the first setting, keeps the best one, then tries every value of the next
# setting with that best value in place, and so on. The settings affect each other, so after the last setting it
# starts again from the first one with all the best values in place (a new round). It stops after ROUNDS rounds, or
# earlier when a whole round changes no value. A combination that was already trained is never trained again.
#
# Every run trains for config.EPOCHS epochs, the same as the dev and final runs, so the results hold for the model
# that is actually used.

import contextlib
import csv
import io
import os

from src.ml import config
from src.ml.evaluation import evaluate, summarize_runs
from src.ml.models.graphsage import graphsage_config as sage_config
from src.ml.models.graphsage.train_graphsage import prepare_data, train_graphsage

# The settings to retest and the values to try for each, swept in this order. Every key must be the name of a setting
# in graphsage_config.py. The search starts from the values in graphsage_config.py. Settings that are not listed keep
# their value from graphsage_config.py.
# Learning rate goes first, because it decides how far the model gets in config.EPOCHS epochs and so changes what the
# other settings look like. CLASS_WEIGHT_POWER is not here: it is in src/ml/config.py and shared with the other models.
# 0.001 and 512 were added for the sweep of 26 Nov 2026 (see graphsage_config.py): group hiding makes GROUPS weight
# updates per epoch instead of 1, so a lower learning rate might have been better, and 256 was the largest hidden size
# tried before.
SWEEP = {
    "LEARNING_RATE": [0.001, 0.005, 0.01],
    "WEIGHT_DECAY": [0, 5e-4, 5e-3],
    "DROPOUT": [0.2, 0.4, 0.6],
    "WORKING_SPACE": [128, 256, 512],
}

# The most rounds to run. A round tries every value of every setting once. 2 would check that the best value of the
# first settings still holds once the later settings have changed, but 1 keeps the sweep short enough to run.
ROUNDS = 1

# The seeds each combination is trained with. Kept separate from config.MODEL_SEEDS, which is shared with the other
# models: the sweep needs several seeds to tell real differences from noise, also when the dev runs use only one.
SWEEP_SEEDS = [0, 1, 2]

# The table is also saved here, so the results are kept after the terminal is closed. The artifacts folder is git-ignored.
# It is saved again after every combination, so the results so far are kept if the sweep is stopped early.
RESULTS_FILE = "artifacts/graphsage_sweep.csv"


def _format_score(score):
    # "0.4612 ± 0.0070", the same format as the development run
    return f"{score['mean']:.4f} ± {score['std']:.4f}" if score["std"] is not None else f"{score['mean']:.4f}"


def _train_combination(data, values):
    # Trains GraphSAGE with these settings once per seed in SWEEP_SEEDS and returns one result row:
    # {"values": the setting values in the order of SWEEP, "summary": summarize_runs() result, "best_epochs": [...]}
    graph, train_mask, val_mask, class_weights, labels = data

    # train_graphsage reads the settings from graphsage_config when it is called, so changing them here changes the
    # model it trains
    for name, value in zip(SWEEP, values):
        setattr(sage_config, name, value)

    results = []
    best_epochs = []
    for seed in SWEEP_SEEDS:
        # Hide the two lines per epoch that train_graphsage prints, so the sweep output stays readable
        with contextlib.redirect_stdout(io.StringIO()):
            predictions, best_epoch = train_graphsage(graph, train_mask, val_mask, class_weights, labels, seed)

        result = evaluate(graph["y"][val_mask], predictions[val_mask], labels)
        results.append(result)
        best_epochs.append(best_epoch)
        print(f"    Seed {seed}: best epoch {best_epoch}, validation macro-F1 {result['macro_f1']:.4f}")

    summary = summarize_runs(results)
    print(f"    Validation macro-F1: {_format_score(summary['macro_f1'])}")
    return {"values": values, "summary": summary, "best_epochs": best_epochs}


def _save_results(rows):
    # Saves every combination trained so far, best first, with mean and standard deviation in separate columns so it
    # is easy to use in a spreadsheet
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(list(SWEEP) + ["macro_f1_mean", "macro_f1_std", "accuracy_mean", "accuracy_std", "best_epochs"])
        for row in sorted(rows, key=lambda row: row["summary"]["macro_f1"]["mean"], reverse=True):
            summary = row["summary"]
            writer.writerow(list(row["values"])
                            + [summary["macro_f1"]["mean"], summary["macro_f1"]["std"],
                               summary["accuracy"]["mean"], summary["accuracy"]["std"],
                               " ".join(str(epoch) for epoch in row["best_epochs"])])


def _print_table(rows):
    # All combinations trained, best macro-F1 first. A combination is only really better than another if the
    # difference is clearly bigger than their standard deviations.
    rows = sorted(rows, key=lambda row: row["summary"]["macro_f1"]["mean"], reverse=True)
    headers = list(SWEEP) + ["macro-F1 (mean ± std)", "accuracy (mean ± std)", "best epochs"]
    table = [[str(value) for value in row["values"]]
             + [_format_score(row["summary"]["macro_f1"]), _format_score(row["summary"]["accuracy"]),
                ", ".join(str(epoch) for epoch in row["best_epochs"])]
             for row in rows]

    widths = [max(len(header), *(len(line[i]) for line in table)) for i, header in enumerate(headers)]
    print("  ".join(header.ljust(width) for header, width in zip(headers, widths)))
    for line in table:
        print("  ".join(cell.ljust(width) for cell, width in zip(line, widths)))


if __name__ == "__main__":
    # Check the names first, so a typo stops the sweep at once instead of silently training with the default value
    for name in SWEEP:
        if not hasattr(sage_config, name):
            raise ValueError(f"{name} is not a setting in graphsage_config.py")

    names = list(SWEEP)
    print(f"Sweeping {', '.join(names)} with {len(SWEEP_SEEDS)} seeds and {config.EPOCHS} epochs per run, "
          f"at most {ROUNDS} round(s)")

    # The data is the same for every combination and seed, so it is prepared once.
    # The _ is the test mask. It is not given a name, because it must not be used during tuning.
    graph, train_mask, val_mask, _, class_weights, labels = prepare_data()
    data = (graph, train_mask, val_mask, class_weights, labels)

    # The original values, so graphsage_config.py's settings can be put back after the sweep. They are also where the
    # search starts.
    original = {name: getattr(sage_config, name) for name in names}
    best = dict(original)

    # Every combination trained so far, by its values in the order of SWEEP, so none is trained twice
    trained = {}

    try:
        for round_number in range(1, ROUNDS + 1):
            changed = False
            for name in names:
                print()
                print(f"Round {round_number}, {name} (others: "
                      + ", ".join(f"{other} = {best[other]}" for other in names if other != name) + ")")

                # One row per value of this setting, with all other settings at their best value so far
                step_rows = []
                for value in SWEEP[name]:
                    values = tuple(value if other == name else best[other] for other in names)
                    if values in trained:
                        print(f"  {name} = {value}: already trained, "
                              f"validation macro-F1 {_format_score(trained[values]['summary']['macro_f1'])}")
                    else:
                        print(f"  {name} = {value}")
                        trained[values] = _train_combination(data, values)
                        _save_results(trained.values())
                    step_rows.append((value, trained[values]))

                # Keep the value with the best mean validation macro-F1
                best_value = max(step_rows, key=lambda pair: pair[1]["summary"]["macro_f1"]["mean"])[0]
                if best_value != best[name]:
                    changed = True
                print(f"  Best {name}: {best_value}")
                best[name] = best_value

            if not changed:
                print()
                print(f"Round {round_number} changed no value, so the search has settled")
                break
    finally:
        # Put the original settings back, also if the sweep was stopped with Ctrl+C or crashed
        for name, value in original.items():
            setattr(sage_config, name, value)

    print()
    print(f"All combinations trained ({len(trained)}, {len(SWEEP_SEEDS)} seeds each, validation set, best first):")
    _print_table(trained.values())
    print("Best epochs close to config.EPOCHS mean the model was still improving when training stopped.")
    print()
    print("Best settings found (put these in graphsage_config.py):")
    for name in names:
        print(f"  {name} = {best[name]}")
    print(f"Saved to {RESULTS_FILE}")
