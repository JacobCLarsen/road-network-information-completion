# src/ml/models/mlp/sweep_mlp.py

"""Compare MLP hyperparameters using validation results across multiple model seeds."""

"""Imports"""
import contextlib
import csv
import io
import itertools
from pathlib import Path
from src.ml import config
from src.ml.data.road_graph_builder import ROAD_CLASSES
from src.ml.evaluation import evaluate, summarize_runs
from src.ml.models.mlp import mlp_config
from src.ml.models.mlp.train_mlp import prepare_data, train_mlp

# Select the experiment to run: "hidden_size", "dropout", or "learning_rate".
SWEEP_NAME = "hidden_size"

# Each experiment changes one hyperparameter while keeping the others fixed.
# After the dropout sweep, replace 0.5 in the learning-rate sweep with the selected value.
SWEEPS = {
    "hidden_size": {
        "WORKING_SPACE": [32, 64, 128, 256],
        "DROPOUT": [0.5],
        "LEARNING_RATE": [0.01],
    },

    "dropout": {
        "WORKING_SPACE": [256],
        "DROPOUT": [0.0, 0.2, 0.5, 0.7],
        "LEARNING_RATE": [0.01],
    },

    "learning_rate": {
        "WORKING_SPACE": [256],
        "DROPOUT": [0.5],  # Update after evaluating the dropout results.
        "LEARNING_RATE": [0.001, 0.005, 0.01, 0.02],
    },
}

# Select the values to test and use a separate CSV file for each experiment.
SWEEP = SWEEPS[SWEEP_NAME]
RESULTS_FILE = Path(f"artifacts/mlp_{SWEEP_NAME}_sweep.csv")

"""Format a metric as mean ± standard deviation across model seeds."""
def format_score(score):
    if score["std"] is None:
        return f"{score['mean']:.4f}"
    return f"{score['mean']:.4f} ± {score['std']:.4f}"


"""Train each configuration, compare validation performance, and save the results."""
def main():
    # Check that the experiment uses supported MLP settings with at least one value.
    for name, values in SWEEP.items():
        if name not in ("WORKING_SPACE", "DROPOUT", "LEARNING_RATE"):
            raise ValueError(f"Unknown MLP setting: {name}")
        if not values:
            raise ValueError(f"No values provided for {name}")

    # itertools.product() creates every combination of the value lists.
    # The * passes each list as a separate argument.
    # Example:
    # For the dropout sweep experiment, the combinations are (256, 0.0, 0.01), (256, 0.2, 0.01), and so on.
    parameter_names = list(SWEEP)
    configurations = list(itertools.product(*(SWEEP[name] for name in parameter_names)))

    # Prepare data once so all configurations use the same split and preprocessing.
    # The unused fourth return value (_) is the test mask; tuning never uses it.
    graph, train_mask, val_mask, _, class_weights, labels = prepare_data()

    # getattr() reads settings by their string names, e.g. "WORKING_SPACE".
    # Save the original values before temporarily changing the imported module.
    original_settings = {name: getattr(mlp_config, name) for name in parameter_names}
    experiment_rows = []

    # Run the experiments while ensuring that temporary configuration changes
    # are restored afterward, even if an experiment fails.
    try:
        # enumerate() provides both the configuration and its position in the list.
        # start=1 makes the numbering begin at 1 instead of 0.
        # Example: (1, (256, 0.0, 0.01)), (2, (256, 0.2, 0.01)), and so on.
        for number, values in enumerate(configurations, start=1):
            # zip() pairs each parameter name with its corresponding value.
            # Example: ["WORKING_SPACE", "DROPOUT"] and (256, 0.2)
            # become ("WORKING_SPACE", 256) and ("DROPOUT", 0.2).
            # setattr() then changes these settings in memory without editing mlp_config.py.
            for name, value in zip(parameter_names, values):
                setattr(mlp_config, name, value)

            # Create a readable description of the current configuration.
            # Example: [2/4] WORKING_SPACE=256, DROPOUT=0.2, LEARNING_RATE=0.01
            description = ", ".join(
                f"{name}={value}" for name, value in zip(parameter_names, values)
            )
            print(f"\n[{number}/{len(configurations)}] {description}")

            # Store one validation result and selected epoch per model seed.
            validation_results = []
            best_epochs = []

            # Train the current configuration with each predefined seed
            # to measure how much its performance varies between runs.
            for seed in config.MODEL_SEEDS:
                # train_mlp() creates and trains a new model for the current seed.
                # StringIO() temporarily captures the training output so that
                # the terminal only displays our results instead of every epoch.
                with contextlib.redirect_stdout(io.StringIO()):
                    predictions, best_epoch = train_mlp(
                        graph, train_mask, val_mask, class_weights, labels, seed
                    )

                # Evaluate on the validation set only, never on test labels.
                scores = evaluate(
                    graph["y"][val_mask],
                    predictions[val_mask],
                    labels,
                    class_names=ROAD_CLASSES,
                )

                # Collect this seed's evaluation metrics and selected epoch
                # so we can compare the results across all five runs.
                validation_results.append(scores)
                best_epochs.append(best_epoch)

                # Display the validation macro-F1 and best epoch for the current seed.
                # Example: Seed 0: macro-F1 0.2689, best epoch 31
                print(
                    f"  Seed {seed}: macro-F1 {scores['macro_f1']:.4f}, "
                    f"best epoch {best_epoch}"
                )

            # Summarize performance across all model seeds for this configuration.
            summary = summarize_runs(validation_results)
            print(f"  Mean validation macro-F1: {format_score(summary['macro_f1'])}")

            # Each dictionary becomes one CSV row containing settings and metrics.
            experiment_rows.append({
                "working_space": mlp_config.WORKING_SPACE,
                "dropout": mlp_config.DROPOUT,
                "learning_rate": mlp_config.LEARNING_RATE,
                "epochs": config.EPOCHS,
                "model_seeds": " ".join(map(str, config.MODEL_SEEDS)),
                "split_seed": config.SPLIT_SEED,
                "split_strategy": config.STRATEGY,
                "train_share": config.TRAIN_SHARE,
                "val_share": config.VAL_SHARE,
                "macro_f1_mean": summary["macro_f1"]["mean"],
                "macro_f1_std": summary["macro_f1"]["std"],
                "macro_f1_by_seed": " ".join(
                    f"{result['macro_f1']:.4f}" for result in validation_results
                ),
                "accuracy_mean": summary["accuracy"]["mean"],
                "accuracy_std": summary["accuracy"]["std"],
                "grouped_macro_f1_mean": summary["grouped_macro_f1"]["mean"],
                "grouped_macro_f1_std": summary["grouped_macro_f1"]["std"],
                "best_epochs": " ".join(map(str, best_epochs)),
            })

    # The finally block always runs after the try block, even if training fails.
    # This prevents the sweep from leaving temporary hyperparameter values in memory.
    finally:
        # Restore the original in-memory MLP settings even if an experiment fails.
        for name, value in original_settings.items():
            setattr(mlp_config, name, value)

    # lambda tells sort() to use each row's macro-F1 mean; reverse=True puts the best first.
    experiment_rows.sort(key=lambda row: row["macro_f1_mean"], reverse=True)

    print("\nValidation results (highest mean macro-F1 first):")

    # Go through the configurations in their newly sorted order,
    # starting with the one that achieved the highest mean validation macro-F1.
    for row in experiment_rows:
        # Format the mean macro-F1 and standard deviation into a readable string.
        # Example: 0.2692 ± 0.0040
        macro_f1 = format_score({
            "mean": row["macro_f1_mean"],
            "std": row["macro_f1_std"],
        })

        # Display the current configuration together with its validation
        # performance and the best epochs selected across the model seeds.
        print(
            f"  hidden={row['working_space']}, dropout={row['dropout']}, "
            f"lr={row['learning_rate']}: macro-F1 {macro_f1}; "
            f"best epochs {row['best_epochs']}"
        )

    # DictWriter uses the dictionary keys as CSV column names and writes one
    # row per configuration. The directory is created if it does not exist.
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Open the CSV file for writing and save all evaluated configurations.
    # writeheader() writes the column names, while writerows() writes the results.
    # An existing file with the same name will be overwritten.
    with RESULTS_FILE.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=experiment_rows[0].keys())
        writer.writeheader()
        writer.writerows(experiment_rows)

    # Confirm where the results were saved so they can be found later.
    print(f"\nSaved results to {RESULTS_FILE}")

# Run the sweep when executed as a script, but not when imported by another file.
if __name__ == "__main__":
    main()
