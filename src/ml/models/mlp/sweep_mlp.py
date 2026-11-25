# src/ml/models/mlp/sweep_mlp.py

"""Compare MLP hyperparameters using validation results across multiple model seeds."""

# Imports
import contextlib
import csv
import io
import itertools
from pathlib import Path
from src.ml import config
from src.ml.class_weights import compute_class_weights
from src.ml.data.road_graph_builder import ROAD_CLASSES
from src.ml.evaluation import evaluate, summarize_runs
from src.ml.models.mlp import mlp_config
from src.ml.models.mlp.train_mlp import prepare_data, train_mlp

# Choose which experiment to run: hidden_size, dropout, learning_rate, or class_weight_power.
SWEEP_NAME = "class_weight_power"

# Each experiment changes one hyperparameter while keeping the others fixed.
# Update the fixed values using the results from the preceding experiments.
SWEEPS = {
    "hidden_size": {
        "WORKING_SPACE": [32, 64, 128, 256],
        "DROPOUT": [0.5],
        "LEARNING_RATE": [0.01],
        "CLASS_WEIGHT_POWER": [1.0],
    },

    "dropout": {
        "WORKING_SPACE": [256],
        "DROPOUT": [0.0, 0.2, 0.5, 0.7],
        "LEARNING_RATE": [0.01],
        "CLASS_WEIGHT_POWER": [1.0],
    },

    "learning_rate": {
        "WORKING_SPACE": [256],
        "DROPOUT": [0.5],
        "LEARNING_RATE": [0.001, 0.005, 0.01, 0.02],
        "CLASS_WEIGHT_POWER": [1.0],
    },

    "class_weight_power": {
        "WORKING_SPACE": [256],
        "DROPOUT": [0.5],
        "LEARNING_RATE": [0.01],
        "CLASS_WEIGHT_POWER": [0.0, 0.5, 1.0],
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
    # MLP-specific settings belong to mlp_config; class-weight power is shared
    # with GraphSAGE and the other models, so it belongs to the shared config.
    setting_modules = {
        "WORKING_SPACE": mlp_config,
        "DROPOUT": mlp_config,
        "LEARNING_RATE": mlp_config,
        "CLASS_WEIGHT_POWER": config,
    }

    # Check the names and ensure every setting has at least one value to test.
    for name, values in SWEEP.items():
        if name not in setting_modules:
            raise ValueError(f"Unknown setting: {name}")
        if not values:
            raise ValueError(f"No values provided for {name}")

    # itertools.product() creates every combination of the value lists.
    # The * passes each list as a separate argument.
    # Example:
    # For the dropout sweep experiment, the combinations are (256, 0.0, 0.01, 1.0),
    # (256, 0.2, 0.01, 1.0), and so on.
    parameter_names = list(SWEEP)
    configurations = list(itertools.product(*(SWEEP[name] for name in parameter_names)))

    # Prepare data once so all configurations use the same split and preprocessing.
    # The unused fourth return value (_) is the test mask; tuning never uses it.
    # The initial class weights are also unused, since we calculate them for each configuration.
    graph, train_mask, val_mask, _, _, labels = prepare_data()

    # getattr() reads settings by their string names, e.g. "WORKING_SPACE".
    # Each setting is read from its corresponding module using setting_modules.
    # Save the original values before temporarily changing the imported modules.
    original_settings = {
        name: getattr(setting_modules[name], name) for name in parameter_names
    }
    experiment_rows = []

    # Run the experiments while ensuring that temporary configuration changes
    # are restored afterward, even if an experiment fails.
    try:
        # enumerate() provides both the configuration and its position in the list.
        # start=1 makes the numbering begin at 1 instead of 0.
        # Example: (1, (256, 0.5, 0.01, 0.0)), (2, (256, 0.5, 0.01, 0.5)), and so on.
        for number, values in enumerate(configurations, start=1):

            # zip() pairs each parameter name with its corresponding value.
            # Example: ["WORKING_SPACE", "CLASS_WEIGHT_POWER"] and (256, 0.5)
            # become ("WORKING_SPACE", 256) and ("CLASS_WEIGHT_POWER", 0.5).
            # setattr() then changes these settings in memory without editing the configuration files.
            for name, value in zip(parameter_names, values):
                setattr(setting_modules[name], name, value)

            # Create a readable description of the current configuration.
            # Example: [2/3] WORKING_SPACE=256, DROPOUT=0.5, LEARNING_RATE=0.01, CLASS_WEIGHT_POWER=0.5
            description = ", ".join(
                f"{name}={value}" for name, value in zip(parameter_names, values)
            )
            print(f"\n[{number}/{len(configurations)}] {description}")

            # Calculate the class weights using the current weighting power.
            # Power 1.0 uses the original inverse-frequency weights, 0.5 uses their
            # square roots, and 0.0 gives all present road types equal weight.
            # Only training labels are used, keeping validation and test labels untouched.
            class_weights = compute_class_weights(
                graph["y"][train_mask],
                len(ROAD_CLASSES),
                config.CLASS_WEIGHT_POWER,
            )

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
                "class_weight_power": config.CLASS_WEIGHT_POWER,
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
        # Restore the original settings in their corresponding modules,
        # including the shared class-weighting power.
        for name, value in original_settings.items():
            setattr(setting_modules[name], name, value)

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
            f"lr={row['learning_rate']}, power={row['class_weight_power']}: "
            f"macro-F1 {macro_f1}; best epochs {row['best_epochs']}"
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
