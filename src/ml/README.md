# Machine Learning

Model training for road type prediction.

## Structure

```
src/ml/
├── data/
│   ├── file_loader.py          # Loads the road network shapefile into road segments
│   ├── road_graph_builder.py   # Builds the line graph (road segments as nodes) and node features
│   └── split.py                # Splits the labelled segments into train, validation and test (shared by all models)
├── models/
│   └── graphsage/
│       ├── graphsage_config.py # GraphSAGE settings (hidden size, dropout, learning rate)
│       ├── graphsage_model.py  # The GraphSAGE model
│       ├── train_graphsage.py  # Prepares the data and trains GraphSAGE (used by the two run scripts)
│       ├── dev_run_graphsage.py        # Development run: trains GraphSAGE and scores it on the validation set
│       └── final_testrun_graphsage.py  # Final run: scores GraphSAGE on the test set, only once at the very end
├── class_weights.py            # Shared class weights for the loss, so rare road types count more
├── config.py                   # Settings shared by all models (data path, split, epochs)
└── evaluation.py               # Shared evaluation: macro-F1, accuracy, per-class and grouped scores
```

## Data

Put the Beijing map from the DRPK dataset in `data/bj_data/map/` (the folder must contain `edges.shp` and its companion files). Training outputs go in `artifacts/`. Both folders are git-ignored.

## Running GraphSAGE

From the repository root:

```bash
python -m src.ml.models.graphsage.dev_run_graphsage
```

Only once, at the very end, after all settings are chosen on the validation set:

```bash
python -m src.ml.models.graphsage.final_testrun_graphsage
```

## Configuration

- `src/ml/config.py` holds the settings that must be the same for every model (data path, train/validation share, split seed, number of epochs), so results can be compared fairly.
- Each model has its own config file in its folder for the settings that are tuned on the validation set (e.g. `src/ml/models/graphsage/graphsage_config.py`).

## Adding a model

Give each model its own folder in `models/` with a model file, a run script and its own config file (e.g. `graphsage_config.py`), like `models/graphsage/`. Use the shared settings from `src/ml/config.py` and score the predictions with `evaluate()` from `src/ml/evaluation.py`, always with the same list of road types, so the results are comparable.

## Status

Still to do for road type prediction:

- The train/validation/test split in `src/ml/data/split.py` is a temporary random split. It will be replaced by the group's final split.
- GraphSAGE: class weights, early stopping, the label trick and tuning.
