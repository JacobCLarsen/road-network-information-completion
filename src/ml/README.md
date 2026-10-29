# Machine Learning

Model training for road type prediction.

## Structure

```
src/ml/
├── data/
│   ├── file_loader.py          # Loads the road network shapefile into road segments
│   └── road_graph_builder.py   # Builds the line graph (road segments as nodes) and node features
├── models/
│   └── graphsage/
│       ├── config.py           # GraphSAGE settings (hidden size, dropout, learning rate)
│       ├── graphsage.py        # The GraphSAGE model
│       └── run_graphsage.py    # Trains GraphSAGE and scores it on the validation set
├── config.py                   # Settings shared by all models (data path, split, epochs)
└── evaluation.py               # Shared evaluation: macro-F1, accuracy, per-class and grouped scores
```

## Data

Put the Beijing map from the DRPK dataset in `data/bj_data/map/` (the folder must contain `edges.shp` and its companion files). Training outputs go in `artifacts/`. Both folders are git-ignored.

## Running GraphSAGE

From the repository root:

```bash
python -m src.ml.models.graphsage.run_graphsage
```

## Configuration

- `src/ml/config.py` holds the settings that must be the same for every model (data path, train/validation share, split seed, number of epochs), so results can be compared fairly.
- Each model has its own `config.py` in its folder for the settings that are tuned on the validation set (e.g. `src/ml/models/graphsage/config.py`).

## Adding a model

Give each model its own folder in `models/` with a model file, a run script and a `config.py`, like `models/graphsage/`. Use the shared settings from `src/ml/config.py` and score the predictions with `evaluate()` from `src/ml/evaluation.py`, always with the same list of road types, so the results are comparable.

## Status

Still to do for road type prediction:

- The train/validation/test split in `run_graphsage.py` is temporary. It will be replaced by a shared split in `src/ml/data/split.py`.
- Class set: merge `*_link` types into their parent type and drop very rare types, as described in the report.
- GraphSAGE: class weights, early stopping, the label trick and tuning.
