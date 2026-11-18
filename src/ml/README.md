# Machine Learning

Model training for road type prediction.

## Structure

```
src/ml/
├── data/
│   ├── database_loader.py      # Loads the road network and the labels from the database into road segments
│   ├── road_graph_builder.py   # Builds the line graph (road segments as nodes) and node features
│   └── split_graph.py          # Splits the labelled segments into train, validation and test, random or spatial (shared by all models)
├── models/
│   ├── majority/
│   │   └── run_majority.py     # Majority-class baseline: always predicts the most frequent training road type
│   ├── mlp/
│   │   ├── mlp_config.py       # MLP settings (hidden size, dropout, learning rate)
│   │   ├── mlp_model.py        # The MLP model: same shape as GraphSAGE, but only uses each segment's own features
│   │   ├── train_mlp.py        # Prepares the data and trains the MLP (used by the two run scripts)
│   │   ├── dev_run_mlp.py      # Development run: trains the MLP and scores it on the validation set
│   │   └── final_testrun_mlp.py # Final run: scores the MLP on the test set, only once at the very end
│   └── graphsage/
│       ├── graphsage_config.py # GraphSAGE settings (hidden size, dropout, learning rate)
│       ├── graphsage_model.py  # The GraphSAGE model
│       ├── train_graphsage.py  # Prepares the data and trains GraphSAGE (used by the two run scripts)
│       ├── dev_run_graphsage.py        # Development run: trains GraphSAGE and scores it on the validation set
│       └── final_testrun_graphsage.py  # Final run: scores GraphSAGE on the test set, only once at the very end
├── class_weights.py            # Shared class weights for the loss, so rare road types count more
├── config.py                   # Settings shared by all models (split, epochs)
└── evaluation.py               # Shared evaluation: macro-F1, accuracy, per-class and grouped scores
```

## Data

The road network and the labels are read from the database (`public.osm_edges` and
`attribute_prediction.segment_label`), using `DATABASE_URL` from `.env`. Before training, the database
needs the map from the data repo and the labels, which you make with:

```bash
python -m src.db.create_road_type_labels
```

See [src/db/README.md](../db/README.md) for how to set up the database. Training outputs go in `artifacts/`, which is git-ignored.

## Running GraphSAGE

From the repository root:

```bash
python -m src.ml.models.graphsage.dev_run_graphsage
```

Only once, at the very end, after all settings are chosen on the validation set:

```bash
python -m src.ml.models.graphsage.final_testrun_graphsage
```

## Running the MLP

The MLP is the non-graph counterpart of GraphSAGE: same layers, training loop, class weights and seeds, but it never looks at connected segments. Comparing the two shows how much the graph structure helps.

```bash
python -m src.ml.models.mlp.dev_run_mlp
```

Only once, at the very end:

```bash
python -m src.ml.models.mlp.final_testrun_mlp
```

## Running the majority-class baseline

```bash
python -m src.ml.models.majority.run_majority
```

It has no randomness and nothing to tune, so it runs once and is scored on both the validation and the test set. Every model should clearly beat its macro-F1.

## Configuration

- `src/ml/config.py` holds the settings that must be the same for every model (data path, train/validation share, split strategy and seed, number of epochs), so results can be compared fairly.
- Each model has its own config file in its folder for the settings that are tuned on the validation set (e.g. `src/ml/models/graphsage/graphsage_config.py`).

## Adding a model

Give each model its own folder in `models/` with a model file, a run script and its own config file (e.g. `graphsage_config.py`), like `models/graphsage/`. Use the shared settings from `src/ml/config.py` and score the predictions with `evaluate()` from `src/ml/evaluation.py`, always with the same list of road types, so the results are comparable.