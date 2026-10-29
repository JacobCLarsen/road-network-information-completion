# Machine Learning

Model training for road type prediction.

## Structure

```
src/ml/
├── data/
│   ├── file_loader.py          # Loads the road network adjacency list and road metadata
│   └── road_graph_builder.py   # Builds the line graph (road segments as nodes) and node features
└── config.py                   # Hyperparameters and device selection
```

Datasets go in `data/` and training outputs in `artifacts/` (models in `artifacts/models/`, result JSON in `artifacts/results/`). Both folders are git-ignored.

## Status

The previous travel-time imputation models and dataset have been removed. Still to do for road type prediction:

- Put the road network files in `data/`: the edge connections CSV (`edge_id, vertex_start_id, vertex_end_id`) and the OSM roads JSON (`{way_id: {nodes, road_type, oneway}}`), same format as before.
- `road_graph_builder.py`: `road_type` is currently one-hot encoded as an input feature. It must become the label (`y`) instead, otherwise the model sees the answer.
- `config.py`: still holds the time-series parameters of the old travel-time trainer (removed). Replace them with the road type settings.
- Add a model and a training entry point.
