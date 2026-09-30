# Machine Learning

Model training for road type prediction.

## Structure

```
src/ml/
├── data/
│   ├── file_loader.py          # Loads the road network adjacency list and road metadata
│   └── road_graph_builder.py   # Builds the line graph (road segments as nodes) and node features
├── config.py                   # Hyperparameters and device selection
├── logging_utils.py            # Shared logger (writes to artifacts/training_debug.log)
└── trainer.py                  # Training, validation, early stopping, saving results
```

Datasets go in `data/` and training outputs in `artifacts/` (models in `artifacts/models/`, result JSON in `artifacts/results/`). Both folders are git-ignored.

## Status

The previous travel-time imputation models and dataset have been removed. Still to do for road type prediction:

- `file_loader.py` / `road_graph_builder.py`: adapt to the new dataset's format.
- `road_graph_builder.py`: `road_type` is currently one-hot encoded as an input feature. It must become the label (`y`) instead, otherwise the model sees the answer.
- `trainer.py`: still uses the regression loss and metrics (MSE, MAPE, bias) and the time-series batch format. Switch to cross-entropy and accuracy / macro-F1.
- `config.py`: remove the time-series parameters (`SEQ_LEN`, `MASK_RATE`) once the trainer no longer uses them.
- Add a model and a training entry point.
