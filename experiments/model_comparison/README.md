# Model Comparison

A simple comparison of road type models. Every model gets the same features, the same train/test split
and the same metrics.

| Model | Library |
|---|---|
| Random Forest | scikit-learn |
| Gradient Boosted Trees | scikit-learn (`HistGradientBoostingClassifier`) |
| Label Propagation | PyTorch Geometric |
| GCN | PyTorch Geometric |
| GraphSAGE | PyTorch Geometric |
| GAT | PyTorch Geometric |

RFN is not included yet, because it has no library implementation and needs extra graph data
(features between connected segments, such as turning angles).

## Setup

- Labelled segments are split randomly: 70% train, 30% test. Unlabelled segments stay in the graph.
- Features: `log(length)`, curvature, one-way and degree, standardized using the training segments.
- Metrics: accuracy and macro-F1 on the test segments. Macro-F1 is the main metric, because most roads are
  residential and accuracy alone hides how well the rare classes are predicted.
- The comparison is repeated with several random splits (`--seeds`) and reported as mean ± standard deviation.

## How to run

All commands are run in PowerShell from the repository root
(`road-network-information-completion/`), not from inside `experiments/`. Otherwise Python cannot find `src`.

### 1. Create a virtual environment (first time only)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

You should now see `(.venv)` at the start of the prompt. If PowerShell refuses to run the activate script,
run this once and try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 2. Install the packages (first time only)

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-ml.txt
```

The first line installs the CPU version of PyTorch, which is smaller and works on any machine.

### 3. Add the map data

Put the shapefile in a folder, for example `data/map`. The folder must contain `edges.shp` and its
companion files:

```
data/map/edges.shp
data/map/edges.dbf
data/map/edges.shx
```

The `data/` folder is git-ignored, so the data is not committed.

### 4. Run the comparison

```powershell
python -m experiments.model_comparison.compare_models data\map
```

Optional settings:

```powershell
python -m experiments.model_comparison.compare_models data\map --seeds 5 --epochs 300
```

| Option | Default | Meaning |
|---|---|---|
| `--seeds` | 3 | Number of random train/test splits to average over |
| `--epochs` | 200 | Training epochs for the graph neural networks |

The results are printed as a table and saved to `artifacts/results/model_comparison.csv`.

### Next time

Only activate the environment and run the script:

```powershell
.venv\Scripts\Activate.ps1
python -m experiments.model_comparison.compare_models data\map
```
