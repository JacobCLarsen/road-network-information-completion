# Road Network Information Completion

Road type prediction for road network segments, combining graph machine learning with a production-ready API microservice.

## Repository Structure

```
road-network-information-completion/
├── src/
│   ├── db/                     # Data layer: connection, ORM models, repositories (see src/db/README.md)
│   ├── api/                    # FastAPI microservice (see src/api/README.md)
│   │   ├── routes/             # API endpoints
│   │   ├── services/           # Business logic
│   │   ├── tables/             # SQLAlchemy ORM tables
│   │   └── schemas/            # Pydantic request/response models
│   └── ml/                     # Model training (see src/ml/README.md)
│       ├── data/               # Dataset loading and road graph building
│       ├── models/             # One folder per model (e.g. graphsage/)
│       ├── config.py           # Settings shared by all models
│       └── evaluation.py       # Shared evaluation (macro-F1 etc.)
├── tests/
│   ├── db/                     # Data layer tests (unit + tests against a real database)
│   ├── api/                    # API unit + integration tests
│   └── ml/                     # ML unit tests
├── data/                       # Local datasets (git-ignored)
├── artifacts/                  # Trained models, results and logs (git-ignored)
├── k8s/                        # Kubernetes manifests
├── .github/workflows/          # CI/CD pipelines
├── requirements.txt            # API + test dependencies
└── requirements-ml.txt         # ML dependencies (torch, torch_geometric)
```

## Database

The database is PostGIS. It runs from the [astep-aau/data](https://github.com/astep-aau/data) repo,
which also owns the migrations (the tables), so this repo never creates tables itself.

### Which database do we connect to?

Everything uses the `DATABASE_URL` setting. It is read from an environment variable first, and if
that is not set, from the `.env` file in the repo root.

| Where | Where DATABASE_URL comes from | Database |
|---|---|---|
| Your machine | `.env` (copy from `.env.example`) | local docker database on `localhost:5432` |
| Your machine, against the server | `.env` with port 5433 + `kubectl port-forward` | server database |
| Server (Kubernetes) | the secret `db-secret` | `dataset-postgres` in namespace `main` |

`.env` is not in git and not in the docker image, so local settings never end up on the server.

### Local setup

```bash
# Start the database (PostGIS + migrations + data import) from the data repo
(cd ../data && cp .env.example .env && docker compose up -d --build)

# Point this repo at it (the defaults are for the local database)
cp .env.example .env

# Check the connection
python -m src.db.check_connection

# Save the road type classes and the label of every segment (only needed once per database)
python -m src.db.create_road_type_labels
```

### Using the server database from your machine

Needs a kubeconfig for the cluster (ask the group).

```bash
# Forward the server database to localhost:5433 (keep this terminal open)
kubectl port-forward svc/dataset-postgres -n main 5433:5432
```

Then set `DATABASE_URL=postgresql+asyncpg://USER:PASSWORD@localhost:5433/data` in `.env` and run
`python -m src.db.check_connection`. The username and password are in the secret `postgres-secret`
in namespace `main`. Change the port back to 5432 to use the local database again.

### Setting DATABASE_URL on the server

The API on the server reads `DATABASE_URL` from the Kubernetes secret `db-secret`. It is not pushed
with the code, you create it once with kubectl (and again only if the password changes):

```bash
kubectl create secret generic db-secret -n cs-25-sw-5-06 \
  --from-literal=DATABASE_URL='postgresql+asyncpg://USER:PASSWORD@dataset-postgres.main.svc.cluster.local:5432/data'
```

If it already exists, delete it first with `kubectl delete secret db-secret -n cs-25-sw-5-06`. Then
restart the API so it reads the new value:

```bash
kubectl rollout restart deployment/road-network-information-completion -n cs-25-sw-5-06
```

You can also copy `k8s/secret.example.yaml` to `k8s/secret.yaml` (git-ignored), fill it in and run
`kubectl apply -f k8s/secret.yaml`.

When the API starts it checks that the database has run the migrations it needs, and it does not
start if not. See [src/db/README.md](src/db/README.md) for the data layer and how to use it.

## Quick Start

### API Microservice
```bash
# Install dependencies
pip install -r requirements.txt

# Set up the database first (see "Database" above)

# Run locally
uvicorn src.api.main:app --reload
```

See [src/api/README.md](src/api/README.md) for detailed setup instructions and
[src/db/README.md](src/db/README.md) for the data layer (database connection and operations).

### Machine Learning
```bash
pip install -r requirements-ml.txt

# Train and evaluate GraphSAGE (needs the database with the map and labels, see "Database" above)
python -m src.ml.models.graphsage.dev_run_graphsage
```

See [src/ml/README.md](src/ml/README.md).

## Testing

```bash
# API tests
pytest tests/api/ -v

# Data layer tests (the ones that need a database use TEST_DATABASE_URL from .env,
# and everything they do is rolled back)
pytest tests/db/ -v

# ML tests (requires requirements-ml.txt)
pytest tests/ml/ -v

# With coverage report
pytest tests/api/ --cov=src/api --cov-report=html
```

## Deployment

The API microservice is automatically deployed to Kubernetes on every push to `main`:
- Docker image: `ghcr.io/jacobclarsen/road-network-information-completion:latest`
- Namespace: `cs-25-sw-5-06`
- Service: `road-network-information-completion-service`
- Database connection: the secret `db-secret` (see "Setting DATABASE_URL on the server")

See [src/api/README.md#deployment](src/api/README.md#deployment) for details.

## License

See [LICENSE](LICENSE) file for details.
