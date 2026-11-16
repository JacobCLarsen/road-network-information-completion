# Road-Network-Information-Completion API Microservice

A FastAPI service that reads the road network database (segments, labels, training runs, metrics and predictions) so a front end can show it. Built for deployment on Kubernetes with automated CI/CD pipelines.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Local Development](#local-development)
- [API Documentation](#api-documentation)
- [Database Schema](#database-schema)
- [Testing](#testing)
- [Deployment](#deployment)
- [Environment Variables](#environment-variables)

## Overview

This service provides a read-only REST API for a web page that shows:
- A map of the road segments, where the segments without a road type (the missing values) stand out
- The results of a model run: the predicted road types on the map, the scores and the training curve

The ML scripts write to the database themselves through the repositories in [src/db](../db/README.md).
The API uses the same repositories to read the data.

## Features

- **Async FastAPI** - High-performance async API with SQLAlchemy 2.0
- **PostgreSQL** - Production database with connection pooling
- **Testing** - Endpoint tests against a real Postgres database (rolled back after each test)
- **CORS** - Any front end can call the API from the browser (GET only)
- **Health Checks** - Kubernetes liveness/readiness probes with DB validation
- **Logging** - Structured logging for all operations
- **CI/CD** - Automated testing, building, and deployment via GitHub Actions
- **Kubernetes Deployment** - Production-ready manifests with resource limits

## Architecture

```
src/api/
├── main.py                 # FastAPI app with middleware, CORS and exception handlers
├── schemas.py              # Pydantic response models
└── routes/                 # API endpoint definitions
    ├── map.py              # GET /map/... (GeoJSON for the web page)
    ├── runs.py             # GET /runs/...
    └── health.py
```

The routes get a session with `Depends(get_session)` and use the repositories from `src/db`.
There is no separate service layer, because the repositories already do the database work.

## Local Development

### Prerequisites

- Python 3.13+
- PostgreSQL 14+
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/JacobCLarsen/road-network-information-completion.git
   cd road-network-information-completion
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv

   # Windows
   .\venv\Scripts\Activate

   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the database**

   The database (PostGIS) and its migrations live in the [astep-aau/data](https://github.com/astep-aau/data) repo.
   Clone it next to this repo and start it with Docker:
   ```bash
   cd ../data
   cp .env.example .env
   docker compose up -d --build        # database + migrations + dataset import
   ```

5. **Configure environment**

   ```bash
   cp .env.example .env
   ```

   The defaults in `.env.example` point at the local Docker database:
   ```
   DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/data
   ```

   To use the server database from your machine instead, forward it to a local port and point
   `DATABASE_URL` at that port (needs cluster access):
   ```bash
   kubectl port-forward svc/dataset-postgres -n main 5433:5432
   ```

6. **Run the application**
   ```bash
   uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
   ```

7. **Access the API**
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs
   - OpenAPI schema: http://localhost:8000/openapi.json

## API Documentation

### Map

- `GET /map/road-types` - The road type classes (`class_id`, `name`, `category`)
- `GET /map/missing?run_id=` - All segments without a label as GeoJSON, with `predicted` if `run_id` is given
- `GET /map/segments?min_lon=&min_lat=&max_lon=&max_lat=&run_id=` - All segments in the box as GeoJSON
  (at most 20000), with `label` (null if missing) and `predicted`

### Runs

- `GET /runs/` - All training runs, newest first
- `GET /runs/{run_id}` - One run with its `epochs` (training curve) and `metrics` (scores)

Example of calling it from a front end:

```js
const missing = await fetch("http://localhost:8000/map/missing?run_id=1").then(r => r.json());
const runs = await fetch("http://localhost:8000/runs/").then(r => r.json());
```

### Health Checks

- `GET /health/` - Basic health check
- `GET /health/live` - Kubernetes liveness probe
- `GET /health/ready` - Kubernetes readiness probe (validates DB connection)

For detailed API documentation, run the server and visit `/docs`.

## Database Schema

The database schema is owned by the migrations in the astep-aau/data repo; the API never creates
tables. The models and repositories for it are in [src/db](../db/README.md).

## Testing

### Running Tests

```bash
# All API tests
pytest tests/api/ -v

# With coverage
pytest tests/api/ --cov=src/api --cov-report=html
```

### Test Structure

- `test_health_routes.py` - works without Postgres
- `test_map_routes.py`, `test_run_routes.py` - need `TEST_DATABASE_URL` (skipped without it).
  They use the rolled back session from `tests/db/integration`, so they do not change your database.

See [tests/README.md](../../tests/README.md) for more details.

## Deployment

### CI/CD Pipeline

The project uses GitHub Actions for automated deployment:

#### Pull Request Workflow
`.github/workflows/pr-build-and-test.yml`
- Runs tests on every PR to `main`
- Builds Docker image
- Blocks merge if tests fail

#### Deployment Workflow
`.github/workflows/deploy.yml`
- Triggers on push to `main`
- Builds and pushes Docker image to `ghcr.io`
- Deploys to Kubernetes (cs-25-sw-5-06 namespace)
- Tags images with `:latest` and `:{git-sha}`

### Kubernetes Deployment

**Resources:**
- CPU: 300m request / 600m limit
- Memory: 512Mi request / 1Gi limit
- Replicas: 1 (Recreate strategy for zero-downtime with quota constraints)

**Manifests:**
- `k8s/deployment.yaml` - Main deployment configuration
- `k8s/service.yaml` - ClusterIP service (port 80 → 8000)
- `k8s/secret.yaml` - Database credentials

**Access:**
- Within cluster: `http://road-network-information-completion-service`
- Other namespaces: `http://road-network-information-completion-service.cs-25-sw-5-06`

### Manual Deployment

```bash
# Build Docker image
docker build -t ghcr.io/jacobclarsen/road-network-information-completion:latest .

# Push to registry
docker push ghcr.io/jacobclarsen/road-network-information-completion:latest

# Deploy to Kubernetes (requires kubeconfig)
kubectl apply -f k8s/deployment.yaml -n cs-25-sw-5-06
kubectl apply -f k8s/service.yaml -n cs-25-sw-5-06
```

## Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DATABASE_URL` | PostgreSQL connection string (`postgresql+asyncpg://...`) | - | Yes |
| `DEBUG` | Enable SQL query logging | `false` | No |
| `DB_CHECK_MIGRATIONS` | Check on startup that the database has run the expected migration | `true` | No |
| `ENV` | Environment (local/cluster), sets the log level | `local` | No |

Settings are read from the environment first, then from `.env` (see `src/db/config.py`).

### Example Configurations

**Local Development:**
```bash
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/data
DEBUG=true
```

**Cluster (Kubernetes):** `DATABASE_URL` comes from the `db-secret` secret (see `k8s/secret.example.yaml`)
and points at `dataset-postgres.main.svc.cluster.local:5432`, the database service in namespace `main`.
```yaml
env:
  - name: ENV
    value: "cluster"
  - name: DEBUG
    value: "false"
  - name: DATABASE_URL
    valueFrom:
      secretKeyRef:
        name: db-secret
        key: DATABASE_URL
```
