# Road-Network-Information-Completion API Microservice

A FastAPI-based microservice for managing machine learning model metrics and model artifacts for road type prediction. Built for deployment on Kubernetes with automated CI/CD pipelines.

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

This service provides a RESTful API for:
- Managing ML model types and metadata
- Storing and retrieving model performance metrics
- Downloading trained model files

## Features

- **Async FastAPI** - High-performance async API with SQLAlchemy 2.0
- **PostgreSQL** - Production database with connection pooling
- **Testing** - Unit + integration tests against in-memory SQLite
- **Health Checks** - Kubernetes liveness/readiness probes with DB validation
- **Logging** - Structured logging for all operations
- **CI/CD** - Automated testing, building, and deployment via GitHub Actions
- **Kubernetes Deployment** - Production-ready manifests with resource limits

## Architecture

```
src/api/
├── main.py                 # FastAPI app with middleware and exception handlers
├── database.py             # Database engine and session management
├── routes/                 # API endpoint definitions
│   ├── model_types.py
│   ├── model_metrics.py
│   ├── model_download.py
│   └── health.py
├── services/               # Business logic layer
│   ├── model_type_service.py
│   ├── model_metric_service.py
│   └── model_download_service.py
├── tables/                 # SQLAlchemy ORM tables
├── schemas/                # Pydantic request/response models
└── exceptions.py           # Custom exception definitions
```

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

### Model Types

- `POST /model-types/create` - Create a new model type
- `GET /model-types/` - List all model types

### Model Metrics

- `POST /model-metrics/create` - Create model metrics
- `GET /model-metrics/{model_type}` - Get metrics for a model type

### Downloads

- `GET /download_model/{model_id}` - Download model file (.pth)

### Health Checks

- `GET /health/` - Basic health check
- `GET /health/live` - Kubernetes liveness probe
- `GET /health/ready` - Kubernetes readiness probe (validates DB connection)

For detailed API documentation, run the server and visit `/docs`.

## Database Schema

The database schema is owned by the migrations in the astep-aau/data repo; the API never creates
tables. The shared data layer for it is in [src/db](../db/README.md).

> The tables below (`models` schema) are the old API tables. They do not exist in the new database
> and will be replaced by the `attribute_prediction` tables (`model_run`, `epoch`, `metric`, ...).

### Tables (old)

**model_types**
- `id` (UUID, PK)
- `name` (String, unique)

**model_metrics**
- `id` (UUID, PK)
- `model_type` (UUID, FK → model_types)
- `train_time_min` (Integer)
- `bias` (Float)
- `gap` (Float)
- `path_to_save` (String)
- `created_at` (Timestamp)

**hyperparams**
- `id` (UUID, PK)
- `model_id` (UUID, FK → model_metrics)
- `param_name` (String)
- `param_value` (String)

**loss**
- `id` (UUID, PK)
- `model_id` (UUID, FK → model_metrics)
- `type` (String)
- `loss_value` (Float)
- `loss_unit` (String)

## Testing

### Running Tests

```bash
# All tests
pytest tests/api/ -v

# With coverage
pytest tests/api/ --cov=src/api --cov-report=html

# Integration tests only
pytest tests/api/integration/ -v

# Unit tests only
pytest tests/api/unit/ -v
```

### Test Structure

- **Unit Tests** (`tests/api/unit/`) - Service layer logic with mocked database
- **Integration Tests** (`tests/api/integration/`) - Full API endpoint testing with SQLite

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
