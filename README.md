# Road Network Information Completion

Road type prediction for road network segments, combining graph machine learning with a production-ready API microservice.

## Repository Structure

```
road-network-information-completion/
├── src/
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
│   ├── api/                    # API unit + integration tests
│   └── ml/                     # ML unit tests
├── data/                       # Local datasets (git-ignored)
├── artifacts/                  # Trained models, results and logs (git-ignored)
├── k8s/                        # Kubernetes manifests
├── .github/workflows/          # CI/CD pipelines
├── requirements.txt            # API + test dependencies
└── requirements-ml.txt         # ML dependencies (torch, torch_geometric)
```

## Quick Start

### API Microservice
```bash
# Install dependencies
pip install -r requirements.txt

# Setup SSH tunnel to database (Required for local development)
ssh -L 5432:cs-astep02.srv.aau.dk:30432 username@student.aau.dk -N

# Configure environment
cp .env.example .env
# Edit .env with your DATABASE_URL (postgresql+asyncpg://user:pass@localhost:5432/db)

# Run locally
uvicorn src.api.main:app --reload
```

See [src/api/README.md](src/api/README.md) for detailed setup instructions.

### Machine Learning
```bash
pip install -r requirements-ml.txt

# Train and evaluate GraphSAGE (needs the map in data/bj_data/map/)
python -m src.ml.models.graphsage.run_graphsage
```

See [src/ml/README.md](src/ml/README.md).

## Testing

```bash
# API tests
pytest tests/api/ -v

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

See [src/api/README.md#deployment](src/api/README.md#deployment) for details.

## License

See [LICENSE](LICENSE) file for details.
