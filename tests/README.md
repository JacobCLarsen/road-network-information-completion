# Tests

```
tests/
├── api/            # API tests (in-memory SQLite, no Postgres needed)
│   ├── conftest.py
│   ├── unit/       # Service layer with mocked database
│   └── integration/# Full HTTP endpoint tests
└── ml/             # ML tests (need requirements-ml.txt)
    └── data/
```

# Run all tests
```bash
pytest
```

# Run only the API tests (what CI runs)
```bash
pytest tests/api/ -v
```

# Run with coverage report
```bash
pytest --cov
```

# Run specific test file
```bash
pytest tests/api/integration/test_health_routes.py
```

# Run specific test function
```bash
pytest tests/api/integration/test_health_routes.py::test_health_liveness
```

See [best_practices.md](best_practices.md) for how to write tests.
