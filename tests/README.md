# Tests

```
tests/
├── db/             # Data layer tests
│   ├── unit/       # No database needed
│   └── integration/# Real Postgres, skipped unless TEST_DATABASE_URL is set
├── api/            # API tests
│   ├── conftest.py
│   └── integration/# HTTP endpoint tests (map/run routes need TEST_DATABASE_URL, health does not)
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

# Run the data layer tests against your local database
# (TEST_DATABASE_URL from .env; every test is rolled back, so nothing is changed)
```bash
pytest tests/db/ -v
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
