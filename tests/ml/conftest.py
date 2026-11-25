# Shared setup for the ML tests.

import pytest


# Run every ML test on the CPU, also on a machine with a GPU. GPU results can differ slightly from CPU results, and
# the tests compare tensors made on the CPU with tensors from training, so they must all be on the same device.
# autouse=True applies this to every test in tests/ml without listing it as a parameter.
@pytest.fixture(autouse=True)
def train_on_cpu(monkeypatch):
    monkeypatch.setenv("ML_DEVICE", "cpu")
