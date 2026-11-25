# Tests for src/ml/device.py: which device the models are trained on.

import pytest
import torch

from src.ml import device as device_module
from src.ml.device import get_device, graph_to_device


# Without ML_DEVICE, a machine with a GPU trains on it
def test_uses_gpu_when_available(monkeypatch):
    monkeypatch.delenv("ML_DEVICE", raising=False)
    monkeypatch.setattr(device_module.torch.cuda, "is_available", lambda: True)
    assert get_device() == torch.device("cuda")


# Without ML_DEVICE, a machine without a GPU falls back to the CPU
def test_falls_back_to_cpu_without_gpu(monkeypatch):
    monkeypatch.delenv("ML_DEVICE", raising=False)
    monkeypatch.setattr(device_module.torch.cuda, "is_available", lambda: False)
    assert get_device() == torch.device("cpu")


# ML_DEVICE=cpu forces the CPU, even when there is a GPU
def test_ml_device_forces_cpu(monkeypatch):
    monkeypatch.setenv("ML_DEVICE", "cpu")
    monkeypatch.setattr(device_module.torch.cuda, "is_available", lambda: True)
    assert get_device() == torch.device("cpu")


# Asking for CUDA without a GPU is an error, not a silent fall back to the (much slower) CPU
def test_ml_device_cuda_without_gpu_raises(monkeypatch):
    monkeypatch.setenv("ML_DEVICE", "cuda")
    monkeypatch.setattr(device_module.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="cannot see a GPU"):
        get_device()


# graph_to_device moves the tensors and keeps everything else, without changing the original dictionary
def test_graph_to_device_moves_tensors_only():
    graph = {"x": torch.ones(2, 3), "name": "beijing"}
    moved = graph_to_device(graph, torch.device("cpu"))
    assert moved["x"].device == torch.device("cpu")
    assert moved["name"] == "beijing"
    assert moved is not graph
