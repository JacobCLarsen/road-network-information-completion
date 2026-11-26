# src/ml/device.py

# Picks where the models are trained: on the GPU (CUDA) when one is available, otherwise on the CPU.
# The training functions call get_device() and move the model and the data there, so the same code runs on a laptop
# without a GPU and on a machine with one.
#
# Set the environment variable ML_DEVICE to force a device, e.g. ML_DEVICE=cpu to compare against the GPU, or
# ML_DEVICE=cuda:1 to use the second GPU. The tests set ML_DEVICE=cpu, so they give the same results everywhere.

import os

import torch


def get_device():
    """
    Returns the torch.device to train on: the value of ML_DEVICE if it is set, else "cuda" if PyTorch can see a GPU,
    else "cpu".

    Raises a RuntimeError if ML_DEVICE asks for CUDA but no GPU is available, instead of silently using the CPU.
    """
    requested = os.environ.get("ML_DEVICE", "").strip()
    if requested:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"ML_DEVICE={requested}, but PyTorch cannot see a GPU. Install a CUDA build of PyTorch "
                "(see src/ml/README.md) or unset ML_DEVICE to use the CPU."
            )
        return device

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_device(device):
    """Returns a short readable name for the device, e.g. "cuda (NVIDIA GeForce RTX 4060)" or "cpu"."""
    if device.type == "cuda":
        return f"{device} ({torch.cuda.get_device_name(device)})"
    return str(device)


def graph_to_device(graph, device):
    """Returns a copy of the graph dictionary with every tensor moved to the device. Other values are kept as they are."""
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in graph.items()}
