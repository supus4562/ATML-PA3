"""
Utility functions for reproducibility, timing, GPU memory monitoring,
parameter metrics, and logging.
"""

import os
import random
import time
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 6304) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for strict reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class CUDATimer:
    """Accurate timing context manager with CUDA synchronization."""
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.start_time


def get_peak_vram_mb() -> float:
    """Return peak allocated GPU memory in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    return 0.0


def reset_peak_vram() -> None:
    """Reset peak GPU memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Return (total_parameters, trainable_parameters),
    counting tied parameters only once.
    """
    visited_params: Set[int] = set()
    total_params = 0
    trainable_params = 0

    for param in model.parameters():
        param_id = id(param)
        if param_id in visited_params:
            continue
        visited_params.add(param_id)

        numel = param.numel()
        total_params += numel
        if param.requires_grad:
            trainable_params += numel

    return total_params, trainable_params


def snapshot_model_parameters(model: nn.Module, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """
    Save a snapshot of unique model parameters (cloned on CPU or specified device)
    to serve as reference theta_0 for L2-SP or displacement metrics.
    """
    snapshot: Dict[str, torch.Tensor] = {}
    visited_ids: Set[int] = set()

    for name, param in model.named_parameters():
        param_id = id(param)
        if param_id in visited_ids:
            continue
        visited_ids.add(param_id)
        snapshot[name] = param.detach().to(device=device, copy=True)

    return snapshot


def compute_parameter_displacement(
    model: nn.Module,
    initial_params: Dict[str, torch.Tensor],
) -> float:
    """
    Compute full-model L2 displacement ||theta - theta_0||_2 = sqrt(sum_j (theta_j - theta_{0,j})^2).
    Counts tied parameters once.
    """
    squared_diff_sum = 0.0
    visited_ids: Set[int] = set()

    for name, param in model.named_parameters():
        param_id = id(param)
        if param_id in visited_ids or name not in initial_params:
            continue
        visited_ids.add(param_id)

        ref_param = initial_params[name].to(device=param.device, dtype=param.dtype)
        diff = (param.detach() - ref_param).float()
        squared_diff_sum += torch.sum(diff ** 2).item()

    return float(np.sqrt(squared_diff_sum))


def get_checkpoint_size_mb(path: str) -> float:
    """Return file or directory size in MB."""
    if not os.path.exists(path):
        return 0.0
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024.0 * 1024.0)
    total_bytes = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if not os.path.islink(fp):
                total_bytes += os.path.getsize(fp)
    return total_bytes / (1024.0 * 1024.0)
