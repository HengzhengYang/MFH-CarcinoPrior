"""Fail-closed GPU helpers for Task 02."""

from __future__ import annotations

import gc
import random
import sys
from dataclasses import dataclass
from typing import Callable, TypeVar

import numpy as np
import torch


EXPECTED_ENV_PREFIX = "/home/administrator/miniforge3/envs/mfh-carcinoprior-gpu"
EXPECTED_TORCH = "2.11.0+cu128"
T = TypeVar("T")


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def require_cuda_runtime() -> torch.device:
    if sys.prefix != EXPECTED_ENV_PREFIX:
        raise RuntimeError(f"Task 02 requires {EXPECTED_ENV_PREFIX}; got {sys.prefix}")
    if torch.__version__ != EXPECTED_TORCH:
        raise RuntimeError(f"Task 02 requires torch {EXPECTED_TORCH}; got {torch.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable and CPU fallback is forbidden")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one CUDA device; found {torch.cuda.device_count()}")
    device = torch.device("cuda:0")
    major, minor = torch.cuda.get_device_capability(device)
    arch = f"sm_{major}{minor}"
    if arch not in torch.cuda.get_arch_list():
        raise RuntimeError(f"PyTorch wheel lacks required architecture {arch}")
    return device


def assert_cuda_tensor(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_cuda:
        device = getattr(value, "device", None)
        raise RuntimeError(f"{name} must be a CUDA tensor; got {device}")
    if value.device.index not in (None, 0):
        raise RuntimeError(f"{name} must be on cuda:0; got {value.device}")


def assert_cuda_module(module: torch.nn.Module, name: str) -> None:
    parameters = list(module.parameters())
    if not parameters:
        raise RuntimeError(f"{name} has no parameters to assert")
    for index, parameter in enumerate(parameters):
        assert_cuda_tensor(parameter, f"{name}.parameter[{index}]")


def cuda_evidence() -> dict:
    device = require_cuda_runtime()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    capability = torch.cuda.get_device_capability(device)
    return {
        "environment_prefix": sys.prefix,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "required_arch": f"sm_{capability[0]}{capability[1]}",
        "free_vram_mib": round(free_bytes / 1024**2, 1),
        "total_vram_mib": round(total_bytes / 1024**2, 1),
        "cpu_fallback_allowed": False,
    }


@dataclass(frozen=True)
class OOMRetryEvidence:
    attempted_batch_sizes: tuple[int, ...]
    reductions_used: int
    cpu_fallback_used: bool = False


def run_with_single_oom_retry(
    operation: Callable[[int], T],
    initial_batch_size: int,
    reduced_batch_size: int,
) -> tuple[T, OOMRetryEvidence]:
    if initial_batch_size <= reduced_batch_size or reduced_batch_size <= 0:
        raise ValueError("A single OOM retry must use one positive, smaller batch size")
    attempts: list[int] = []
    for batch_size in (initial_batch_size, reduced_batch_size):
        attempts.append(batch_size)
        try:
            result = operation(batch_size)
            return result, OOMRetryEvidence(tuple(attempts), len(attempts) - 1)
        except torch.cuda.OutOfMemoryError:
            if len(attempts) == 2:
                raise RuntimeError(
                    "Persistent GPU OOM after the one registered batch-size reduction; "
                    "CPU fallback is forbidden"
                )
            gc.collect()
            torch.cuda.empty_cache()
    raise AssertionError("unreachable")
