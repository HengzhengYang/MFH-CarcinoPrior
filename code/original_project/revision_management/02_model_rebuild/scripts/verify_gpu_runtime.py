#!/usr/bin/env python3
"""Fail-closed CUDA runtime gate for Task 02.

This script does not read model data or fit a model. It verifies that the
interpreter used to launch a Task 02 run is the registered CUDA interpreter and
that a real CUDA autograd operation succeeds. CPU fallback is intentionally not
implemented.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone


EXPECTED_ENV_PREFIX = "/home/administrator/miniforge3/envs/mfh-carcinoprior-gpu"
EXPECTED_TORCH = "2.11.0+cu128"
EXPECTED_DEVICE_SUBSTRING = "NVIDIA GeForce RTX 5070 Ti"


def fail(message: str) -> None:
    raise RuntimeError(f"TASK02 GPU-ONLY GATE FAILED: {message}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum-free-vram-mib", type=int, default=1024)
    args = parser.parse_args()

    try:
        import torch

        if sys.prefix != EXPECTED_ENV_PREFIX:
            fail(f"unexpected environment prefix {sys.prefix!r}")
        if torch.__version__ != EXPECTED_TORCH:
            fail(f"torch {torch.__version__!r} != {EXPECTED_TORCH!r}")
        if not torch.cuda.is_available():
            fail("torch.cuda.is_available() is false; CPU fallback is forbidden")
        if torch.cuda.device_count() != 1:
            fail(f"expected exactly one CUDA device, found {torch.cuda.device_count()}")

        device = torch.device("cuda:0")
        device_name = torch.cuda.get_device_name(device)
        capability = torch.cuda.get_device_capability(device)
        required_arch = f"sm_{capability[0]}{capability[1]}"
        arch_list = torch.cuda.get_arch_list()
        if EXPECTED_DEVICE_SUBSTRING not in device_name:
            fail(f"unexpected CUDA device {device_name!r}")
        if required_arch not in arch_list:
            fail(f"wheel lacks native {required_arch} support: {arch_list}")

        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        free_mib = free_bytes / 1024**2
        if free_mib < args.minimum_free_vram_mib:
            fail(
                f"free VRAM {free_mib:.1f} MiB is below the requested "
                f"{args.minimum_free_vram_mib} MiB"
            )

        torch.manual_seed(20260725)
        x = torch.randn((512, 512), device=device, requires_grad=True)
        loss = (x @ x.T).mean()
        loss.backward()
        torch.cuda.synchronize(device)
        if x.grad is None or not bool(torch.isfinite(x.grad).all().item()):
            fail("CUDA autograd returned a missing or non-finite gradient")

        result = {
            "artifact": "TASK02_GPU_RUNTIME_GATE",
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "PASS",
            "cpu_fallback_allowed": False,
            "environment_prefix": sys.prefix,
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "device": "cuda:0",
            "device_name": device_name,
            "compute_capability": f"{capability[0]}.{capability[1]}",
            "required_arch": required_arch,
            "free_vram_mib": round(free_mib, 1),
            "total_vram_mib": round(total_bytes / 1024**2, 1),
            "autograd_smoke": "PASS",
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
