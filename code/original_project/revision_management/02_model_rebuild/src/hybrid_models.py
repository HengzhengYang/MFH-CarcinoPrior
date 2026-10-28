"""P002-authorized sklearn CPU learners with fail-closed whitelist auditing."""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import numpy as np
import psutil
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from threadpoolctl import threadpool_info, threadpool_limits

from runtime_guards import assert_cuda_tensor, require_cuda_runtime, set_reproducible_seed


RF_FQN = "sklearn.ensemble.RandomForestClassifier"
SVC_FQN = "sklearn.svm.SVC"
_FIT_LOCK = threading.Lock()
_ACTIVE_CPU_FITS = 0


def estimator_fqn(estimator) -> str:
    if type(estimator) is RandomForestClassifier:
        return RF_FQN
    if type(estimator) is SVC:
        return SVC_FQN
    return f"{estimator.__class__.__module__}.{estimator.__class__.__name__}"


def assert_cpu_estimator_whitelisted(estimator) -> str:
    fqn = estimator_fqn(estimator)
    if type(estimator) is RandomForestClassifier:
        return fqn
    if type(estimator) is SVC:
        params = estimator.get_params(deep=False)
        if params.get("kernel") != "rbf" or params.get("probability") is not False:
            raise RuntimeError("P002 permits sklearn SVC only with kernel=rbf and probability=False")
        return fqn
    raise RuntimeError(f"P002 CPU estimator whitelist violation: {fqn}")


def assert_process_rss(maximum_process_rss_gib: float) -> float:
    rss_gib = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
    if rss_gib > float(maximum_process_rss_gib):
        raise MemoryError(
            f"P002 process RSS {rss_gib:.3f} GiB exceeds registered limit "
            f"{maximum_process_rss_gib:.3f} GiB"
        )
    return float(rss_gib)


@contextmanager
def exclusive_whitelisted_cpu_fit(estimator, maximum_process_rss_gib: float):
    global _ACTIVE_CPU_FITS
    fqn = assert_cpu_estimator_whitelisted(estimator)
    if not _FIT_LOCK.acquire(blocking=False):
        raise RuntimeError("P002 permits only one concurrent CPU estimator fit")
    try:
        _ACTIVE_CPU_FITS += 1
        if _ACTIVE_CPU_FITS != 1:
            raise RuntimeError("P002 concurrent CPU estimator fit limit exceeded")
        before = assert_process_rss(maximum_process_rss_gib)
        yield fqn, before
        assert_process_rss(maximum_process_rss_gib)
    finally:
        _ACTIVE_CPU_FITS -= 1
        _FIT_LOCK.release()


@dataclass
class CpuFitEvidence:
    estimator_fqn: str
    estimator_parameters: dict
    fit_device: str
    prediction_device: str
    whitelist_status: str
    process_id: int
    configured_estimator_threads: int
    native_threadpool_limit: int
    maximum_process_rss_gib: float
    peak_observed_process_rss_gib: float
    fit_duration_seconds: float
    seed: int
    threadpool_snapshot: list[dict]
    gpu_preprocessing: dict | None = None


class TorchCudaFoldStandardizer:
    """Fold-local standardization whose fitted statistics live on CUDA."""

    def __init__(self, seed: int):
        self.seed = int(seed)
        self.mean: torch.Tensor | None = None
        self.scale: torch.Tensor | None = None

    def fit(self, x: np.ndarray) -> "TorchCudaFoldStandardizer":
        device = require_cuda_runtime()
        set_reproducible_seed(self.seed)
        tensor = torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)
        assert_cuda_tensor(tensor, "p002_svc_standardizer.fit_input")
        self.mean = tensor.mean(dim=0, keepdim=True)
        self.scale = tensor.std(dim=0, unbiased=False, keepdim=True).clamp_min(1.0e-6)
        assert_cuda_tensor(self.mean, "p002_svc_standardizer.mean")
        assert_cuda_tensor(self.scale, "p002_svc_standardizer.scale")
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise RuntimeError("CUDA fold standardizer is not fitted")
        device = require_cuda_runtime()
        tensor = torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)
        with torch.no_grad():
            transformed = (tensor - self.mean) / self.scale
        assert_cuda_tensor(transformed, "p002_svc_standardizer.output")
        if not bool(torch.isfinite(transformed).all().item()):
            raise RuntimeError("CUDA fold standardizer produced non-finite values")
        return np.ascontiguousarray(transformed.detach().cpu().numpy(), dtype=np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def evidence(self) -> dict:
        if self.mean is None or self.scale is None:
            raise RuntimeError("CUDA fold standardizer is not fitted")
        return {
            "backend": "native_pytorch_cuda_fold_standardizer",
            "device": str(self.mean.device),
            "seed": self.seed,
            "feature_count": int(self.mean.shape[1]),
            "minimum_scale": float(self.scale.min().detach().cpu()),
        }


class SklearnCpuRandomForestBinary:
    def __init__(self, parameters: dict, seed: int, resource_limits: dict):
        self.parameters = dict(parameters)
        self.seed = int(seed)
        self.resource_limits = dict(resource_limits)
        self.model: RandomForestClassifier | None = None
        self.fit_evidence: CpuFitEvidence | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SklearnCpuRandomForestBinary":
        if not np.isfinite(x).all():
            raise RuntimeError("P002 RandomForest input contains non-finite values; imputation is forbidden")
        params = dict(self.parameters)
        params["random_state"] = self.seed
        if int(params.get("n_jobs", -1)) != int(self.resource_limits["random_forest_n_jobs"]):
            raise RuntimeError("RandomForest n_jobs differs from the registered P002 limit")
        model = RandomForestClassifier(**params)
        start = time.perf_counter()
        with exclusive_whitelisted_cpu_fit(
            model, float(self.resource_limits["maximum_process_rss_gib"])
        ) as (fqn, before_rss):
            with threadpool_limits(limits=int(self.resource_limits["native_threadpool_limit"])):
                model.fit(np.ascontiguousarray(x, dtype=np.float32), np.asarray(y, dtype=np.int64))
        duration = time.perf_counter() - start
        peak_rss = max(before_rss, assert_process_rss(self.resource_limits["maximum_process_rss_gib"]))
        self.model = model
        self.fit_evidence = CpuFitEvidence(
            estimator_fqn=fqn,
            estimator_parameters=model.get_params(deep=False),
            fit_device="cpu_p002_whitelist",
            prediction_device="cpu_p002_whitelist",
            whitelist_status="PASS",
            process_id=os.getpid(),
            configured_estimator_threads=int(params["n_jobs"]),
            native_threadpool_limit=int(self.resource_limits["native_threadpool_limit"]),
            maximum_process_rss_gib=float(self.resource_limits["maximum_process_rss_gib"]),
            peak_observed_process_rss_gib=peak_rss,
            fit_duration_seconds=float(duration),
            seed=self.seed,
            threadpool_snapshot=threadpool_info(),
        )
        return self

    def predict_raw(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("P002 RandomForest is not fitted")
        assert_cpu_estimator_whitelisted(self.model)
        probability = np.asarray(self.model.predict_proba(np.asarray(x, dtype=np.float32)), dtype=np.float64)
        if probability.ndim != 2 or probability.shape[1] != 2:
            raise RuntimeError("RandomForest predict_proba has unexpected shape")
        raw = probability[:, 1]
        if not np.isfinite(raw).all() or np.any(raw < 0.0) or np.any(raw > 1.0):
            raise RuntimeError("RandomForest produced an invalid raw probability")
        return raw

    def evidence(self) -> dict:
        if self.fit_evidence is None:
            raise RuntimeError("P002 RandomForest is not fitted")
        return asdict(self.fit_evidence)


class SklearnCpuRbfSvcBinary:
    def __init__(self, parameters: dict, seed: int, resource_limits: dict):
        self.parameters = dict(parameters)
        self.seed = int(seed)
        self.resource_limits = dict(resource_limits)
        self.standardizer = TorchCudaFoldStandardizer(seed + 1)
        self.model: SVC | None = None
        self.fit_evidence: CpuFitEvidence | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SklearnCpuRbfSvcBinary":
        if not np.isfinite(x).all():
            raise RuntimeError("P002 SVC input contains non-finite values; imputation is forbidden")
        scaled = self.standardizer.fit_transform(x)
        params = dict(self.parameters)
        params["random_state"] = self.seed
        model = SVC(**params)
        assert_cpu_estimator_whitelisted(model)
        start = time.perf_counter()
        with exclusive_whitelisted_cpu_fit(
            model, float(self.resource_limits["maximum_process_rss_gib"])
        ) as (fqn, before_rss):
            with threadpool_limits(limits=int(self.resource_limits["native_threadpool_limit"])):
                model.fit(scaled, np.asarray(y, dtype=np.int64))
        duration = time.perf_counter() - start
        if int(getattr(model, "fit_status_", 1)) != 0:
            raise RuntimeError("P002 RBF-SVC did not converge within the registered fit")
        if model.get_params(deep=False)["probability"] is not False:
            raise RuntimeError("P002 forbids SVC internal probability fitting")
        peak_rss = max(before_rss, assert_process_rss(self.resource_limits["maximum_process_rss_gib"]))
        self.model = model
        self.fit_evidence = CpuFitEvidence(
            estimator_fqn=fqn,
            estimator_parameters=model.get_params(deep=False),
            fit_device="cpu_p002_whitelist",
            prediction_device="cpu_p002_whitelist",
            whitelist_status="PASS",
            process_id=os.getpid(),
            configured_estimator_threads=1,
            native_threadpool_limit=int(self.resource_limits["native_threadpool_limit"]),
            maximum_process_rss_gib=float(self.resource_limits["maximum_process_rss_gib"]),
            peak_observed_process_rss_gib=peak_rss,
            fit_duration_seconds=float(duration),
            seed=self.seed,
            threadpool_snapshot=threadpool_info(),
            gpu_preprocessing=self.standardizer.evidence(),
        )
        return self

    def predict_raw(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("P002 RBF-SVC is not fitted")
        assert_cpu_estimator_whitelisted(self.model)
        scaled = self.standardizer.transform(x)
        raw = np.asarray(self.model.decision_function(scaled), dtype=np.float64).reshape(-1)
        if not np.isfinite(raw).all():
            raise RuntimeError("RBF-SVC produced a non-finite decision score")
        return raw

    def evidence(self) -> dict:
        if self.fit_evidence is None:
            raise RuntimeError("P002 RBF-SVC is not fitted")
        return asdict(self.fit_evidence)
