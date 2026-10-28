"""GPU-only classical learners, logistic heads, and sigmoid calibration."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass

import cupy as cp
import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator

from runtime_guards import (
    assert_cuda_module,
    assert_cuda_tensor,
    require_cuda_runtime,
    set_reproducible_seed,
)


def build_fingerprint_matrices(smiles: list[str]) -> dict[str, np.ndarray]:
    morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    ecfp_rows = []
    maccs_rows = []
    for value in smiles:
        mol = Chem.MolFromSmiles(str(value))
        if mol is None:
            raise ValueError(f"Invalid locked SMILES: {value!r}")
        ecfp_rows.append(morgan.GetFingerprintAsNumPy(mol).astype(np.float32, copy=False))
        fp = MACCSkeys.GenMACCSKeys(mol)
        array = np.zeros((fp.GetNumBits(),), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, array)
        maccs_rows.append(array.astype(np.float32, copy=False))
    return {
        "ECFP4": np.ascontiguousarray(np.vstack(ecfp_rows), dtype=np.float32),
        "MACCS": np.ascontiguousarray(np.vstack(maccs_rows), dtype=np.float32),
    }


@dataclass
class FitEvidence:
    backend: str
    fit_device: str
    prediction_device: str
    warnings: list[str]
    seed: int


class XGBoostCudaBinary:
    def __init__(self, parameters: dict, seed: int):
        self.parameters = dict(parameters)
        self.seed = int(seed)
        self.booster: xgb.Booster | None = None
        self.evidence: FitEvidence | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "XGBoostCudaBinary":
        require_cuda_runtime()
        cp.cuda.Device(0).use()
        x_gpu = cp.asarray(x, dtype=cp.float32)
        y_gpu = cp.asarray(y, dtype=cp.float32)
        rounds = int(self.parameters["num_boost_round"])
        params = {key: value for key, value in self.parameters.items() if key != "num_boost_round"}
        params.update(
            {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "seed": self.seed,
                "nthread": 1,
                "verbosity": 0,
            }
        )
        if params.get("device") != "cuda":
            raise RuntimeError("XGBoost device must be cuda")
        matrix = xgb.QuantileDMatrix(x_gpu, label=y_gpu)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.booster = xgb.train(params, matrix, num_boost_round=rounds)
        config = json.loads(self.booster.save_config())
        reported = config["learner"]["generic_param"]["device"]
        if reported != "cuda:0":
            raise RuntimeError(f"XGBoost reported unexpected device {reported}")
        warning_messages = [str(item.message) for item in caught]
        if any("device" in message.lower() and "mismatch" in message.lower() for message in warning_messages):
            raise RuntimeError("XGBoost device mismatch warning during fit")
        self.evidence = FitEvidence("xgboost", reported, "not_yet_predicted", warning_messages, self.seed)
        return self

    def predict_raw(self, x: np.ndarray) -> np.ndarray:
        if self.booster is None or self.evidence is None:
            raise RuntimeError("XGBoost model is not fitted")
        x_gpu = cp.asarray(x, dtype=cp.float32)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            prediction = self.booster.inplace_predict(x_gpu, predict_type="margin")
        cp.cuda.Stream.null.synchronize()
        warning_messages = [str(item.message) for item in caught]
        if warning_messages:
            raise RuntimeError(f"XGBoost prediction warnings are forbidden: {warning_messages}")
        if type(prediction).__module__.split(".")[0] != "cupy":
            raise RuntimeError(f"XGBoost prediction did not remain on CuPy/CUDA: {type(prediction)}")
        if not bool(cp.isfinite(prediction).all()):
            raise RuntimeError("XGBoost produced a non-finite raw score")
        self.evidence.prediction_device = "cuda:0"
        return cp.asnumpy(prediction).astype(np.float64, copy=False)


class LightGBMCudaBinary:
    def __init__(self, parameters: dict, seed: int):
        self.parameters = dict(parameters)
        self.seed = int(seed)
        self.booster: lgb.Booster | None = None
        self.evidence: FitEvidence | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LightGBMCudaBinary":
        require_cuda_runtime()
        rounds = int(self.parameters["num_boost_round"])
        params = {key: value for key, value in self.parameters.items() if key != "num_boost_round"}
        params.update(
            {
                "objective": "binary",
                "metric": "binary_logloss",
                "verbosity": -1,
                "seed": self.seed,
                "data_random_seed": self.seed,
                "feature_fraction_seed": self.seed,
                "bagging_seed": self.seed,
                "drop_seed": self.seed,
                "num_threads": 1,
                "gpu_device_id": 0,
                "gpu_use_dp": True,
            }
        )
        if params.get("device_type") != "cuda":
            raise RuntimeError("LightGBM device_type must be cuda")
        dataset = lgb.Dataset(np.asarray(x, dtype=np.float32), label=np.asarray(y, dtype=np.float32))
        self.booster = lgb.train(params, dataset, num_boost_round=rounds)
        reported = str(self.booster.params.get("device_type"))
        if reported != "cuda":
            raise RuntimeError(f"LightGBM reported unexpected device_type {reported}")
        self.evidence = FitEvidence("lightgbm", "cuda:0", "lightgbm_cuda_predictor", [], self.seed)
        return self

    def predict_raw(self, x: np.ndarray) -> np.ndarray:
        if self.booster is None or self.evidence is None:
            raise RuntimeError("LightGBM model is not fitted")
        prediction = np.asarray(
            self.booster.predict(np.asarray(x, dtype=np.float32), raw_score=True),
            dtype=np.float64,
        )
        if not np.isfinite(prediction).all():
            raise RuntimeError("LightGBM produced a non-finite raw score")
        return prediction


class TorchCudaLogisticHead:
    def __init__(self, c_value: float, max_iter: int, seed: int):
        if c_value <= 0:
            raise ValueError("C must be positive")
        self.c_value = float(c_value)
        self.max_iter = int(max_iter)
        self.seed = int(seed)
        self.mean: torch.Tensor | None = None
        self.scale: torch.Tensor | None = None
        self.module: nn.Linear | None = None
        self.final_loss: float | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TorchCudaLogisticHead":
        device = require_cuda_runtime()
        set_reproducible_seed(self.seed)
        x_tensor = torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)
        y_tensor = torch.as_tensor(np.asarray(y, dtype=np.float32), device=device)
        assert_cuda_tensor(x_tensor, "logistic_head.x")
        self.mean = x_tensor.mean(dim=0, keepdim=True)
        self.scale = x_tensor.std(dim=0, unbiased=False, keepdim=True).clamp_min(1.0e-6)
        standardized = (x_tensor - self.mean) / self.scale
        self.module = nn.Linear(standardized.shape[1], 1, device=device)
        nn.init.zeros_(self.module.weight)
        prevalence = float(np.mean(y))
        initial_bias = np.log(np.clip(prevalence, 1e-6, 1 - 1e-6) / np.clip(1 - prevalence, 1e-6, 1))
        nn.init.constant_(self.module.bias, float(initial_bias))
        assert_cuda_module(self.module, "logistic_head")

        positives = y_tensor.sum().clamp_min(1.0)
        negatives = (len(y_tensor) - y_tensor.sum()).clamp_min(1.0)
        sample_weight = torch.where(
            y_tensor > 0.5,
            len(y_tensor) / (2.0 * positives),
            len(y_tensor) / (2.0 * negatives),
        )
        optimizer = torch.optim.LBFGS(
            self.module.parameters(),
            lr=1.0,
            max_iter=self.max_iter,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad(set_to_none=True)
            logits = self.module(standardized).squeeze(1)
            data_loss = F.binary_cross_entropy_with_logits(logits, y_tensor, weight=sample_weight)
            penalty = 0.5 * self.module.weight.square().sum() / (self.c_value * len(y_tensor))
            loss = data_loss + penalty
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            logits = self.module(standardized).squeeze(1)
            data_loss = F.binary_cross_entropy_with_logits(logits, y_tensor, weight=sample_weight)
            penalty = 0.5 * self.module.weight.square().sum() / (self.c_value * len(y_tensor))
            value = (data_loss + penalty).detach()
        self.final_loss = float(value.cpu())
        assert_cuda_module(self.module, "logistic_head")
        return self

    def predict_raw(self, x: np.ndarray) -> np.ndarray:
        if self.module is None or self.mean is None or self.scale is None:
            raise RuntimeError("CUDA logistic head is not fitted")
        device = require_cuda_runtime()
        x_tensor = torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)
        with torch.no_grad():
            logits = self.module((x_tensor - self.mean) / self.scale).squeeze(1)
        assert_cuda_tensor(logits, "logistic_head.logits")
        if not bool(torch.isfinite(logits).all().item()):
            raise RuntimeError("CUDA logistic head produced non-finite logits")
        return logits.detach().cpu().numpy().astype(np.float64, copy=False)

    def evidence(self) -> dict:
        if self.module is None:
            raise RuntimeError("CUDA logistic head is not fitted")
        assert_cuda_module(self.module, "logistic_head")
        return {
            "backend": "native_pytorch_cuda_logistic",
            "device": str(next(self.module.parameters()).device),
            "c_value": self.c_value,
            "max_iter": self.max_iter,
            "seed": self.seed,
            "final_loss": self.final_loss,
        }


class TorchCudaPlattCalibrator:
    def __init__(self, max_iter: int, seed: int):
        self.max_iter = int(max_iter)
        self.seed = int(seed)
        self.module: nn.Linear | None = None
        self.final_loss: float | None = None

    def fit(self, raw_scores: np.ndarray, y: np.ndarray) -> "TorchCudaPlattCalibrator":
        device = require_cuda_runtime()
        set_reproducible_seed(self.seed)
        score_tensor = torch.as_tensor(
            np.asarray(raw_scores, dtype=np.float32).reshape(-1, 1),
            device=device,
        )
        y_tensor = torch.as_tensor(np.asarray(y, dtype=np.float32), device=device)
        self.module = nn.Linear(1, 1, device=device)
        nn.init.ones_(self.module.weight)
        nn.init.zeros_(self.module.bias)
        optimizer = torch.optim.LBFGS(
            self.module.parameters(),
            lr=1.0,
            max_iter=self.max_iter,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad(set_to_none=True)
            calibrated_logits = self.module(score_tensor).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(calibrated_logits, y_tensor)
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            calibrated_logits = self.module(score_tensor).squeeze(1)
            self.final_loss = float(
                F.binary_cross_entropy_with_logits(calibrated_logits, y_tensor).detach().cpu()
            )
        assert_cuda_module(self.module, "platt_calibrator")
        return self

    def predict_proba(self, raw_scores: np.ndarray) -> np.ndarray:
        if self.module is None:
            raise RuntimeError("CUDA Platt calibrator is not fitted")
        device = require_cuda_runtime()
        score_tensor = torch.as_tensor(
            np.asarray(raw_scores, dtype=np.float32).reshape(-1, 1),
            device=device,
        )
        with torch.no_grad():
            probabilities = torch.sigmoid(self.module(score_tensor).squeeze(1))
        assert_cuda_tensor(probabilities, "platt_calibrator.probabilities")
        if not bool(torch.isfinite(probabilities).all().item()):
            raise RuntimeError("CUDA Platt calibrator produced non-finite probabilities")
        output = probabilities.detach().cpu().numpy().astype(np.float64, copy=False)
        if np.any(output < 0.0) or np.any(output > 1.0):
            raise RuntimeError("CUDA Platt calibrator produced an out-of-range probability")
        return output

    def evidence(self) -> dict:
        if self.module is None:
            raise RuntimeError("CUDA Platt calibrator is not fitted")
        assert_cuda_module(self.module, "platt_calibrator")
        return {
            "backend": "native_pytorch_cuda_two_parameter_platt",
            "device": str(next(self.module.parameters()).device),
            "max_iter": self.max_iter,
            "seed": self.seed,
            "final_loss": self.final_loss,
            "slope": float(self.module.weight.detach().cpu().reshape(-1)[0]),
            "intercept": float(self.module.bias.detach().cpu().reshape(-1)[0]),
        }
