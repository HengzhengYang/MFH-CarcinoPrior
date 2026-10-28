#!/usr/bin/env python3
"""Checkpointed full Task 02 training under protocol v4 + E001."""

from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import hashlib
import json
import math
import os
import shutil
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import joblib
import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score


SCRIPT_PATH = Path(__file__).resolve()
TASK_DIR = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]
sys.path.insert(0, str(TASK_DIR / "src"))
sys.path.insert(0, str(TASK_DIR / "scripts"))

from generate_and_validate_full_execution_plan import candidate_registry
from gpu_models import (
    LightGBMCudaBinary,
    TorchCudaLogisticHead,
    TorchCudaPlattCalibrator,
    XGBoostCudaBinary,
    build_fingerprint_matrices,
)
from graph_models import build_graph_records, predict_raw as predict_graph_raw, train_with_isolated_early_stop
from hybrid_models import SklearnCpuRandomForestBinary, SklearnCpuRbfSvcBinary, assert_process_rss
from lm_embeddings import extract_or_load_embeddings
from runtime_guards import cuda_evidence, require_cuda_runtime
from split_protocol import sha256_file


PROTOCOL_PATH = TASK_DIR / "configs" / "registered_protocol_v4_hybrid_restricted.json"
V3_PATH = TASK_DIR / "configs" / "registered_protocol_v3_gpu_only.json"
GO_FULL_PATH = TASK_DIR / "TASK00_GO_FULL_TRAINING_APPROVAL.json"
SUPPLEMENT_PATH = TASK_DIR / "TASK00_PROTOCOL_EXECUTION_SUPPLEMENT.json"
PLAN_PATH = TASK_DIR / "manifests" / "full_execution_plan_e001.csv"
PLAN_REPORT_PATH = TASK_DIR / "reports" / "full_execution_plan_validation_e001.json"
PROGRESS_PATH = TASK_DIR / "runtime" / "full_training_progress.json"
STATUS_PATH = TASK_DIR / "STATUS.json"
LOCK_PATH = TASK_DIR / "runtime" / "full_training_e001.lock"
PID_PATH = TASK_DIR / "runtime" / "full_training_e001.pid"
EVENTS_PATH = TASK_DIR / "logs" / "full_training_fit_events_e001.jsonl"
RESULT_ROOT = TASK_DIR / "runtime" / "job_results_e001"
CHECKPOINT_ROOT = TASK_DIR / "runtime" / "context_checkpoints_e001"
OUTPUT_ROOT = TASK_DIR / "predictions" / "full_e001"
MODEL_ROOT = TASK_DIR / "models" / "full_e001"
FEATURE_ROOT = TASK_DIR / "features" / "full_e001"
METRIC_ROOT = TASK_DIR / "metrics" / "full_e001"

EXPECTED_PROTOCOL_SHA256 = "2fc3cc0df70895643440dec40e27f51d045b88c62c7c50a8370af5688d592663"
EXPECTED_GO_FULL_SHA256 = "dae60ff147881db239e6b2f36dd0843f8f69687da8e4a9435fb9f9d4ff31dda4"
EXPECTED_SUPPLEMENT_SHA256 = "8ed6b3828c3acc53c2eca182833ad2b9e27fd6cfb099d358146f4ccba165c045"
EXPECTED_PLAN_SHA256 = "03ed011fed5b2b3270bb6106eeb76d1dc284ef8e60ccd412e8975aefe2f44ca5"
PLANNED_JOBS = 26498
HARD_CEILING = 27300
STORAGE_LIMIT_GIB = 60.0
RUNTIME_REVIEW_HOURS = 60.0


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def hash_ids(values) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()


def array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8).tobytes()).hexdigest()


def best_mcc_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], probability)))
    ranked = []
    for threshold in candidates:
        mcc = matthews_corrcoef(y, (probability >= threshold).astype(int))
        ranked.append((float(mcc), -abs(float(threshold) - 0.5), -float(threshold), float(threshold)))
    ranked.sort(reverse=True)
    return ranked[0][3]


def sigmoid_platt(raw: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    value = np.clip(np.asarray(raw, dtype=np.float64) * slope + intercept, -80.0, 80.0)
    probability = 1.0 / (1.0 + np.exp(-value))
    if not np.isfinite(probability).all() or np.any(probability < 0.0) or np.any(probability > 1.0):
        raise RuntimeError("Invalid calibrated probability")
    return probability


def directory_size_gib(path: Path) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except FileNotFoundError:
                pass
    return total / 1024**3


def normalize_plan_value(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value)
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text


class PlanIndex:
    fields = ("phase", "scenario_id", "design_id", "outer_fold", "inner_fold", "stage", "member", "candidate_id")

    def __init__(self, path: Path):
        if sha256_file(path) != EXPECTED_PLAN_SHA256:
            raise RuntimeError("E001 execution-plan SHA256 mismatch")
        with path.open("r", encoding="utf-8", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        if len(self.rows) != PLANNED_JOBS:
            raise RuntimeError("E001 execution-plan row-count mismatch")
        self.lookup = {}
        for row in self.rows:
            key = tuple(normalize_plan_value(row[field]) for field in self.fields)
            if key in self.lookup:
                raise RuntimeError(f"Duplicate execution-plan key: {key}")
            self.lookup[key] = row

    def get(self, *, phase, scenario_id, design_id, outer_fold="", inner_fold="", stage, member, candidate_id):
        key = tuple(normalize_plan_value(value) for value in (phase, scenario_id, design_id, outer_fold, inner_fold, stage, member, candidate_id))
        try:
            return self.lookup[key]
        except KeyError as exc:
            raise RuntimeError(f"Unregistered fit-job request: {key}") from exc


class JobLedger:
    def __init__(self, progress: dict):
        self.progress = progress
        RESULT_ROOT.mkdir(parents=True, exist_ok=True)
        EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.attempts_by_job = Counter()
        if EVENTS_PATH.exists():
            with EVENTS_PATH.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("event") == "FIT_ATTEMPT_STARTED":
                        self.attempts_by_job[event["job_id"]] += 1
        completed = sum(1 for _ in RESULT_ROOT.glob("*.json"))
        attempted = sum(self.attempts_by_job.values())
        self.progress["formal_fit_jobs_completed"] = completed
        self.progress["formal_fit_jobs_attempted"] = attempted
        self.write_progress()

    def write_progress(self) -> None:
        start = self.progress.get("started_at_utc")
        elapsed = 0.0
        if start:
            elapsed = max(0.0, (datetime.now(timezone.utc) - datetime.fromisoformat(start)).total_seconds() / 3600.0)
        completed = int(self.progress.get("formal_fit_jobs_completed", 0))
        empirical = None
        if completed > 0 and elapsed > 0:
            empirical = max(0.0, elapsed * (PLANNED_JOBS - completed) / completed)
        self.progress.update({
            "updated_at_utc": now_utc(),
            "elapsed_hours": elapsed,
            "progress_fraction": completed / PLANNED_JOBS,
            "eta_hours": empirical,
            "process_rss_gib": psutil.Process(os.getpid()).memory_info().rss / 1024**3,
        })
        atomic_json(PROGRESS_PATH, self.progress)

    def append_event(self, payload: dict) -> None:
        with EVENTS_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(json_ready(payload), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def result_path(self, row: dict) -> Path:
        return RESULT_ROOT / f"{row['job_id']}.json"

    def execute(self, row: dict, operation, oom_retry_batch_size: int | None = None) -> dict:
        result_path = self.result_path(row)
        if result_path.exists():
            return json.loads(result_path.read_text(encoding="utf-8"))
        batch_overrides = [None]
        if oom_retry_batch_size is not None:
            batch_overrides.append(int(oom_retry_batch_size))
        for attempt_position, batch_override in enumerate(batch_overrides):
            if int(self.progress["formal_fit_jobs_attempted"]) >= HARD_CEILING:
                self.hard_fail("FIT_JOB_HARD_CEILING_REACHED")
            if float(self.progress.get("elapsed_hours", 0.0)) >= RUNTIME_REVIEW_HOURS:
                self.progress["state"] = "PAUSED_RUNTIME_REVIEW_REQUIRED"
                self.progress["blocker"] = {"code": "RUNTIME_EXCEEDED_60_HOURS"}
                self.write_progress()
                raise RuntimeError("Runtime exceeded 60 hours; Task 00 review required before next fit")
            if int(self.progress["formal_fit_jobs_attempted"]) % 50 == 0:
                observed_storage = directory_size_gib(TASK_DIR)
                self.progress["task02_storage_gib"] = observed_storage
                if observed_storage >= STORAGE_LIMIT_GIB:
                    self.hard_fail("TASK02_STORAGE_LIMIT_REACHED")
            assert_process_rss(12.0)
            self.attempts_by_job[row["job_id"]] += 1
            attempt_index = self.attempts_by_job[row["job_id"]]
            self.progress["formal_fit_jobs_attempted"] = int(self.progress["formal_fit_jobs_attempted"]) + 1
            self.progress.update({
                "state": "RUNNING",
                "current_job_id": row["job_id"],
                "current_phase": row["phase"],
                "current_scenario": row["scenario_id"],
                "current_design": row["design_id"] or None,
                "current_outer_fold": int(row["outer_fold"]) if row["outer_fold"] else None,
                "current_member": row["member"],
                "current_stage": row["stage"],
                "current_candidate_id": row["candidate_id"],
            })
            self.write_progress()
            started = time.perf_counter()
            self.append_event({"event": "FIT_ATTEMPT_STARTED", "at_utc": now_utc(), "attempt_index": attempt_index, "batch_override": batch_override, **row})
            try:
                payload = operation(batch_override)
                result = {
                    "job_id": row["job_id"],
                    "attempt_index": attempt_index,
                    "completed_at_utc": now_utc(),
                    "duration_seconds": time.perf_counter() - started,
                    "plan": row,
                    "result": json_ready(payload),
                }
                atomic_json(result_path, result)
                self.progress["formal_fit_jobs_completed"] = int(self.progress["formal_fit_jobs_completed"]) + 1
                self.progress["last_completed_fit_at_utc"] = now_utc()
                self.append_event({"event": "FIT_ATTEMPT_COMPLETED", "at_utc": now_utc(), "attempt_index": attempt_index, "duration_seconds": result["duration_seconds"], "job_id": row["job_id"]})
                self.write_progress()
                return result
            except torch.cuda.OutOfMemoryError as exc:
                self.append_event({"event": "FIT_ATTEMPT_FAILED", "at_utc": now_utc(), "attempt_index": attempt_index, "job_id": row["job_id"], "failure": "CUDA_OOM", "error": str(exc)})
                gc.collect()
                torch.cuda.empty_cache()
                if attempt_position + 1 < len(batch_overrides):
                    continue
                self.hard_fail("PERSISTENT_GPU_OOM", exc)
            except Exception as exc:
                self.append_event({"event": "FIT_ATTEMPT_FAILED", "at_utc": now_utc(), "attempt_index": attempt_index, "job_id": row["job_id"], "failure": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
                self.hard_fail("FIT_JOB_FAILED", exc)
        raise AssertionError("unreachable")

    def hard_fail(self, code: str, exc: Exception | None = None):
        self.progress["state"] = "HARD_FAILED_RETURN_TO_TASK00"
        self.progress["blocker"] = {"code": code, "error": str(exc) if exc else None}
        self.write_progress()
        raise RuntimeError(code) from exc


def indices_for_ids(data: pd.DataFrame, identifiers) -> np.ndarray:
    requested = set(map(str, identifiers))
    indices = np.flatnonzero(data["model_unit_id"].astype(str).isin(requested).to_numpy())
    if len(indices) != len(requested):
        raise RuntimeError("Split identifiers do not align one-to-one with locked data")
    return indices.astype(np.int64)


def load_fold(data: pd.DataFrame, scenario_id: str, design_id: str, outer_fold: int):
    manifest_path = TASK_DIR / "splits" / scenario_id / design_id / "split_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    outer = manifest[(manifest.level == "outer") & (manifest.outer_fold == outer_fold)]
    inner = manifest[(manifest.level == "inner") & (manifest.outer_fold == outer_fold)]
    outer_train = indices_for_ids(data, outer.loc[outer.subset == "train", "model_unit_id"])
    outer_test = indices_for_ids(data, outer.loc[outer.subset == "test", "model_unit_id"])
    if set(outer_train) & set(outer_test):
        raise RuntimeError("Outer train/test leakage")
    inner_splits = []
    for inner_fold in range(5):
        frame = inner[inner.inner_fold == inner_fold]
        train = indices_for_ids(data, frame.loc[frame.subset == "train", "model_unit_id"])
        validation = indices_for_ids(data, frame.loc[frame.subset == "validation", "model_unit_id"])
        if set(train) & set(validation) or set(validation) & set(outer_test):
            raise RuntimeError("Inner/outer split leakage")
        inner_splits.append((inner_fold, train, validation))
    group_map = outer.drop_duplicates("model_unit_id").set_index("model_unit_id")["split_group_id"].astype(str).to_dict()
    split_groups = np.asarray([group_map[str(identifier)] for identifier in data["model_unit_id"]], dtype=object)
    return outer_train, outer_test, inner_splits, split_groups


def load_or_build_fingerprints(data: pd.DataFrame, scenario: dict) -> dict[str, np.ndarray]:
    root = FEATURE_ROOT / scenario["scenario_id"]
    array_path = root / "fingerprints.npz"
    metadata_path = root / "fingerprints.metadata.json"
    expected_ids_hash = hash_ids(data["model_unit_id"].astype(str))
    if array_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("input_sha256") != scenario["sha256"] or metadata.get("ordered_model_unit_id_sha256") != expected_ids_hash:
            raise RuntimeError("Fingerprint cache metadata mismatch")
        with np.load(array_path, allow_pickle=False) as payload:
            matrices = {name: np.ascontiguousarray(payload[name], dtype=np.float32) for name in ("ECFP4", "MACCS")}
        for name, matrix in matrices.items():
            if metadata["array_sha256"][name] != array_hash(matrix):
                raise RuntimeError("Fingerprint cache content mismatch")
        return matrices
    matrices = build_fingerprint_matrices(data["canonical_isomeric_smiles"].astype(str).tolist())
    root.mkdir(parents=True, exist_ok=True)
    with array_path.open("wb") as handle:
        np.savez_compressed(handle, ECFP4=matrices["ECFP4"], MACCS=matrices["MACCS"])
    atomic_json(metadata_path, {
        "input_sha256": scenario["sha256"],
        "ordered_model_unit_id_sha256": expected_ids_hash,
        "array_sha256": {name: array_hash(matrix) for name, matrix in matrices.items()},
        "shape": {name: list(matrix.shape) for name, matrix in matrices.items()},
        "model_fit_executed": False,
    })
    return matrices


def prepare_assets(data: pd.DataFrame, scenario: dict, protocol: dict, v3: dict):
    fingerprints = load_or_build_fingerprints(data, scenario)
    lm_config = v3["approved_candidates"]["smiles_language_models"]
    embeddings = {}
    for model_index, model_spec in enumerate(lm_config["models"]):
        matrix, _ = extract_or_load_embeddings(
            data=data,
            model_spec=model_spec,
            input_sha256=scenario["sha256"],
            protocol_sha256=EXPECTED_SUPPLEMENT_SHA256,
            cache_root=FEATURE_ROOT / scenario["scenario_id"] / "lm_embeddings",
            max_token_length=int(lm_config["max_token_length"]),
            seed=int(protocol["base_seed"] + 900000 + model_index),
        )
        embeddings[model_spec["name"]] = matrix
    graphs = build_graph_records(data)
    return {"fingerprints": fingerprints, "embeddings": embeddings, "graphs": graphs}


def fit_member(member: dict, candidate: dict, train_indices: np.ndarray, split_groups: np.ndarray, seed: int, y: np.ndarray, assets: dict, batch_override: int | None = None):
    params = dict(candidate["parameters"])
    family = member["family"]
    if family == "RandomForest":
        model = SklearnCpuRandomForestBinary(params, seed, PROTOCOL["cpu_resource_limits"]).fit(assets["fingerprints"][member["feature"]][train_indices], y[train_indices])
        return model, model.evidence()
    if family == "RBF_SVM":
        model = SklearnCpuRbfSvcBinary(params, seed, PROTOCOL["cpu_resource_limits"]).fit(assets["fingerprints"][member["feature"]][train_indices], y[train_indices])
        return model, model.evidence()
    if family == "XGBoost":
        model = XGBoostCudaBinary(params, seed).fit(assets["fingerprints"][member["feature"]][train_indices], y[train_indices])
        return model, asdict(model.evidence)
    if family == "LightGBM":
        model = LightGBMCudaBinary(params, seed).fit(assets["fingerprints"][member["feature"]][train_indices], y[train_indices])
        return model, asdict(model.evidence)
    if family == "language_model":
        model = TorchCudaLogisticHead(float(params["c_value"]), int(params["max_iter"]), seed).fit(assets["embeddings"][member["model_name"]][train_indices], y[train_indices])
        return model, model.evidence()
    if family == "graph":
        if batch_override is not None:
            params["batch_size"] = int(batch_override)
        model, evidence = train_with_isolated_early_stop(assets["graphs"], train_indices, split_groups[train_indices], member["architecture"], params, seed)
        return model, evidence
    raise RuntimeError(f"Unregistered member family: {family}")


def predict_member(member: dict, model, indices: np.ndarray, candidate: dict, assets: dict) -> np.ndarray:
    family = member["family"]
    if family in {"RandomForest", "RBF_SVM", "XGBoost", "LightGBM"}:
        return model.predict_raw(assets["fingerprints"][member["feature"]][indices])
    if family == "language_model":
        return model.predict_raw(assets["embeddings"][member["model_name"]][indices])
    batch_size = int(candidate["parameters"]["batch_size"])
    return predict_graph_raw(model, assets["graphs"], indices, batch_size)


def save_model(model, member: dict, candidate: dict, path_base: Path) -> str:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    family = member["family"]
    if family == "RandomForest":
        path = path_base.with_suffix(".joblib")
        joblib.dump({"model": model.model, "candidate": candidate}, path, compress=3)
    elif family == "RBF_SVM":
        path = path_base.with_suffix(".joblib")
        joblib.dump({
            "model": model.model,
            "standardizer_mean": model.standardizer.mean.detach().cpu().numpy(),
            "standardizer_scale": model.standardizer.scale.detach().cpu().numpy(),
            "candidate": candidate,
        }, path, compress=3)
    elif family == "XGBoost":
        path = path_base.with_suffix(".json")
        model.booster.save_model(path)
    elif family == "LightGBM":
        path = path_base.with_suffix(".txt")
        model.booster.save_model(path)
    elif family == "language_model":
        path = path_base.with_suffix(".pt")
        torch.save({
            "state_dict": model.module.state_dict(),
            "mean": model.mean.detach().cpu(),
            "scale": model.scale.detach().cpu(),
            "candidate": candidate,
        }, path)
    else:
        path = path_base.with_suffix(".pt")
        torch.save({"state_dict": model.state_dict(), "architecture": member["architecture"], "candidate": candidate}, path)
    return path.relative_to(REPO_ROOT).as_posix()


def release_model(model) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()


def candidate_by_id(member: dict, candidate_id: str) -> dict:
    matches = [item for item in member["candidates"] if item["id"] == candidate_id]
    if len(matches) != 1:
        raise RuntimeError(f"Candidate identity mismatch: {member['member']} {candidate_id}")
    return matches[0]


def job_seed(row: dict) -> int:
    return int(row["seed"])


def run_fit_prediction(ledger: JobLedger, row: dict, member: dict, candidate: dict, train: np.ndarray, validation: np.ndarray, split_groups: np.ndarray, y: np.ndarray, assets: dict, save_path: Path | None = None) -> dict:
    def operation(batch_override):
        model, evidence = fit_member(member, candidate, train, split_groups, job_seed(row), y, assets, batch_override)
        raw = predict_member(member, model, validation, candidate, assets)
        payload = {
            "training_units": len(train),
            "prediction_units": len(validation),
            "raw_scores": raw,
            "device_evidence": evidence,
            "training_index_sha256": hash_ids(train),
            "prediction_index_sha256": hash_ids(validation),
        }
        if save_path is not None:
            payload["model_path"] = save_model(model, member, candidate, save_path)
        release_model(model)
        return payload
    oom_batch = int(candidate["parameters"]["single_oom_reduced_batch_size"]) if member["family"] == "graph" else None
    return ledger.execute(row, operation, oom_retry_batch_size=oom_batch)["result"]


def select_candidate(candidate_results: list[tuple[dict, dict]]) -> tuple[dict, list[dict]]:
    grouped = defaultdict(list)
    candidates = {}
    for candidate, result in candidate_results:
        grouped[candidate["id"]].append(result)
        candidates[candidate["id"]] = candidate
    ranking = []
    for candidate_id, results in grouped.items():
        if len(results) != 5:
            raise RuntimeError(f"Candidate does not have five inner-fold results: {candidate_id}")
        aucs = np.asarray([item["roc_auc"] for item in results], dtype=np.float64)
        auprcs = np.asarray([item["auprc"] for item in results], dtype=np.float64)
        candidate = candidates[candidate_id]
        ranking.append({
            "candidate_id": candidate_id,
            "candidate_ordinal": candidate["ordinal"],
            "mean_inner_roc_auc": float(aucs.mean()),
            "mean_inner_auprc": float(auprcs.mean()),
            "population_std_inner_roc_auc": float(aucs.std(ddof=0)),
        })
    ranking.sort(key=lambda item: (-item["mean_inner_roc_auc"], -item["mean_inner_auprc"], item["population_std_inner_roc_auc"], item["candidate_ordinal"]))
    return candidates[ranking[0]["candidate_id"]], ranking


def run_context(scenario: dict, design_id: str, outer_fold: int, data: pd.DataFrame, y: np.ndarray, assets: dict, members: list[dict], plan: PlanIndex, ledger: JobLedger) -> None:
    scenario_id = scenario["scenario_id"]
    checkpoint = CHECKPOINT_ROOT / scenario_id / design_id / f"outer_{outer_fold}.json"
    if checkpoint.exists():
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        for output in payload["outputs"]:
            path = REPO_ROOT / output["path"]
            if not path.exists() or sha256_file(path) != output["sha256"]:
                raise RuntimeError("Context checkpoint output fingerprint mismatch")
        return
    outer_train, outer_test, inner_splits, split_groups = load_fold(data, scenario_id, design_id, outer_fold)
    member_train = {}
    member_test = {}
    selected_rows = []
    train_rows = []
    test_rows = []
    context_model_root = MODEL_ROOT / "nested" / scenario_id / design_id / f"outer_{outer_fold}"
    for member in members:
        candidate_results = []
        for candidate in member["candidates"]:
            for inner_fold, inner_train, inner_validation in inner_splits:
                row = plan.get(phase="nested_cv", scenario_id=scenario_id, design_id=design_id, outer_fold=outer_fold, inner_fold=inner_fold, stage="inner_candidate_fit", member=member["member"], candidate_id=candidate["id"])
                expected_params = json.loads(row["parameters_json"])
                if expected_params != candidate["parameters"]:
                    raise RuntimeError("Runtime candidate parameters differ from E001 plan")
                result = run_fit_prediction(ledger, row, member, candidate, inner_train, inner_validation, split_groups, y, assets)
                labels = y[inner_validation]
                raw = np.asarray(result["raw_scores"], dtype=np.float64)
                metrics = {"roc_auc": float(roc_auc_score(labels, raw)), "auprc": float(average_precision_score(labels, raw)), "inner_fold": inner_fold}
                candidate_results.append((candidate, metrics))
        selected, ranking = select_candidate(candidate_results)
        selected_rows.append({"scenario_id": scenario_id, "design_id": design_id, "outer_fold": outer_fold, "member": member["member"], "selected_candidate_id": selected["id"], "selected_candidate_ordinal": selected["ordinal"], "ranking": ranking})

        cross_raw = np.full(len(data), np.nan, dtype=np.float64)
        for inner_fold, inner_train, inner_validation in inner_splits:
            row = plan.get(phase="nested_cv", scenario_id=scenario_id, design_id=design_id, outer_fold=outer_fold, inner_fold=inner_fold, stage="selected_member_crossfit", member=member["member"], candidate_id="__OUTER_SELECTED__")
            result = run_fit_prediction(ledger, row, member, selected, inner_train, inner_validation, split_groups, y, assets)
            cross_raw[inner_validation] = np.asarray(result["raw_scores"], dtype=np.float64)
        if not np.isfinite(cross_raw[outer_train]).all() or np.isfinite(cross_raw[outer_test]).any():
            raise RuntimeError("Selected-member crossfit coverage or isolation failure")

        calibration_row = plan.get(phase="nested_cv", scenario_id=scenario_id, design_id=design_id, outer_fold=outer_fold, stage="gpu_platt_calibration", member=member["member"], candidate_id="__OUTER_SELECTED__")
        calibration_model_path = context_model_root / f"{member['member']}_platt.pt"
        def calibrate(_):
            calibrator = TorchCudaPlattCalibrator(int(PROTOCOL["calibration"]["max_iter"]), job_seed(calibration_row)).fit(cross_raw[outer_train], y[outer_train])
            probability = calibrator.predict_proba(cross_raw[outer_train])
            calibration_model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": calibrator.module.state_dict(), "evidence": calibrator.evidence(), "candidate_id": selected["id"]}, calibration_model_path)
            evidence = calibrator.evidence()
            del calibrator
            torch.cuda.empty_cache()
            return {"train_probabilities": probability, "slope": evidence["slope"], "intercept": evidence["intercept"], "device_evidence": evidence, "model_path": calibration_model_path.relative_to(REPO_ROOT).as_posix()}
        calibration = ledger.execute(calibration_row, calibrate)["result"]
        train_probability = np.asarray(calibration["train_probabilities"], dtype=np.float64)

        refit_row = plan.get(phase="nested_cv", scenario_id=scenario_id, design_id=design_id, outer_fold=outer_fold, stage="selected_member_outer_refit", member=member["member"], candidate_id="__OUTER_SELECTED__")
        refit = run_fit_prediction(ledger, refit_row, member, selected, outer_train, outer_test, split_groups, y, assets, context_model_root / member["member"])
        outer_raw = np.asarray(refit["raw_scores"], dtype=np.float64)
        outer_probability = sigmoid_platt(outer_raw, float(calibration["slope"]), float(calibration["intercept"]))
        member_train[member["member"]] = train_probability
        member_test[member["member"]] = outer_probability
        for local, index in enumerate(outer_train):
            train_rows.append({"scenario_id": scenario_id, "design_id": design_id, "outer_fold": outer_fold, "member": member["member"], "branch": member["branch"], "model_unit_id": data.iloc[index]["model_unit_id"], "cv_group_id": data.iloc[index]["cv_group_id"], "final_label": int(y[index]), "raw_crossfit_score": float(cross_raw[index]), "calibrated_probability": float(train_probability[local]), "selected_candidate_id": selected["id"]})
        for local, index in enumerate(outer_test):
            test_rows.append({"scenario_id": scenario_id, "design_id": design_id, "outer_fold": outer_fold, "member": member["member"], "branch": member["branch"], "model_unit_id": data.iloc[index]["model_unit_id"], "cv_group_id": data.iloc[index]["cv_group_id"], "final_label": int(y[index]), "raw_score": float(outer_raw[local]), "calibrated_probability": float(outer_probability[local]), "selected_candidate_id": selected["id"]})

    branches = {member["member"]: member["branch"] for member in members}
    train_branch = {branch: np.mean([member_train[name] for name, value in branches.items() if value == branch], axis=0) for branch in ("fingerprint", "graph", "language_model")}
    test_branch = {branch: np.mean([member_test[name] for name, value in branches.items() if value == branch], axis=0) for branch in ("fingerprint", "graph", "language_model")}
    ensemble_train = np.mean(list(train_branch.values()), axis=0)
    ensemble_test = np.mean(list(test_branch.values()), axis=0)
    threshold = best_mcc_threshold(y[outer_train], ensemble_train)
    for local, index in enumerate(outer_train):
        train_rows.append({"scenario_id": scenario_id, "design_id": design_id, "outer_fold": outer_fold, "member": "FINAL_ENSEMBLE", "branch": "ensemble", "model_unit_id": data.iloc[index]["model_unit_id"], "cv_group_id": data.iloc[index]["cv_group_id"], "final_label": int(y[index]), "raw_crossfit_score": np.nan, "calibrated_probability": float(ensemble_train[local]), "selected_candidate_id": "FIXED_14_MEMBER_BRANCH_BALANCED"})
    for local, index in enumerate(outer_test):
        test_rows.append({"scenario_id": scenario_id, "design_id": design_id, "outer_fold": outer_fold, "member": "FINAL_ENSEMBLE", "branch": "ensemble", "model_unit_id": data.iloc[index]["model_unit_id"], "cv_group_id": data.iloc[index]["cv_group_id"], "final_label": int(y[index]), "raw_score": np.nan, "calibrated_probability": float(ensemble_test[local]), "selected_candidate_id": "FIXED_14_MEMBER_BRANCH_BALANCED", "threshold": threshold, "predicted_label": int(ensemble_test[local] >= threshold)})
    output_dir = OUTPUT_ROOT / "nested" / scenario_id / design_id
    train_path = output_dir / f"outer_{outer_fold}_train_crossfit.csv"
    test_path = output_dir / f"outer_{outer_fold}_test_predictions.csv"
    selection_path = METRIC_ROOT / "candidate_selection" / scenario_id / design_id / f"outer_{outer_fold}.json"
    atomic_csv(train_path, pd.DataFrame(train_rows))
    atomic_csv(test_path, pd.DataFrame(test_rows))
    atomic_json(selection_path, {"scenario_id": scenario_id, "design_id": design_id, "outer_fold": outer_fold, "threshold": threshold, "members": selected_rows})
    outputs = [{"path": path.relative_to(REPO_ROOT).as_posix(), "sha256": sha256_file(path)} for path in (train_path, test_path, selection_path)]
    atomic_json(checkpoint, {"status": "PASS", "completed_at_utc": now_utc(), "scenario_id": scenario_id, "design_id": design_id, "outer_fold": outer_fold, "outputs": outputs})


def global_primary_selection(members: list[dict], plan: PlanIndex) -> dict[str, dict]:
    selections = {}
    scenario_id = "primary_conflict_excluded"
    for member in members:
        ranking = []
        for candidate in member["candidates"]:
            context_auc, context_auprc = [], []
            for design_id in PROTOCOL["split_policy"]["designs"]:
                for outer_fold in range(5):
                    aucs, auprcs = [], []
                    for inner_fold in range(5):
                        row = plan.get(phase="nested_cv", scenario_id=scenario_id, design_id=design_id, outer_fold=outer_fold, inner_fold=inner_fold, stage="inner_candidate_fit", member=member["member"], candidate_id=candidate["id"])
                        result_path = RESULT_ROOT / f"{row['job_id']}.json"
                        if not result_path.exists():
                            raise RuntimeError("Primary search result missing before final selection")
                        result = json.loads(result_path.read_text(encoding="utf-8"))["result"]
                        raw = np.asarray(result["raw_scores"], dtype=np.float64)
                        data_spec = next(item for item in PROTOCOL["inputs"] if item["scenario_id"] == scenario_id)
                        data = pd.read_csv(REPO_ROOT / data_spec["path"])
                        _, _, inner_splits, _ = load_fold(data, scenario_id, design_id, outer_fold)
                        validation = inner_splits[inner_fold][2]
                        labels = data["final_label"].to_numpy(dtype=int)[validation]
                        aucs.append(float(roc_auc_score(labels, raw)))
                        auprcs.append(float(average_precision_score(labels, raw)))
                    context_auc.append(float(np.mean(aucs)))
                    context_auprc.append(float(np.mean(auprcs)))
            ranking.append({"candidate_id": candidate["id"], "candidate_ordinal": candidate["ordinal"], "mean_roc_auc": float(np.mean(context_auc)), "mean_auprc": float(np.mean(context_auprc)), "roc_auc_context_std": float(np.std(context_auc, ddof=0))})
        ranking.sort(key=lambda item: (-item["mean_roc_auc"], -item["mean_auprc"], item["roc_auc_context_std"], item["candidate_ordinal"]))
        selections[member["member"]] = {"selected": candidate_by_id(member, ranking[0]["candidate_id"]), "ranking": ranking}
    return selections


def run_final_primary(data: pd.DataFrame, y: np.ndarray, assets: dict, members: list[dict], plan: PlanIndex, ledger: JobLedger) -> None:
    checkpoint = CHECKPOINT_ROOT / "final_primary.json"
    if checkpoint.exists():
        return
    selections = global_primary_selection(members, plan)
    scenario_id = "primary_conflict_excluded"
    design_id = "scaffold_group_nested_cv"
    member_probability = {}
    final_rows = []
    model_manifest = []
    for member in members:
        candidate = selections[member["member"]]["selected"]
        cross_raw = np.full(len(data), np.nan, dtype=np.float64)
        split_groups = None
        for fold in range(5):
            train, test, _, groups = load_fold(data, scenario_id, design_id, fold)
            split_groups = groups
            row = plan.get(phase="final_primary", scenario_id=scenario_id, design_id=design_id, outer_fold=fold, inner_fold=fold, stage="final_primary_crossfit", member=member["member"], candidate_id="__GLOBAL_SELECTED__")
            result = run_fit_prediction(ledger, row, member, candidate, train, test, groups, y, assets)
            cross_raw[test] = np.asarray(result["raw_scores"], dtype=np.float64)
        if not np.isfinite(cross_raw).all():
            raise RuntimeError("Final-primary crossfit coverage failure")
        calibration_row = plan.get(phase="final_primary", scenario_id=scenario_id, design_id=design_id, stage="final_primary_gpu_platt_calibration", member=member["member"], candidate_id="__GLOBAL_SELECTED__")
        calibration_path = MODEL_ROOT / "final_primary" / f"{member['member']}_platt.pt"
        def calibrate(_):
            calibrator = TorchCudaPlattCalibrator(int(PROTOCOL["calibration"]["max_iter"]), job_seed(calibration_row)).fit(cross_raw, y)
            probability = calibrator.predict_proba(cross_raw)
            calibration_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": calibrator.module.state_dict(), "evidence": calibrator.evidence(), "candidate_id": candidate["id"]}, calibration_path)
            evidence = calibrator.evidence()
            return {"probabilities": probability, "slope": evidence["slope"], "intercept": evidence["intercept"], "device_evidence": evidence, "model_path": calibration_path.relative_to(REPO_ROOT).as_posix()}
        calibration = ledger.execute(calibration_row, calibrate)["result"]
        probability = np.asarray(calibration["probabilities"], dtype=np.float64)
        member_probability[member["member"]] = probability
        for index in range(len(data)):
            final_rows.append({"scenario_id": scenario_id, "design_id": design_id, "member": member["member"], "branch": member["branch"], "model_unit_id": data.iloc[index]["model_unit_id"], "cv_group_id": data.iloc[index]["cv_group_id"], "final_label": int(y[index]), "raw_crossfit_score": float(cross_raw[index]), "calibrated_probability": float(probability[index]), "selected_candidate_id": candidate["id"]})
        refit_row = plan.get(phase="final_primary", scenario_id=scenario_id, design_id=design_id, stage="final_primary_full_refit", member=member["member"], candidate_id="__GLOBAL_SELECTED__")
        model_path_base = MODEL_ROOT / "final_primary" / member["member"]
        def full_refit(batch_override):
            model, evidence = fit_member(member, candidate, np.arange(len(data), dtype=np.int64), split_groups, job_seed(refit_row), y, assets, batch_override)
            model_path = save_model(model, member, candidate, model_path_base)
            release_model(model)
            return {"training_units": len(data), "device_evidence": evidence, "model_path": model_path, "candidate_id": candidate["id"]}
        oom_batch = int(candidate["parameters"]["single_oom_reduced_batch_size"]) if member["family"] == "graph" else None
        refit = ledger.execute(refit_row, full_refit, oom_retry_batch_size=oom_batch)["result"]
        model_manifest.append({"member": member["member"], "candidate_id": candidate["id"], "base_model_path": refit["model_path"], "base_model_sha256": sha256_file(REPO_ROOT / refit["model_path"]), "calibrator_path": calibration["model_path"], "calibrator_sha256": sha256_file(REPO_ROOT / calibration["model_path"])})

    branches = {member["member"]: member["branch"] for member in members}
    branch_probability = {branch: np.mean([member_probability[name] for name, value in branches.items() if value == branch], axis=0) for branch in ("fingerprint", "graph", "language_model")}
    ensemble = np.mean(list(branch_probability.values()), axis=0)
    threshold = best_mcc_threshold(y, ensemble)
    for index in range(len(data)):
        final_rows.append({"scenario_id": scenario_id, "design_id": design_id, "member": "FINAL_ENSEMBLE", "branch": "ensemble", "model_unit_id": data.iloc[index]["model_unit_id"], "cv_group_id": data.iloc[index]["cv_group_id"], "final_label": int(y[index]), "raw_crossfit_score": np.nan, "calibrated_probability": float(ensemble[index]), "selected_candidate_id": "FIXED_14_MEMBER_BRANCH_BALANCED", "threshold": threshold, "predicted_label": int(ensemble[index] >= threshold)})
    prediction_path = OUTPUT_ROOT / "final_primary_crossfit_predictions.csv"
    selection_path = METRIC_ROOT / "final_primary_global_selection.json"
    manifest_path = MODEL_ROOT / "final_primary" / "model_manifest.json"
    atomic_csv(prediction_path, pd.DataFrame(final_rows))
    atomic_json(selection_path, {"selection_source": "primary_inner_metrics_only", "sensitivity_metrics_used": False, "outer_oof_performance_used": False, "members": selections, "threshold": threshold})
    atomic_json(manifest_path, {"status": "DRAFT_PENDING_TASK00_MODEL_LOCK", "members": model_manifest, "threshold": threshold, "ensemble": "equal within branch then equal across three branches"})
    atomic_json(checkpoint, {"status": "PASS", "completed_at_utc": now_utc(), "outputs": [{"path": path.relative_to(REPO_ROOT).as_posix(), "sha256": sha256_file(path)} for path in (prediction_path, selection_path, manifest_path)]})


def verify_prelaunch() -> tuple[dict, dict, dict, list[dict], PlanIndex]:
    for path, expected in ((PROTOCOL_PATH, EXPECTED_PROTOCOL_SHA256), (GO_FULL_PATH, EXPECTED_GO_FULL_SHA256), (SUPPLEMENT_PATH, EXPECTED_SUPPLEMENT_SHA256), (PLAN_PATH, EXPECTED_PLAN_SHA256)):
        if sha256_file(path) != expected:
            raise RuntimeError(f"Prelaunch SHA256 mismatch: {path}")
    report = json.loads(PLAN_REPORT_PATH.read_text(encoding="utf-8"))
    if report.get("status") != "PASS" or report.get("model_fit_executed") is not False or report.get("whole_run_planned_jobs") != PLANNED_JOBS or report.get("plan_manifest_sha256") != EXPECTED_PLAN_SHA256:
        raise RuntimeError("No-fit execution-plan validation is not a matching PASS")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    approval = json.loads(GO_FULL_PATH.read_text(encoding="utf-8"))
    supplement = json.loads(SUPPLEMENT_PATH.read_text(encoding="utf-8"))
    if approval.get("full_training_authorized") is not True or supplement["authorization"]["formal_training_launch_authorized_after_no_fit_pass"] is not True:
        raise RuntimeError("Formal launch authorization is absent")
    members = candidate_registry(protocol, supplement)
    plan = PlanIndex(PLAN_PATH)
    backend_counts = Counter(row["backend_policy"] for row in plan.rows)
    if set(backend_counts) != {"cpu_p002_whitelist", "cuda_required"}:
        raise RuntimeError("Execution plan contains an unregistered backend policy")
    if len(members) != 14 or len(plan.rows) != PLANNED_JOBS:
        raise RuntimeError("Prelaunch member or plan count mismatch")
    require_cuda_runtime()
    return protocol, approval, supplement, members, plan


def update_status_running() -> None:
    status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    status.update({"phase": "FULL_TRAINING_E001_RUNNING", "updated_at_utc": now_utc(), "formal_full_training_started": True})
    status["execution_readiness"]["status"] = "NO_FIT_PLAN_PASS_FORMAL_TRAINING_RUNNING"
    status["execution_readiness"]["formal_fit_jobs_attempted"] = int(json.loads(PROGRESS_PATH.read_text(encoding="utf-8")).get("formal_fit_jobs_attempted", 0))
    atomic_json(STATUS_PATH, status)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    global PROTOCOL
    PROTOCOL, _, _, members, plan = verify_prelaunch()
    if args.verify_only:
        print(json.dumps({"status": "PASS", "mode": "NO_FIT_RUNNER_VERIFICATION", "plan_sha256": EXPECTED_PLAN_SHA256, "planned_jobs": PLANNED_JOBS, "member_count": len(members), "formal_fit_jobs_attempted": 0, "cuda": cuda_evidence()}, indent=2, sort_keys=True))
        return 0

    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = LOCK_PATH.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("Another E001 full-training process holds the launch lock") from exc
    PID_PATH.write_text(f"{os.getpid()}\n", encoding="utf-8")
    progress = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    if progress.get("no_fit_validation_status") != "PASS" or progress.get("execution_supplement_sha256") != EXPECTED_SUPPLEMENT_SHA256 or progress.get("execution_plan_sha256") != EXPECTED_PLAN_SHA256:
        raise RuntimeError("Progress ledger does not record the matching E001 no-fit PASS")
    progress.update({"state": "RUNNING", "process_id": os.getpid(), "training_started": True, "started_at_utc": progress.get("started_at_utc") or now_utc(), "blocker": None})
    atomic_json(PROGRESS_PATH, progress)
    update_status_running()
    ledger = JobLedger(progress)
    v3 = json.loads(V3_PATH.read_text(encoding="utf-8"))
    try:
        for scenario in PROTOCOL["inputs"]:
            input_path = REPO_ROOT / scenario["path"]
            if sha256_file(input_path) != scenario["sha256"]:
                ledger.hard_fail("LOCKED_INPUT_SHA256_CHANGED")
            data = pd.read_csv(input_path, encoding="utf-8-sig")
            y = data["final_label"].to_numpy(dtype=np.int64)
            assets = prepare_assets(data, scenario, PROTOCOL, v3)
            for design_id in PROTOCOL["split_policy"]["designs"]:
                for outer_fold in range(5):
                    run_context(scenario, design_id, outer_fold, data, y, assets, members, plan, ledger)
            if scenario["role"] == "sensitivity":
                status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
                status["sensitivity_model_fitting_started"] = True
                atomic_json(STATUS_PATH, status)
            del assets
            gc.collect()
            torch.cuda.empty_cache()

        primary = next(item for item in PROTOCOL["inputs"] if item["scenario_id"] == "primary_conflict_excluded")
        data = pd.read_csv(REPO_ROOT / primary["path"], encoding="utf-8-sig")
        y = data["final_label"].to_numpy(dtype=np.int64)
        assets = prepare_assets(data, primary, PROTOCOL, v3)
        status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        status["final_primary_full_data_fit_started"] = True
        atomic_json(STATUS_PATH, status)
        run_final_primary(data, y, assets, members, plan, ledger)
        if int(ledger.progress["formal_fit_jobs_completed"]) != PLANNED_JOBS:
            ledger.hard_fail("PLANNED_JOB_COMPLETION_COUNT_MISMATCH")
        ledger.progress.update({"state": "COMPLETED_PENDING_TASK02_VALIDATION", "process_id": None, "current_job_id": None, "current_stage": None, "current_member": None, "eta_hours": 0.0})
        ledger.write_progress()
        status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        status.update({"phase": "FULL_TRAINING_COMPLETED_PENDING_VALIDATION", "updated_at_utc": now_utc()})
        atomic_json(STATUS_PATH, status)
        return 0
    finally:
        if ledger.progress.get("state") != "RUNNING":
            try:
                PID_PATH.unlink()
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
