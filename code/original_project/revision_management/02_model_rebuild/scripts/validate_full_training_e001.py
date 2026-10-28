#!/usr/bin/env python3
"""Post-training integrity validation and Task 00 review-package assembly for E001.

This entry point never trains or refits a predictive model.  It reads the
immutable protocol, split manifests, job ledger and completed artifacts,
recomputes registered selection/evaluation quantities, and writes Task 02
validation deliverables.  Metric-only calibration slope/intercept estimation
is executed on CUDA and is never saved or used for selection, calibration,
thresholding, or prediction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
import psutil
import rdkit
import sklearn
import torch
import xgboost
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)


SCRIPT_PATH = Path(__file__).resolve()
TASK_DIR = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]
sys.path.insert(0, str(TASK_DIR / "src"))
sys.path.insert(0, str(TASK_DIR / "scripts"))

from generate_and_validate_full_execution_plan import candidate_registry
from split_protocol import load_locked_scenario, sha256_file, validate_manifest


PROTOCOL_PATH = TASK_DIR / "configs" / "registered_protocol_v4_hybrid_restricted.json"
GO_FULL_PATH = TASK_DIR / "TASK00_GO_FULL_TRAINING_APPROVAL.json"
SUPPLEMENT_PATH = TASK_DIR / "TASK00_PROTOCOL_EXECUTION_SUPPLEMENT.json"
PLAN_PATH = TASK_DIR / "manifests" / "full_execution_plan_e001.csv"
SPLIT_INDEX_PATH = TASK_DIR / "splits" / "split_manifest_index.csv"
EVENTS_PATH = TASK_DIR / "logs" / "full_training_fit_events_e001.jsonl"
RESULT_ROOT = TASK_DIR / "runtime" / "job_results_e001"
CHECKPOINT_ROOT = TASK_DIR / "runtime" / "context_checkpoints_e001"
PROGRESS_PATH = TASK_DIR / "runtime" / "full_training_progress.json"
PAUSE_RECORD_PATH = TASK_DIR / "runtime" / "reboot_pause_20260726T121408Z.json"
OUTPUT_ROOT = TASK_DIR / "predictions" / "full_e001"
MODEL_ROOT = TASK_DIR / "models" / "full_e001"
FEATURE_ROOT = TASK_DIR / "features" / "full_e001"
METRIC_ROOT = TASK_DIR / "metrics" / "full_e001"

OOF_PATH = TASK_DIR / "oof_predictions.csv"
VALIDATION_METRICS_PATH = TASK_DIR / "validation_metrics.csv"
CALIBRATION_METRICS_PATH = TASK_DIR / "calibration_metrics.csv"
SENSITIVITY_METRICS_PATH = TASK_DIR / "label_sensitivity_metrics.csv"
ENVIRONMENT_PATH = TASK_DIR / "environment" / "full_training_environment_e001.json"
JOB_HASH_MANIFEST_PATH = TASK_DIR / "manifests" / "full_training_job_result_hashes_e001.csv"
OUTPUT_MANIFEST_PATH = TASK_DIR / "manifests" / "full_training_output_manifest_e001.csv"
VALIDATION_REPORT_PATH = TASK_DIR / "reports" / "full_training_integrity_validation_e001.json"
PACKAGE_REPORT_PATH = TASK_DIR / "reports" / "full_training_package_verification_e001.json"
MODEL_LOCK_PATH = TASK_DIR / "MODEL_LOCK.json"
STATUS_PATH = TASK_DIR / "STATUS.json"
HANDOFF_PATH = TASK_DIR / "HANDOFF.md"
REQUEST_PATH = TASK_DIR / "TASK00_MODEL_LOCK_REVIEW_REQUEST_E001.json"
REQUEST_MD_PATH = TASK_DIR / "TASK00_MODEL_LOCK_REVIEW_REQUEST_E001.md"
VALIDATION_LOG_PATH = TASK_DIR / "logs" / "full_training_integrity_validation_e001.log"

EXPECTED_PROTOCOL_SHA256 = "2fc3cc0df70895643440dec40e27f51d045b88c62c7c50a8370af5688d592663"
EXPECTED_GO_FULL_SHA256 = "dae60ff147881db239e6b2f36dd0843f8f69687da8e4a9435fb9f9d4ff31dda4"
EXPECTED_SUPPLEMENT_SHA256 = "8ed6b3828c3acc53c2eca182833ad2b9e27fd6cfb099d358146f4ccba165c045"
EXPECTED_PLAN_SHA256 = "03ed011fed5b2b3270bb6106eeb76d1dc284ef8e60ccd412e8975aefe2f44ca5"
EXPECTED_SPLIT_INDEX_SHA256 = "5ea19ab7c5d7c89bbbc79110bd2baa7fd86690c389127d2f88c9cba7aba41b8f"
EXPECTED_DATASET_LOCK_SHA256 = "ae431ad4990ad6c0d2fc33ad3254d9b73ba2ba65732d0403461a6ccca620c96a"
EXPECTED_MAPPING_LOCK_SHA256 = "75675625ddd52736ba792bede407124f7fb44d00561f5b16d2db896f9bb32d32"
EXPECTED_FROZEN_MANIFEST_SHA256 = "b82a3fb4708759fb0f362c89944249868b45b3136005f28292e26eb963c689e1"

PLANNED_JOBS = 26498
HARD_CEILING = 27300
EXPECTED_ATTEMPTS = 26499
EXPECTED_MEMBERS = 14
TOLERANCE = 1e-10
PLAN_COLUMNS = [
    "job_id", "planned_ordinal", "phase", "scenario_id", "design_id",
    "outer_fold", "inner_fold", "stage", "member", "branch", "family",
    "candidate_id", "candidate_ordinal", "seed", "backend_policy",
    "parameters_json",
]


class IntegrityError(RuntimeError):
    pass


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def hash_ids(values) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()


def array_hash(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8).tobytes()).hexdigest()


def active_training_processes() -> list[dict]:
    matches = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            command = process.info.get("cmdline") or []
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        if any(Path(argument).name == "run_full_training_e001.py" for argument in command):
            matches.append({"pid": int(process.info["pid"]), "cmdline": command})
    return matches


def best_mcc_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], np.asarray(probability, dtype=float))))
    ranked = []
    for threshold in candidates:
        mcc = matthews_corrcoef(y, (probability >= threshold).astype(int))
        ranked.append((float(mcc), -abs(float(threshold) - 0.5), -float(threshold), float(threshold)))
    ranked.sort(reverse=True)
    return ranked[0][3]


class Validator:
    def __init__(self) -> None:
        self.checks: dict[str, dict] = {}

    def require(self, name: str, condition: bool, **details) -> None:
        self.checks[name] = {"status": "PASS" if condition else "FAIL", **details}
        if not condition:
            raise IntegrityError(f"{name}: {details}")


def ordered_indices(data: pd.DataFrame, identifiers) -> np.ndarray:
    requested = set(map(str, identifiers))
    selected = np.flatnonzero(data["model_unit_id"].astype(str).isin(requested).to_numpy())
    if len(selected) != len(requested):
        raise IntegrityError("Manifest identifiers do not align one-to-one with frozen data")
    return selected.astype(np.int64)


def finite_probability(values, context: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all() or np.any(array < 0.0) or np.any(array > 1.0):
        raise IntegrityError(f"Invalid probability: {context}")
    return array


def finite_scores(values, context: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise IntegrityError(f"Non-finite raw score: {context}")
    return array


def ece_equal_width(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    value = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (probability >= edges[index]) & (probability <= edges[index + 1])
        else:
            mask = (probability >= edges[index]) & (probability < edges[index + 1])
        count = int(mask.sum())
        if count:
            value += count / total * abs(float(probability[mask].mean()) - float(y[mask].mean()))
    return float(value)


def calibration_line_cuda(y: np.ndarray, probability: np.ndarray) -> tuple[float, float, int]:
    """Metric-only logistic recalibration line on CUDA; no model is retained."""
    if not torch.cuda.is_available():
        raise IntegrityError("CUDA is required for calibration slope/intercept metric calculation")
    clipped = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    logit = np.log(clipped / (1.0 - clipped))
    device = torch.device("cuda:0")
    x = torch.as_tensor(np.column_stack([np.ones(len(logit)), logit]), dtype=torch.float64, device=device)
    target = torch.as_tensor(y, dtype=torch.float64, device=device)
    beta = torch.tensor([0.0, 1.0], dtype=torch.float64, device=device)

    def likelihood(parameter: torch.Tensor) -> torch.Tensor:
        eta = x @ parameter
        return torch.sum(target * torch.nn.functional.logsigmoid(eta) + (1.0 - target) * torch.nn.functional.logsigmoid(-eta))

    iterations = 0
    for iterations in range(1, 101):
        eta = x @ beta
        mean = torch.sigmoid(eta)
        gradient = x.T @ (target - mean)
        weight = torch.clamp(mean * (1.0 - mean), min=1e-12)
        information = x.T @ (weight[:, None] * x) + torch.eye(2, dtype=torch.float64, device=device) * 1e-10
        try:
            step = torch.linalg.solve(information, gradient)
        except RuntimeError:
            step = torch.linalg.pinv(information) @ gradient
        old_likelihood = likelihood(beta)
        scale = 1.0
        proposed = beta + step
        while scale > 1e-6 and likelihood(proposed) < old_likelihood:
            scale *= 0.5
            proposed = beta + step * scale
        beta = proposed
        if float(torch.max(torch.abs(step * scale)).item()) < 1e-10:
            break
    intercept, slope = [float(item) for item in beta.detach().cpu().tolist()]
    del x, target, beta
    torch.cuda.empty_cache()
    if not math.isfinite(intercept) or not math.isfinite(slope):
        raise IntegrityError("Non-finite calibration intercept/slope")
    return intercept, slope, iterations


def discrimination_classification_metrics(
    *, y: np.ndarray, probability: np.ndarray, predicted: np.ndarray,
    threshold_rule: str, threshold_value: float | None, base: dict,
) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        **base,
        "threshold_rule": threshold_rule,
        "threshold_value": threshold_value,
        "n": int(len(y)),
        "positive_n": int(np.sum(y == 1)),
        "negative_n": int(np.sum(y == 0)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "auprc": float(average_precision_score(y, probability)),
        "mcc": float(matthews_corrcoef(y, predicted)),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def calibration_metrics(*, y: np.ndarray, probability: np.ndarray, base: dict) -> dict:
    intercept, slope, iterations = calibration_line_cuda(y, probability)
    return {
        **base,
        "n": int(len(y)),
        "brier_score": float(brier_score_loss(y, probability)),
        "ece_equal_width_10": ece_equal_width(y, probability, bins=10),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "calibration_metric_device": "cuda:0",
        "calibration_metric_iterations": iterations,
        "calibration_metric_role": "evaluation_only_not_saved_not_used_for_selection",
    }


def assert_device_evidence(plan_row: dict, result: dict, device_counts: Counter) -> bool:
    evidence = result.get("device_evidence")
    if not isinstance(evidence, dict):
        raise IntegrityError(f"Missing device evidence: {plan_row['job_id']}")
    backend_policy = plan_row["backend_policy"]
    family = plan_row["family"]
    if backend_policy == "cpu_p002_whitelist":
        if family == "RandomForest":
            if evidence.get("estimator_fqn") != "sklearn.ensemble.RandomForestClassifier":
                raise IntegrityError(f"RF whitelist violation: {plan_row['job_id']}")
        elif family == "RBF_SVM":
            params = evidence.get("estimator_parameters", {})
            preprocessing = evidence.get("gpu_preprocessing", {})
            if evidence.get("estimator_fqn") != "sklearn.svm.SVC" or params.get("kernel") != "rbf" or params.get("probability") is not False:
                raise IntegrityError(f"SVC whitelist/probability violation: {plan_row['job_id']}")
            if preprocessing.get("device") != "cuda:0":
                raise IntegrityError(f"SVC fitted standardization was not CUDA: {plan_row['job_id']}")
        else:
            raise IntegrityError(f"Non-whitelisted CPU family: {plan_row['job_id']} {family}")
        if evidence.get("fit_device") != "cpu_p002_whitelist" or evidence.get("whitelist_status") != "PASS":
            raise IntegrityError(f"CPU evidence violation: {plan_row['job_id']}")
        device_counts[f"cpu:{family}"] += 1
        return False

    if backend_policy != "cuda_required":
        raise IntegrityError(f"Unknown backend policy: {plan_row['job_id']}")
    if family == "calibrator":
        ok = evidence.get("backend") == "native_pytorch_cuda_two_parameter_platt" and evidence.get("device") == "cuda:0"
    elif family == "XGBoost":
        parameters = json.loads(plan_row["parameters_json"])
        ok = (
            evidence.get("backend") == "xgboost"
            and evidence.get("fit_device") == "cuda:0"
            and evidence.get("warnings") == []
            and (not parameters or parameters.get("device") == "cuda")
        )
    elif family == "LightGBM":
        ok = evidence.get("backend") == "lightgbm" and evidence.get("fit_device") == "cuda:0"
    elif family == "graph":
        ok = evidence.get("backend") == "native_pytorch_cuda" and evidence.get("device") == "cuda:0" and int(evidence.get("early_stop_group_overlap_count", -1)) == 0
    elif family == "language_model":
        ok = evidence.get("backend") == "native_pytorch_cuda_logistic" and evidence.get("device") == "cuda:0"
    else:
        ok = False
    if not ok:
        raise IntegrityError(f"Required CUDA evidence violation: {plan_row['job_id']} {family}")
    device_counts[f"cuda:{family}"] += 1
    return family == "XGBoost" and evidence.get("prediction_device") == "not_yet_predicted"


def load_governance(validator: Validator):
    expected = {
        PROTOCOL_PATH: EXPECTED_PROTOCOL_SHA256,
        GO_FULL_PATH: EXPECTED_GO_FULL_SHA256,
        SUPPLEMENT_PATH: EXPECTED_SUPPLEMENT_SHA256,
        PLAN_PATH: EXPECTED_PLAN_SHA256,
        SPLIT_INDEX_PATH: EXPECTED_SPLIT_INDEX_SHA256,
        REPO_ROOT / "revision_management/01_data_audit/DATASET_LOCK.json": EXPECTED_DATASET_LOCK_SHA256,
        REPO_ROOT / "revision_management/01_data_audit/MAPPING_LOCK.json": EXPECTED_MAPPING_LOCK_SHA256,
        REPO_ROOT / "revision_management/01_data_audit/frozen/FROZEN_INPUT_MANIFEST.csv": EXPECTED_FROZEN_MANIFEST_SHA256,
    }
    observed = {relative(path): sha256_file(path) for path in expected}
    validator.require("governance_hashes_match", all(observed[relative(path)] == digest for path, digest in expected.items()), observed=observed)
    dataset_lock = json.loads((REPO_ROOT / "revision_management/01_data_audit/DATASET_LOCK.json").read_text(encoding="utf-8"))
    approval = json.loads(GO_FULL_PATH.read_text(encoding="utf-8"))
    supplement = json.loads(SUPPLEMENT_PATH.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    validator.require("task01_lock_authorized", dataset_lock.get("status") == "LOCKED" and dataset_lock.get("downstream_training_authorized") is True)
    validator.require("go_full_scope_valid", approval.get("status") == "APPROVED" and approval.get("full_training_authorized") is True and approval.get("model_lock_authorized") is False)
    validator.require("supplement_scope_valid", supplement.get("status") == "APPROVED_EXECUTABLE_SUPPLEMENT" and supplement["authorization"]["model_lock_authorized"] is False)
    return protocol, approval, supplement, observed


def load_splits(protocol: dict, validator: Validator):
    index = pd.read_csv(SPLIT_INDEX_PATH, dtype=str, keep_default_na=False)
    validator.require("split_index_has_ten_entries", len(index) == 10, observed=int(len(index)))
    data_by_scenario: dict[str, pd.DataFrame] = {}
    split_by_context: dict[tuple[str, str], pd.DataFrame] = {}
    split_validations = []
    specs = {item["scenario_id"]: item for item in protocol["inputs"]}
    for spec in protocol["inputs"]:
        data = load_locked_scenario(REPO_ROOT / spec["path"], spec["scenario_id"], spec["sha256"], int(spec["rows"]))
        data_by_scenario[spec["scenario_id"]] = data.reset_index(drop=True)
    for row in index.itertuples(index=False):
        spec = specs[row.scenario_id]
        path = REPO_ROOT / row.manifest_path
        observed = sha256_file(path)
        validator.require(f"split_hash:{row.scenario_id}:{row.design_id}", observed == row.manifest_sha256, observed=observed, expected=row.manifest_sha256)
        validator.require(f"split_input_hash:{row.scenario_id}:{row.design_id}", row.input_sha256 == spec["sha256"])
        manifest = pd.read_csv(path, dtype={"model_unit_id": str, "cv_group_id": str, "split_group_id": str})
        result = validate_manifest(manifest, data_by_scenario[row.scenario_id], row.design_id)
        validator.require(f"split_leakage_and_coverage:{row.scenario_id}:{row.design_id}", result["status"] == "PASS", **result)
        split_by_context[(row.scenario_id, row.design_id)] = manifest
        split_validations.append({"scenario_id": row.scenario_id, "design_id": row.design_id, "manifest_sha256": observed, **result})
    return index, data_by_scenario, split_by_context, split_validations


def split_indices(data: pd.DataFrame, manifest: pd.DataFrame, outer_fold: int):
    outer = manifest[(manifest["level"] == "outer") & (manifest["outer_fold"] == outer_fold)]
    outer_train = ordered_indices(data, outer.loc[outer["subset"] == "train", "model_unit_id"])
    outer_test = ordered_indices(data, outer.loc[outer["subset"] == "test", "model_unit_id"])
    inner = {}
    for inner_fold in range(5):
        frame = manifest[(manifest["level"] == "inner") & (manifest["outer_fold"] == outer_fold) & (manifest["inner_fold"] == inner_fold)]
        inner[inner_fold] = (
            ordered_indices(data, frame.loc[frame["subset"] == "train", "model_unit_id"]),
            ordered_indices(data, frame.loc[frame["subset"] == "validation", "model_unit_id"]),
        )
    return outer_train, outer_test, inner


def validate_plan(protocol: dict, supplement: dict, validator: Validator):
    plan = pd.read_csv(PLAN_PATH, dtype=str, keep_default_na=False)
    validator.require("execution_plan_columns", list(plan.columns) == PLAN_COLUMNS, observed=list(plan.columns))
    validator.require("execution_plan_count", len(plan) == PLANNED_JOBS, observed=int(len(plan)))
    validator.require("execution_plan_job_ids_unique", plan["job_id"].is_unique)
    validator.require("execution_plan_ordinals_exact", plan["planned_ordinal"].astype(int).tolist() == list(range(1, PLANNED_JOBS + 1)))
    stage_counts = plan["stage"].value_counts().to_dict()
    expected_stage_counts = {
        "inner_candidate_fit": 21500,
        "selected_member_crossfit": 3500,
        "gpu_platt_calibration": 700,
        "selected_member_outer_refit": 700,
        "final_primary_crossfit": 70,
        "final_primary_gpu_platt_calibration": 14,
        "final_primary_full_refit": 14,
    }
    validator.require("execution_plan_stage_counts", stage_counts == expected_stage_counts, observed=stage_counts)
    phase_counts = plan["phase"].value_counts().to_dict()
    validator.require("execution_plan_phase_counts", phase_counts == {"nested_cv": 26400, "final_primary": 98}, observed=phase_counts)
    scenario_counts = plan.loc[plan["phase"] == "nested_cv", "scenario_id"].value_counts().to_dict()
    validator.require("execution_plan_scenario_counts", set(scenario_counts.values()) == {5280} and len(scenario_counts) == 5, observed=scenario_counts)
    backend_counts = plan["backend_policy"].value_counts().to_dict()
    validator.require("execution_plan_backend_counts", backend_counts == {"cuda_required": 18274, "cpu_p002_whitelist": 8224}, observed=backend_counts)
    members = candidate_registry(protocol, supplement)
    validator.require("candidate_registry_members", len(members) == EXPECTED_MEMBERS, observed=len(members))
    return plan, {row.job_id: row._asdict() for row in plan.itertuples(index=False)}, members, stage_counts, backend_counts


def validate_events(plan_map: dict[str, dict], validator: Validator):
    started = Counter()
    completed = Counter()
    failed = Counter()
    event_rows = 0
    with EVENTS_PATH.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            event_rows += 1
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise IntegrityError(f"Invalid event JSON at line {line_number}") from exc
            job_id = event.get("job_id")
            if job_id not in plan_map:
                raise IntegrityError(f"Unregistered job in event ledger: {job_id}")
            if event.get("event") == "FIT_ATTEMPT_STARTED":
                started[job_id] += 1
                for field in PLAN_COLUMNS:
                    if str(event.get(field, "")) != str(plan_map[job_id][field]):
                        raise IntegrityError(f"Event-plan mismatch {job_id} {field}")
            elif event.get("event") == "FIT_ATTEMPT_COMPLETED":
                completed[job_id] += 1
            elif event.get("event") == "FIT_ATTEMPT_FAILED":
                failed[job_id] += 1
            else:
                raise IntegrityError(f"Unknown event type at line {line_number}")
    validator.require("event_row_count", event_rows == 52997, observed=event_rows)
    validator.require("event_started_count", sum(started.values()) == EXPECTED_ATTEMPTS, observed=sum(started.values()))
    validator.require("event_completed_count", sum(completed.values()) == PLANNED_JOBS, observed=sum(completed.values()))
    validator.require("event_failed_count", sum(failed.values()) == 0, observed=sum(failed.values()))
    unusual = {job: count for job, count in started.items() if count != 1}
    validator.require("event_only_registered_reboot_retry", unusual == {"E001-J11471": 2}, observed=unusual)
    validator.require("event_one_completion_per_job", len(completed) == PLANNED_JOBS and all(value == 1 for value in completed.values()))
    pause = json.loads(PAUSE_RECORD_PATH.read_text(encoding="utf-8"))
    validator.require(
        "reboot_interruption_accounted",
        pause.get("interrupted_job_id") == "E001-J11471" and pause.get("interrupted_attempt_counts_toward_ceiling") is True and pause.get("process_stopped") is True,
        pause_record=relative(PAUSE_RECORD_PATH),
    )
    return {"event_rows": event_rows, "started": sum(started.values()), "completed": sum(completed.values()), "failed": sum(failed.values()), "reboot_retry_job": "E001-J11471"}


def validate_results(
    *, plan: pd.DataFrame, plan_map: dict[str, dict], data_by_scenario: dict[str, pd.DataFrame],
    split_by_context: dict[tuple[str, str], pd.DataFrame], validator: Validator,
):
    result_paths = sorted(RESULT_ROOT.glob("E001-J*.json"))
    expected_ids = set(plan["job_id"])
    observed_ids = {path.stem for path in result_paths}
    validator.require("job_result_file_count", len(result_paths) == PLANNED_JOBS, observed=len(result_paths))
    validator.require("job_result_id_coverage", observed_ids == expected_ids, missing=sorted(expected_ids - observed_ids)[:10], extra=sorted(observed_ids - expected_ids)[:10])
    folds = {}
    for (scenario, design), manifest in split_by_context.items():
        data = data_by_scenario[scenario]
        for outer_fold in range(5):
            folds[(scenario, design, outer_fold)] = split_indices(data, manifest, outer_fold)

    device_counts = Counter()
    audit_counts = Counter()
    inner_metrics = defaultdict(dict)
    model_paths_from_results = set()
    job_hash_rows = []
    result_attempt_indices = Counter()
    embedded_job_ids = set()

    for ordinal, path in enumerate(result_paths, start=1):
        raw_bytes = path.read_bytes()
        payload = json.loads(raw_bytes)
        job_id = payload.get("job_id")
        if job_id not in plan_map or path.stem != job_id or job_id in embedded_job_ids:
            raise IntegrityError(f"Job-result identity mismatch or duplicate: {relative(path)} -> {job_id}")
        embedded_job_ids.add(job_id)
        row = plan_map[job_id]
        result_attempt_indices[int(payload.get("attempt_index", -1))] += 1
        if payload.get("plan") != {field: row[field] for field in PLAN_COLUMNS}:
            raise IntegrityError(f"Job result embeds a nonmatching plan row: {job_id}")
        if not math.isfinite(float(payload.get("duration_seconds", math.nan))) or float(payload["duration_seconds"]) < 0:
            raise IntegrityError(f"Invalid job duration: {job_id}")
        result = payload.get("result")
        if not isinstance(result, dict):
            raise IntegrityError(f"Missing result payload: {job_id}")
        if assert_device_evidence(row, result, device_counts):
            audit_counts["xgboost_prediction_device_snapshot_not_yet_predicted"] += 1
        scenario = row["scenario_id"]
        design = row["design_id"]
        stage = row["stage"]
        data = data_by_scenario[scenario]
        y = data["final_label"].to_numpy(dtype=int)
        expected_training_indices = None
        expected_prediction_indices = None
        if stage in {"inner_candidate_fit", "selected_member_crossfit"}:
            _, _, inner = folds[(scenario, design, int(row["outer_fold"]))]
            expected_training_indices, expected_prediction_indices = inner[int(row["inner_fold"])]
        elif stage in {"selected_member_outer_refit", "final_primary_crossfit"}:
            expected_training_indices, expected_prediction_indices, _ = folds[(scenario, design, int(row["outer_fold"]))]

        if expected_training_indices is not None:
            if int(result.get("training_units", -1)) != len(expected_training_indices):
                raise IntegrityError(f"Training length mismatch: {job_id}")
            if result.get("training_index_sha256") != hash_ids(expected_training_indices):
                raise IntegrityError(f"Training index hash mismatch: {job_id}")
            if set(expected_training_indices) & set(expected_prediction_indices):
                raise IntegrityError(f"Training/prediction index overlap: {job_id}")

        if expected_prediction_indices is not None:
            scores = finite_scores(result.get("raw_scores", []), job_id)
            if len(scores) != len(expected_prediction_indices) or int(result.get("prediction_units", -1)) != len(expected_prediction_indices):
                raise IntegrityError(f"Prediction length mismatch: {job_id}")
            expected_hash = hash_ids(expected_prediction_indices)
            if result.get("prediction_index_sha256") != expected_hash:
                raise IntegrityError(f"Prediction index hash mismatch: {job_id}")
            if stage == "inner_candidate_fit":
                labels = y[expected_prediction_indices]
                key = (scenario, design, int(row["outer_fold"]), row["member"], row["candidate_id"])
                inner_metrics[key][int(row["inner_fold"])] = {
                    "roc_auc": float(roc_auc_score(labels, scores)),
                    "auprc": float(average_precision_score(labels, scores)),
                }
        elif stage == "gpu_platt_calibration":
            outer_train = folds[(scenario, design, int(row["outer_fold"]))][0]
            probability = finite_probability(result.get("train_probabilities", []), job_id)
            if len(probability) != len(outer_train):
                raise IntegrityError(f"Nested calibrator coverage mismatch: {job_id}")
        elif stage == "final_primary_gpu_platt_calibration":
            probability = finite_probability(result.get("probabilities", []), job_id)
            if len(probability) != len(data):
                raise IntegrityError(f"Final calibrator coverage mismatch: {job_id}")
        elif stage == "final_primary_full_refit":
            if int(result.get("training_units", -1)) != len(data):
                raise IntegrityError(f"Final refit training count mismatch: {job_id}")

        if "model_path" in result:
            model_path = REPO_ROOT / result["model_path"]
            if not model_path.is_file():
                raise IntegrityError(f"Recorded model file missing: {result['model_path']}")
            model_paths_from_results.add(model_path.resolve())
        job_hash_rows.append({
            "job_id": job_id,
            "planned_ordinal": int(row["planned_ordinal"]),
            "attempt_index": int(payload["attempt_index"]),
            "phase": row["phase"],
            "scenario_id": scenario,
            "design_id": design,
            "stage": stage,
            "member": row["member"],
            "backend_policy": row["backend_policy"],
            "size_bytes": len(raw_bytes),
            "sha256": bytes_sha256(raw_bytes),
            "path": relative(path),
        })
        if ordinal % 2500 == 0:
            print(f"validated_job_results={ordinal}/{PLANNED_JOBS}", flush=True)

    validator.require("job_result_embedded_identity_coverage", embedded_job_ids == expected_ids, observed=len(embedded_job_ids))
    validator.require("job_result_attempt_indices", result_attempt_indices == Counter({1: PLANNED_JOBS - 1, 2: 1}), observed=dict(result_attempt_indices))
    validator.require("device_evidence_counts", sum(device_counts.values()) == PLANNED_JOBS, observed=dict(device_counts))
    validator.require("cpu_device_evidence_count", sum(v for k, v in device_counts.items() if k.startswith("cpu:")) == 8224, observed=dict(device_counts))
    validator.require("cuda_device_evidence_count", sum(v for k, v in device_counts.items() if k.startswith("cuda:")) == 18274, observed=dict(device_counts))
    validator.require(
        "xgboost_prediction_evidence_snapshot_count",
        audit_counts["xgboost_prediction_device_snapshot_not_yet_predicted"] == 4612,
        observed=dict(audit_counts),
    )
    nested_model_files = {path.resolve() for path in (MODEL_ROOT / "nested").rglob("*") if path.is_file()}
    final_model_files = {path.resolve() for path in (MODEL_ROOT / "final_primary").iterdir() if path.is_file() and path.name != "model_manifest.json"}
    validator.require("nested_model_file_count", len(nested_model_files) == 1400, observed=len(nested_model_files))
    validator.require("final_model_file_count", len(final_model_files) == 28, observed=len(final_model_files))
    validator.require("recorded_model_path_coverage", model_paths_from_results == nested_model_files | final_model_files, recorded=len(model_paths_from_results), actual=len(nested_model_files | final_model_files))
    job_hash_frame = pd.DataFrame(job_hash_rows).sort_values("planned_ordinal", kind="mergesort").reset_index(drop=True)
    return inner_metrics, device_counts, audit_counts, job_hash_frame, folds


def compare_float(observed, expected, tolerance=TOLERANCE) -> bool:
    return math.isclose(float(observed), float(expected), rel_tol=tolerance, abs_tol=tolerance)


def validate_candidate_selection(
    *, members: list[dict], inner_metrics: dict, protocol: dict, validator: Validator,
):
    member_map = {item["member"]: item for item in members}
    contexts = {}
    selection_files = sorted((METRIC_ROOT / "candidate_selection").rglob("outer_*.json"))
    validator.require("candidate_selection_file_count", len(selection_files) == 50, observed=len(selection_files))
    for scenario in [item["scenario_id"] for item in protocol["inputs"]]:
        for design in protocol["split_policy"]["designs"]:
            for outer_fold in range(5):
                path = METRIC_ROOT / "candidate_selection" / scenario / design / f"outer_{outer_fold}.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                validator.require(f"candidate_selection_member_count:{scenario}:{design}:{outer_fold}", len(payload["members"]) == EXPECTED_MEMBERS)
                observed_members = {entry["member"]: entry for entry in payload["members"]}
                validator.require(f"candidate_selection_member_identity:{scenario}:{design}:{outer_fold}", set(observed_members) == set(member_map))
                context_ranking = {}
                for member_name, member in member_map.items():
                    ranking = []
                    for candidate in member["candidates"]:
                        key = (scenario, design, outer_fold, member_name, candidate["id"])
                        metrics = inner_metrics.get(key, {})
                        if set(metrics) != set(range(5)):
                            raise IntegrityError(f"Incomplete inner candidate metrics: {key}")
                        aucs = [metrics[index]["roc_auc"] for index in range(5)]
                        auprcs = [metrics[index]["auprc"] for index in range(5)]
                        ranking.append({
                            "candidate_id": candidate["id"],
                            "candidate_ordinal": int(candidate["ordinal"]),
                            "mean_inner_roc_auc": float(np.mean(aucs)),
                            "mean_inner_auprc": float(np.mean(auprcs)),
                            "population_std_inner_roc_auc": float(np.std(aucs, ddof=0)),
                        })
                    ranking.sort(key=lambda item: (-item["mean_inner_roc_auc"], -item["mean_inner_auprc"], item["population_std_inner_roc_auc"], item["candidate_ordinal"]))
                    observed = observed_members[member_name]
                    if observed["selected_candidate_id"] != ranking[0]["candidate_id"] or int(observed["selected_candidate_ordinal"]) != ranking[0]["candidate_ordinal"]:
                        raise IntegrityError(f"Candidate selection mismatch: {scenario} {design} {outer_fold} {member_name}")
                    observed_ranking = observed["ranking"]
                    if [item["candidate_id"] for item in observed_ranking] != [item["candidate_id"] for item in ranking]:
                        raise IntegrityError(f"Candidate ranking order mismatch: {scenario} {design} {outer_fold} {member_name}")
                    for left, right in zip(observed_ranking, ranking):
                        for field in ("mean_inner_roc_auc", "mean_inner_auprc", "population_std_inner_roc_auc"):
                            if not compare_float(left[field], right[field]):
                                raise IntegrityError(f"Candidate metric mismatch: {scenario} {design} {outer_fold} {member_name} {field}")
                    context_ranking[member_name] = ranking
                contexts[(scenario, design, outer_fold)] = {"payload": payload, "ranking": context_ranking}
    validator.require("all_candidate_selections_recomputed", len(contexts) == 50, contexts=len(contexts), member_contexts=50 * EXPECTED_MEMBERS)
    return contexts


def validate_prediction_frame(frame: pd.DataFrame, data: pd.DataFrame, expected_ids: set[str], members: list[str], context: str, raw_column: str):
    if set(frame["member"].astype(str)) != set(members):
        raise IntegrityError(f"Prediction member identity mismatch: {context}")
    expected_rows = len(expected_ids) * len(members)
    if len(frame) != expected_rows:
        raise IntegrityError(f"Prediction row count mismatch: {context}")
    reference = data.set_index("model_unit_id")[["cv_group_id", "final_label"]]
    for member in members:
        subset = frame[frame["member"] == member]
        ids = subset["model_unit_id"].astype(str)
        if ids.duplicated().any() or set(ids) != expected_ids:
            raise IntegrityError(f"Prediction coverage mismatch: {context} {member}")
        aligned = reference.loc[ids]
        if aligned["cv_group_id"].astype(str).tolist() != subset["cv_group_id"].astype(str).tolist() or aligned["final_label"].astype(int).tolist() != subset["final_label"].astype(int).tolist():
            raise IntegrityError(f"Prediction label/group mismatch: {context} {member}")
        finite_probability(subset["calibrated_probability"], f"{context}:{member}")
        if member != "FINAL_ENSEMBLE":
            finite_scores(subset[raw_column], f"{context}:{member}")


def aligned_probabilities(frame: pd.DataFrame, member: str, ordered_ids: list[str]) -> np.ndarray:
    series = frame[frame["member"] == member].set_index("model_unit_id")["calibrated_probability"]
    return series.loc[ordered_ids].to_numpy(dtype=np.float64)


def add_metric_rows(
    frame: pd.DataFrame, *, scenario: str, design: str, outer_fold: int | None,
    aggregation_scope: str, member_names: list[str], validation_rows: list[dict], calibration_rows: list[dict],
) -> None:
    ordered_ids = sorted(frame["model_unit_id"].astype(str).unique())
    for member in member_names:
        subset = frame[frame["member"] == member].set_index("model_unit_id").loc[ordered_ids]
        y = subset["final_label"].to_numpy(dtype=int)
        probability = subset["calibrated_probability"].to_numpy(dtype=np.float64)
        base = {
            "scenario_id": scenario,
            "design_id": design,
            "aggregation_scope": aggregation_scope,
            "outer_fold": outer_fold,
            "member": member,
            "branch": str(subset["branch"].iloc[0]),
        }
        fixed_predicted = (probability >= 0.5).astype(int)
        validation_rows.append(discrimination_classification_metrics(
            y=y, probability=probability, predicted=fixed_predicted,
            threshold_rule="fixed_0.5", threshold_value=0.5, base=base,
        ))
        if member == "FINAL_ENSEMBLE":
            if aggregation_scope == "outer_fold":
                thresholds = subset["threshold"].dropna().astype(float).unique()
                if len(thresholds) != 1:
                    raise IntegrityError(f"Fold ensemble threshold is not unique: {scenario} {design} {outer_fold}")
                predicted = subset["predicted_label"].astype(int).to_numpy()
                threshold_value = float(thresholds[0])
            else:
                predicted = subset["predicted_label"].astype(int).to_numpy()
                threshold_value = None
            validation_rows.append(discrimination_classification_metrics(
                y=y, probability=probability, predicted=predicted,
                threshold_rule="outer_train_MCC_fold_specific", threshold_value=threshold_value, base=base,
            ))
        calibration_rows.append(calibration_metrics(y=y, probability=probability, base=base))


def validate_predictions_and_metrics(
    *, protocol: dict, members: list[dict], contexts: dict, data_by_scenario: dict[str, pd.DataFrame],
    folds: dict, validator: Validator,
):
    member_names = [item["member"] for item in members] + ["FINAL_ENSEMBLE"]
    branch_map = {item["member"]: item["branch"] for item in members}
    nested_test_files = sorted((OUTPUT_ROOT / "nested").rglob("outer_*_test_predictions.csv"))
    nested_train_files = sorted((OUTPUT_ROOT / "nested").rglob("outer_*_train_crossfit.csv"))
    validator.require("nested_prediction_file_counts", len(nested_test_files) == 50 and len(nested_train_files) == 50, test_files=len(nested_test_files), train_files=len(nested_train_files))
    oof_parts = []
    validation_rows: list[dict] = []
    calibration_rows: list[dict] = []
    pooled_parts = defaultdict(list)

    for scenario in [item["scenario_id"] for item in protocol["inputs"]]:
        data = data_by_scenario[scenario]
        for design in protocol["split_policy"]["designs"]:
            for outer_fold in range(5):
                outer_train, outer_test, _ = folds[(scenario, design, outer_fold)]
                train_ids = set(data.iloc[outer_train]["model_unit_id"].astype(str))
                test_ids = set(data.iloc[outer_test]["model_unit_id"].astype(str))
                if train_ids & test_ids:
                    raise IntegrityError(f"Train/test prediction ID overlap: {scenario} {design} {outer_fold}")
                directory = OUTPUT_ROOT / "nested" / scenario / design
                train = pd.read_csv(directory / f"outer_{outer_fold}_train_crossfit.csv")
                test = pd.read_csv(directory / f"outer_{outer_fold}_test_predictions.csv")
                validate_prediction_frame(train, data, train_ids, member_names, f"train:{scenario}:{design}:{outer_fold}", "raw_crossfit_score")
                validate_prediction_frame(test, data, test_ids, member_names, f"test:{scenario}:{design}:{outer_fold}", "raw_score")
                selection = contexts[(scenario, design, outer_fold)]["payload"]
                selected_ids = {entry["member"]: entry["selected_candidate_id"] for entry in selection["members"]}
                for member in members:
                    observed = set(test.loc[test["member"] == member["member"], "selected_candidate_id"].astype(str))
                    if observed != {selected_ids[member["member"]]}:
                        raise IntegrityError(f"Prediction candidate mismatch: {scenario} {design} {outer_fold} {member['member']}")
                for frame, ids, context_name in ((train, sorted(train_ids), "train"), (test, sorted(test_ids), "test")):
                    branch_probability = {}
                    for branch in ("fingerprint", "graph", "language_model"):
                        arrays = [aligned_probabilities(frame, member["member"], ids) for member in members if member["branch"] == branch]
                        branch_probability[branch] = np.mean(arrays, axis=0)
                    recomputed = np.mean(list(branch_probability.values()), axis=0)
                    observed = aligned_probabilities(frame, "FINAL_ENSEMBLE", ids)
                    if not np.allclose(recomputed, observed, rtol=0.0, atol=TOLERANCE):
                        raise IntegrityError(f"Branch-balanced ensemble mismatch: {scenario} {design} {outer_fold} {context_name}")
                train_ensemble = train[train["member"] == "FINAL_ENSEMBLE"].set_index("model_unit_id").loc[sorted(train_ids)]
                threshold = best_mcc_threshold(train_ensemble["final_label"].to_numpy(dtype=int), train_ensemble["calibrated_probability"].to_numpy(dtype=float))
                if not compare_float(threshold, selection["threshold"]):
                    raise IntegrityError(f"Registered threshold mismatch: {scenario} {design} {outer_fold}")
                test_ensemble = test[test["member"] == "FINAL_ENSEMBLE"]
                if not np.allclose(test_ensemble["threshold"].astype(float), threshold, rtol=0.0, atol=TOLERANCE):
                    raise IntegrityError(f"Outer-test threshold application mismatch: {scenario} {design} {outer_fold}")
                expected_predicted = (test_ensemble["calibrated_probability"].to_numpy(dtype=float) >= threshold).astype(int)
                if not np.array_equal(expected_predicted, test_ensemble["predicted_label"].astype(int).to_numpy()):
                    raise IntegrityError(f"Outer-test class prediction mismatch: {scenario} {design} {outer_fold}")
                test = test.copy()
                test["prediction_scope"] = "nested_outer_test_oof"
                oof_parts.append(test)
                pooled_parts[(scenario, design)].append(test)
                add_metric_rows(
                    test, scenario=scenario, design=design, outer_fold=outer_fold,
                    aggregation_scope="outer_fold", member_names=member_names,
                    validation_rows=validation_rows, calibration_rows=calibration_rows,
                )

    oof = pd.concat(oof_parts, ignore_index=True)
    expected_oof_rows = sum(len(data) for data in data_by_scenario.values()) * 2 * len(member_names)
    validator.require("nested_oof_total_rows", len(oof) == expected_oof_rows == 208980, observed=len(oof), expected=expected_oof_rows)
    for (scenario, design), parts in pooled_parts.items():
        pooled = pd.concat(parts, ignore_index=True)
        expected_ids = set(data_by_scenario[scenario]["model_unit_id"].astype(str))
        for member in member_names:
            subset = pooled[pooled["member"] == member]
            validator.require(
                f"pooled_oof_coverage:{scenario}:{design}:{member}",
                len(subset) == len(expected_ids) and not subset["model_unit_id"].duplicated().any() and set(subset["model_unit_id"].astype(str)) == expected_ids,
            )
        add_metric_rows(
            pooled, scenario=scenario, design=design, outer_fold=None,
            aggregation_scope="pooled_nested_oof", member_names=member_names,
            validation_rows=validation_rows, calibration_rows=calibration_rows,
        )
    oof = oof.sort_values(["scenario_id", "design_id", "outer_fold", "member", "model_unit_id"], kind="mergesort").reset_index(drop=True)
    validation_frame = pd.DataFrame(validation_rows).sort_values(["scenario_id", "design_id", "aggregation_scope", "outer_fold", "member", "threshold_rule"], kind="mergesort", na_position="last").reset_index(drop=True)
    calibration_frame = pd.DataFrame(calibration_rows).sort_values(["scenario_id", "design_id", "aggregation_scope", "outer_fold", "member"], kind="mergesort", na_position="last").reset_index(drop=True)
    sensitivity = validation_frame[(validation_frame["aggregation_scope"] == "pooled_nested_oof") & (validation_frame["member"] == "FINAL_ENSEMBLE")].merge(
        calibration_frame[(calibration_frame["aggregation_scope"] == "pooled_nested_oof") & (calibration_frame["member"] == "FINAL_ENSEMBLE")],
        on=["scenario_id", "design_id", "aggregation_scope", "outer_fold", "member", "branch", "n"],
        how="left",
        suffixes=("", "_calibration"),
        validate="many_to_one",
    )
    validator.require("metric_row_counts", len(validation_frame) == 960 and len(calibration_frame) == 900 and len(sensitivity) == 20, validation_rows=len(validation_frame), calibration_rows=len(calibration_frame), sensitivity_rows=len(sensitivity))
    return oof, validation_frame, calibration_frame, sensitivity


def validate_final_primary(
    *, members: list[dict], contexts: dict, data_by_scenario: dict[str, pd.DataFrame],
    validator: Validator,
):
    scenario = "primary_conflict_excluded"
    design = "scaffold_group_nested_cv"
    data = data_by_scenario[scenario]
    member_names = [item["member"] for item in members] + ["FINAL_ENSEMBLE"]
    frame = pd.read_csv(OUTPUT_ROOT / "final_primary_crossfit_predictions.csv")
    expected_ids = set(data["model_unit_id"].astype(str))
    validate_prediction_frame(frame, data, expected_ids, member_names, "final_primary_crossfit", "raw_crossfit_score")
    validator.require("final_primary_crossfit_rows", len(frame) == 1494 * 15, observed=len(frame))
    ordered_ids = sorted(expected_ids)
    branch_probability = {}
    for branch in ("fingerprint", "graph", "language_model"):
        branch_probability[branch] = np.mean([aligned_probabilities(frame, member["member"], ordered_ids) for member in members if member["branch"] == branch], axis=0)
    recomputed_ensemble = np.mean(list(branch_probability.values()), axis=0)
    observed_ensemble = aligned_probabilities(frame, "FINAL_ENSEMBLE", ordered_ids)
    validator.require("final_primary_branch_balanced_ensemble", np.allclose(recomputed_ensemble, observed_ensemble, rtol=0.0, atol=TOLERANCE))
    ensemble = frame[frame["member"] == "FINAL_ENSEMBLE"].set_index("model_unit_id").loc[ordered_ids]
    threshold = best_mcc_threshold(ensemble["final_label"].to_numpy(dtype=int), ensemble["calibrated_probability"].to_numpy(dtype=float))
    threshold_values = ensemble["threshold"].astype(float).unique()
    validator.require("final_primary_threshold", len(threshold_values) == 1 and compare_float(threshold_values[0], threshold), observed=threshold_values.tolist(), recomputed=threshold)
    validator.require("final_primary_predicted_label", np.array_equal((ensemble["calibrated_probability"].to_numpy(dtype=float) >= threshold).astype(int), ensemble["predicted_label"].astype(int).to_numpy()))

    global_path = METRIC_ROOT / "final_primary_global_selection.json"
    global_payload = json.loads(global_path.read_text(encoding="utf-8"))
    validator.require("global_selection_source", global_payload.get("selection_source") == "primary_inner_metrics_only" and global_payload.get("sensitivity_metrics_used") is False and global_payload.get("outer_oof_performance_used") is False)
    validator.require("global_selection_threshold", compare_float(global_payload["threshold"], threshold))
    global_members = global_payload["members"]
    for member in members:
        ranking = []
        for candidate in member["candidates"]:
            context_auc = []
            context_auprc = []
            for design_id in ("group_random_nested_cv", "scaffold_group_nested_cv"):
                for outer_fold in range(5):
                    found = next(item for item in contexts[(scenario, design_id, outer_fold)]["ranking"][member["member"]] if item["candidate_id"] == candidate["id"])
                    context_auc.append(found["mean_inner_roc_auc"])
                    context_auprc.append(found["mean_inner_auprc"])
            ranking.append({
                "candidate_id": candidate["id"],
                "candidate_ordinal": int(candidate["ordinal"]),
                "mean_roc_auc": float(np.mean(context_auc)),
                "mean_auprc": float(np.mean(context_auprc)),
                "roc_auc_context_std": float(np.std(context_auc, ddof=0)),
            })
        ranking.sort(key=lambda item: (-item["mean_roc_auc"], -item["mean_auprc"], item["roc_auc_context_std"], item["candidate_ordinal"]))
        observed = global_members[member["member"]]
        if observed["selected"]["id"] != ranking[0]["candidate_id"] or [item["candidate_id"] for item in observed["ranking"]] != [item["candidate_id"] for item in ranking]:
            raise IntegrityError(f"Global primary candidate selection mismatch: {member['member']}")
        for left, right in zip(observed["ranking"], ranking):
            for field in ("mean_roc_auc", "mean_auprc", "roc_auc_context_std"):
                if not compare_float(left[field], right[field]):
                    raise IntegrityError(f"Global primary metric mismatch: {member['member']} {field}")

    manifest_path = MODEL_ROOT / "final_primary" / "model_manifest.json"
    model_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validator.require("final_model_manifest_draft", model_manifest.get("status") == "DRAFT_PENDING_TASK00_MODEL_LOCK")
    validator.require("final_model_manifest_members", len(model_manifest.get("members", [])) == EXPECTED_MEMBERS and {item["member"] for item in model_manifest["members"]} == {item["member"] for item in members})
    validator.require("final_model_manifest_threshold", compare_float(model_manifest["threshold"], threshold))
    for entry in model_manifest["members"]:
        expected_candidate = global_members[entry["member"]]["selected"]["id"]
        if entry["candidate_id"] != expected_candidate:
            raise IntegrityError(f"Final model candidate mismatch: {entry['member']}")
        base_path = REPO_ROOT / entry["base_model_path"]
        calibrator_path = REPO_ROOT / entry["calibrator_path"]
        if sha256_file(base_path) != entry["base_model_sha256"] or sha256_file(calibrator_path) != entry["calibrator_sha256"]:
            raise IntegrityError(f"Final model hash mismatch: {entry['member']}")
        if base_path.suffix == ".joblib":
            loaded = joblib.load(base_path)
            if "model" not in loaded or "candidate" not in loaded:
                raise IntegrityError(f"Invalid joblib final model: {entry['member']}")
            if entry["member"].startswith("RandomForest") and loaded["model"].__class__.__module__ + "." + loaded["model"].__class__.__name__ != "sklearn.ensemble._forest.RandomForestClassifier":
                raise IntegrityError(f"Invalid final RF class: {entry['member']}")
            if entry["member"].startswith("RBF_SVM") and (loaded["model"].__class__.__name__ != "SVC" or loaded["model"].probability is not False):
                raise IntegrityError(f"Invalid final SVC class/probability: {entry['member']}")
        elif base_path.suffix == ".json":
            booster = xgboost.Booster()
            booster.load_model(base_path)
        elif base_path.suffix == ".txt":
            lightgbm.Booster(model_file=str(base_path))
        elif base_path.suffix == ".pt":
            loaded = torch.load(base_path, map_location="cpu", weights_only=False)
            if "state_dict" not in loaded:
                raise IntegrityError(f"Invalid PyTorch final model: {entry['member']}")
        else:
            raise IntegrityError(f"Unknown final model format: {base_path}")
        calibrator = torch.load(calibrator_path, map_location="cpu", weights_only=False)
        evidence = calibrator.get("evidence", {})
        if "state_dict" not in calibrator or evidence.get("backend") != "native_pytorch_cuda_two_parameter_platt" or evidence.get("device") != "cuda:0":
            raise IntegrityError(f"Invalid final calibrator: {entry['member']}")
    validator.require("final_model_files_loadable_and_hashed", True, members=EXPECTED_MEMBERS, artifacts=28)
    return frame, global_payload, model_manifest, threshold


def validate_checkpoints(validator: Validator):
    paths = sorted(CHECKPOINT_ROOT.rglob("*.json"))
    validator.require("checkpoint_count", len(paths) == 51, observed=len(paths))
    checked_outputs = 0
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "PASS":
            raise IntegrityError(f"Checkpoint is not PASS: {relative(path)}")
        for output in payload.get("outputs", []):
            target = REPO_ROOT / output["path"]
            if not target.is_file() or sha256_file(target) != output["sha256"]:
                raise IntegrityError(f"Checkpoint output hash mismatch: {output['path']}")
            checked_outputs += 1
    validator.require("checkpoint_output_hashes", checked_outputs == 153, observed=checked_outputs)
    return paths


def validate_feature_evidence(protocol: dict, data_by_scenario: dict[str, pd.DataFrame], validator: Validator):
    metadata_paths = sorted(FEATURE_ROOT.rglob("metadata.json"))
    fingerprint_metadata = sorted(FEATURE_ROOT.rglob("fingerprints.metadata.json"))
    validator.require("language_embedding_metadata_count", len(metadata_paths) == 10, observed=len(metadata_paths))
    validator.require("fingerprint_metadata_count", len(fingerprint_metadata) == 5, observed=len(fingerprint_metadata))
    specs = {item["scenario_id"]: item for item in protocol["inputs"]}
    for path in metadata_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        scenario = path.relative_to(FEATURE_ROOT).parts[0]
        data = data_by_scenario[scenario]
        if payload.get("input_sha256") != specs[scenario]["sha256"] or payload.get("device") != "cuda:0" or payload.get("cpu_fallback_used") is not False or payload.get("local_files_only") is not True:
            raise IntegrityError(f"Language embedding evidence mismatch: {relative(path)}")
        if int(payload.get("row_count", -1)) != int(specs[scenario]["rows"]):
            raise IntegrityError(f"Language embedding row mismatch: {relative(path)}")
        with np.load(path.parent / "embeddings.npz", allow_pickle=False) as bundle:
            cached_ids = bundle["model_unit_ids"].astype(str).tolist()
            embeddings = np.ascontiguousarray(bundle["embeddings"], dtype=np.float32)
        expected_ids = data["model_unit_id"].astype(str).tolist()
        if cached_ids != expected_ids or payload.get("ordered_model_unit_id_sha256") != hash_ids(expected_ids):
            raise IntegrityError(f"Language embedding unit order mismatch: {relative(path)}")
        if payload.get("ordered_smiles_sha256") != hash_ids(data["canonical_isomeric_smiles"].astype(str)):
            raise IntegrityError(f"Language embedding SMILES order mismatch: {relative(path)}")
        if list(embeddings.shape) != [len(data), int(payload.get("embedding_dimension", -1))]:
            raise IntegrityError(f"Language embedding shape mismatch: {relative(path)}")
        if payload.get("embedding_array_sha256") != array_hash(embeddings):
            raise IntegrityError(f"Language embedding array hash mismatch: {relative(path)}")

    for path in fingerprint_metadata:
        payload = json.loads(path.read_text(encoding="utf-8"))
        scenario = path.parent.name
        data = data_by_scenario[scenario]
        expected_ids = data["model_unit_id"].astype(str).tolist()
        if payload.get("input_sha256") != specs[scenario]["sha256"] or payload.get("ordered_model_unit_id_sha256") != hash_ids(expected_ids):
            raise IntegrityError(f"Fingerprint evidence mismatch: {relative(path)}")
        if payload.get("model_fit_executed") is not False:
            raise IntegrityError(f"Fingerprint metadata unexpectedly records a fit: {relative(path)}")
        with np.load(path.parent / "fingerprints.npz", allow_pickle=False) as bundle:
            matrices = {name: np.ascontiguousarray(bundle[name], dtype=np.float32) for name in ("ECFP4", "MACCS")}
        for name, matrix in matrices.items():
            if payload.get("shape", {}).get(name) != list(matrix.shape) or matrix.shape[0] != len(data):
                raise IntegrityError(f"Fingerprint shape mismatch: {relative(path)} {name}")
            if payload.get("array_sha256", {}).get(name) != array_hash(matrix):
                raise IntegrityError(f"Fingerprint array hash mismatch: {relative(path)} {name}")
    validator.require("registered_feature_cache_content_hashes", True, language_metadata_files=10, fingerprint_metadata_files=5)
    validator.require("language_features_cuda_only", True, metadata_files=10)


def environment_payload(progress: dict, device_counts: Counter) -> dict:
    command = ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"]
    nvidia = subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()
    return {
        "artifact": "TASK02_E001_FULL_TRAINING_ENVIRONMENT_CANDIDATE",
        "status": "VALIDATED_PENDING_TASK00_MODEL_LOCK_REVIEW",
        "captured_at_utc": now_utc(),
        "compute_policy": "HYBRID_RESTRICTED",
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
            "rdkit": rdkit.__version__, "torch": torch.__version__, "xgboost": xgboost.__version__,
            "lightgbm": lightgbm.__version__, "psutil": psutil.__version__,
        },
        "cuda": {
            "available": bool(torch.cuda.is_available()),
            "torch_cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "compute_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
            "nvidia_smi": nvidia,
        },
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "go_full_training_sha256": EXPECTED_GO_FULL_SHA256,
        "execution_supplement_sha256": EXPECTED_SUPPLEMENT_SHA256,
        "execution_plan_sha256": EXPECTED_PLAN_SHA256,
        "split_manifest_index_sha256": EXPECTED_SPLIT_INDEX_SHA256,
        "formal_fit_jobs_attempted": int(progress["formal_fit_jobs_attempted"]),
        "formal_fit_jobs_completed": int(progress["formal_fit_jobs_completed"]),
        "device_evidence_counts": dict(sorted(device_counts.items())),
        "notes": [
            "Package versions are captured after the completed E001 process in the same isolated environment.",
            "Per-job device evidence is independently validated from all 26,498 job-result records.",
            "This candidate becomes an immutable environment lock only if Task 00 approves MODEL_LOCK.json.",
        ],
    }


def authoritative_scientific_paths() -> list[Path]:
    paths = [
        PROTOCOL_PATH, GO_FULL_PATH, SUPPLEMENT_PATH, PLAN_PATH, SPLIT_INDEX_PATH,
        EVENTS_PATH, PAUSE_RECORD_PATH, PROGRESS_PATH,
        OOF_PATH, VALIDATION_METRICS_PATH, CALIBRATION_METRICS_PATH,
        SENSITIVITY_METRICS_PATH, ENVIRONMENT_PATH, JOB_HASH_MANIFEST_PATH,
        VALIDATION_REPORT_PATH, VALIDATION_LOG_PATH,
        TASK_DIR / "src" / "gpu_models.py",
        TASK_DIR / "src" / "graph_models.py",
        TASK_DIR / "src" / "hybrid_models.py",
        TASK_DIR / "src" / "lm_embeddings.py",
        TASK_DIR / "src" / "runtime_guards.py",
        TASK_DIR / "src" / "split_protocol.py",
        TASK_DIR / "scripts" / "run_full_training_e001.py",
        TASK_DIR / "scripts" / "generate_and_validate_full_execution_plan.py",
        SCRIPT_PATH,
    ]
    paths.extend(REPO_ROOT / row["manifest_path"] for row in pd.read_csv(SPLIT_INDEX_PATH).to_dict("records"))
    paths.extend(sorted(path for path in FEATURE_ROOT.rglob("*") if path.is_file()))
    paths.extend(sorted(path for path in CHECKPOINT_ROOT.rglob("*.json") if path.is_file()))
    paths.extend(sorted(path for path in OUTPUT_ROOT.rglob("*") if path.is_file()))
    paths.extend(sorted(path for path in METRIC_ROOT.rglob("*.json") if path.is_file()))
    paths.extend(sorted(path for path in MODEL_ROOT.rglob("*") if path.is_file()))
    unique = {path.resolve(): path for path in paths}
    return [unique[key] for key in sorted(unique, key=lambda item: item.as_posix())]


def build_output_manifest(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        if not path.is_file():
            raise IntegrityError(f"Authoritative output is missing: {path}")
        rel = relative(path)
        if "/models/" in f"/{rel}":
            role = "fitted_model_or_calibrator"
        elif "/predictions/" in f"/{rel}" or rel.endswith("oof_predictions.csv"):
            role = "prediction"
        elif "/metrics/" in f"/{rel}" or rel.endswith("_metrics.csv"):
            role = "metric_or_selection"
        elif "/splits/" in f"/{rel}":
            role = "immutable_split"
        elif "/features/" in f"/{rel}":
            role = "registered_feature_cache"
        elif "/runtime/context_checkpoints_e001/" in f"/{rel}":
            role = "completion_checkpoint"
        elif rel.endswith("full_training_fit_events_e001.jsonl") or rel.endswith("full_training_job_result_hashes_e001.csv"):
            role = "fit_audit_ledger"
        elif "/src/" in f"/{rel}" or "/scripts/" in f"/{rel}":
            role = "implementation_source"
        else:
            role = "governance_or_validation_evidence"
        rows.append({"path": rel, "role": role, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return pd.DataFrame(rows).sort_values("path", kind="mergesort").reset_index(drop=True)


def primary_summary(validation_frame: pd.DataFrame, calibration_frame: pd.DataFrame) -> list[dict]:
    validation = validation_frame[
        (validation_frame["scenario_id"] == "primary_conflict_excluded")
        & (validation_frame["aggregation_scope"] == "pooled_nested_oof")
        & (validation_frame["member"] == "FINAL_ENSEMBLE")
        & (validation_frame["threshold_rule"] == "outer_train_MCC_fold_specific")
    ]
    calibration = calibration_frame[
        (calibration_frame["scenario_id"] == "primary_conflict_excluded")
        & (calibration_frame["aggregation_scope"] == "pooled_nested_oof")
        & (calibration_frame["member"] == "FINAL_ENSEMBLE")
    ][["design_id", "brier_score", "ece_equal_width_10", "calibration_intercept", "calibration_slope"]]
    merged = validation.merge(calibration, on="design_id", how="left", validate="one_to_one")
    fields = ["design_id", "n", "roc_auc", "auprc", "mcc", "sensitivity", "specificity", "brier_score", "ece_equal_width_10", "calibration_intercept", "calibration_slope"]
    return merged[fields].to_dict("records")


def assemble_governance_package(
    *, progress: dict, validator: Validator, validation_frame: pd.DataFrame,
    calibration_frame: pd.DataFrame, global_payload: dict, model_manifest: dict,
    threshold: float, event_summary: dict, device_counts: Counter, audit_counts: Counter,
    output_manifest_sha256: str,
) -> None:
    generated = now_utc()
    summary = primary_summary(validation_frame, calibration_frame)
    environment_sha = sha256_file(ENVIRONMENT_PATH)
    oof_sha = sha256_file(OOF_PATH)
    validation_sha = sha256_file(VALIDATION_METRICS_PATH)
    calibration_sha = sha256_file(CALIBRATION_METRICS_PATH)
    sensitivity_sha = sha256_file(SENSITIVITY_METRICS_PATH)
    model_manifest_path = MODEL_ROOT / "final_primary" / "model_manifest.json"
    selected_models = [
        {
            "member": entry["member"], "candidate_id": entry["candidate_id"],
            "base_model_path": entry["base_model_path"], "base_model_sha256": entry["base_model_sha256"],
            "calibrator_path": entry["calibrator_path"], "calibrator_sha256": entry["calibrator_sha256"],
        }
        for entry in model_manifest["members"]
    ]

    old_lock = json.loads(MODEL_LOCK_PATH.read_text(encoding="utf-8"))
    lock_candidate = {
        **old_lock,
        "status": "DRAFT",
        "validation_state": "VALIDATED_PENDING_TASK00_MODEL_LOCK_REVIEW",
        "locked": False,
        "usable_by_downstream_tasks": False,
        "lock_authority": "Task 00 only",
        "task00_review_status": "PENDING_TASK00_MODEL_LOCK_REVIEW",
        "training_executed": True,
        "formal_full_training_executed": True,
        "formal_fit_jobs_planned": PLANNED_JOBS,
        "formal_fit_jobs_attempted": int(progress["formal_fit_jobs_attempted"]),
        "formal_fit_jobs_completed": int(progress["formal_fit_jobs_completed"]),
        "training_completed_at_utc": progress["last_completed_fit_at_utc"],
        "task02_validation_completed_at_utc": generated,
        "environment_lock_path": relative(ENVIRONMENT_PATH),
        "environment_lock_sha256": environment_sha,
        "selected_model_specification": {
            "member_count": 14,
            "ensemble": "equal mean within fingerprint, graph and language_model branches; equal mean across three branches",
            "global_selection_source": "primary inner-validation metrics only, equally aggregated across five outer contexts and two registered designs",
            "model_manifest_path": relative(model_manifest_path),
            "model_manifest_sha256": sha256_file(model_manifest_path),
            "members": selected_models,
        },
        "calibration_specification": {
            "method": "native PyTorch CUDA two-parameter Platt",
            "source": "member-specific scaffold-group five-fold cross-fitted raw scores on all 1494 locked primary model units",
            "outer_or_external_test_used": False,
            "calibrator_count": 14,
        },
        "threshold_specification": {
            "value": float(threshold),
            "rule": "maximize MCC; ties resolved by closest to 0.5 then lower numeric threshold",
            "source": "final-primary scaffold-group cross-fitted branch-balanced ensemble probabilities",
            "external_validation_used": False,
        },
        "oof_predictions_path": relative(OOF_PATH),
        "oof_predictions_sha256": oof_sha,
        "validation_metrics_path": relative(VALIDATION_METRICS_PATH),
        "validation_metrics_sha256": validation_sha,
        "calibration_metrics_path": relative(CALIBRATION_METRICS_PATH),
        "calibration_metrics_sha256": calibration_sha,
        "label_sensitivity_metrics_path": relative(SENSITIVITY_METRICS_PATH),
        "label_sensitivity_metrics_sha256": sensitivity_sha,
        "full_training_integrity_report_path": relative(VALIDATION_REPORT_PATH),
        "full_training_integrity_report_sha256": sha256_file(VALIDATION_REPORT_PATH),
        "full_training_output_manifest_path": relative(OUTPUT_MANIFEST_PATH),
        "full_training_output_manifest_sha256": output_manifest_sha256,
        "primary_nested_oof_summary": summary,
        "known_limitations": [
            "This is a Task 02 validated candidate package, not a Task 00 model lock; locked remains false and downstream use remains prohibited.",
            "No external validation has been performed, and external data did not influence selection, calibration, ensemble weights or threshold.",
            "The endpoint is an evidence-integrated potential carcinogenicity signal, not definitive human carcinogenicity or exposure risk.",
            "Nested OOF results are the internal performance estimates; final-primary cross-fitted predictions are reserved for deployable calibration and threshold derivation and are not substituted for nested OOF performance.",
            "ECE uses a declared deterministic 10-bin equal-width calculation. No post-result bootstrap confidence-interval scheme was introduced because v4/E001 did not preregister one.",
            "One user-requested reboot interrupted E001-J11471 attempt 1; attempt 2 reused the immutable plan row and seed and completed successfully. Both attempts count toward the 27,300 ceiling.",
            "All 4,612 XGBoost result records captured device evidence immediately after fit, so prediction_device remains not_yet_predicted. The executed prediction implementation requires CuPy input/output, rejects any prediction warning, and produced finite scores; zero CPU fallback was observed. This audit-trace timing limitation is disclosed to Task 00 without rewriting historical result records.",
        ],
    }
    atomic_json(MODEL_LOCK_PATH, lock_candidate)

    status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    status.update({
        "status": "UNDER_REVIEW",
        "phase": "VALIDATED_PENDING_TASK00_MODEL_LOCK_REVIEW",
        "updated_at_utc": generated,
        "formal_full_training_started": True,
        "formal_full_training_completed": True,
        "formal_fit_jobs_attempted": int(progress["formal_fit_jobs_attempted"]),
        "formal_fit_jobs_completed": int(progress["formal_fit_jobs_completed"]),
        "next_gate": "TASK00_MODEL_LOCK_REVIEW",
        "model_lock_state": "DRAFT_VALIDATED_NOT_LOCKED",
        "task02_validation": {
            "status": "PASS",
            "state": "VALIDATED_PENDING_TASK00_MODEL_LOCK_REVIEW",
            "report_path": relative(VALIDATION_REPORT_PATH),
            "report_sha256": sha256_file(VALIDATION_REPORT_PATH),
            "output_manifest_path": relative(OUTPUT_MANIFEST_PATH),
            "output_manifest_sha256": output_manifest_sha256,
            "request_path": relative(REQUEST_PATH),
        },
    })
    status["execution_readiness"].update({
        "status": "FULL_TRAINING_COMPLETED_TASK02_VALIDATION_PASS",
        "reason": "All 26,498 registered E001 jobs completed; 26,499 attempts include the single registered user-reboot interruption; Task 02 integrity validation passed.",
        "formal_fit_jobs_attempted": int(progress["formal_fit_jobs_attempted"]),
        "formal_fit_jobs_completed": int(progress["formal_fit_jobs_completed"]),
    })
    atomic_json(STATUS_PATH, status)

    summary_lines = []
    for row in summary:
        summary_lines.append(
            f"| {row['design_id']} | {row['n']} | {row['roc_auc']:.6f} | {row['auprc']:.6f} | {row['mcc']:.6f} | {row['sensitivity']:.6f} | {row['specificity']:.6f} | {row['brier_score']:.6f} | {row['ece_equal_width_10']:.6f} |"
        )
    handoff = f"""# Task 02 to Task 00 Handoff — E001 Model-Lock Review

Status: `VALIDATED_PENDING_TASK00_MODEL_LOCK_REVIEW`  
Generated at UTC: `{generated}`

E001 completed all {PLANNED_JOBS:,} registered jobs in {int(progress['formal_fit_jobs_attempted']):,} attempts. The one extra attempt is the documented user-reboot interruption of `E001-J11471`; its second attempt used the unchanged registered plan row and seed. Task 02 integrity validation passed. This handoff is a review submission, not a model lock.

## Primary nested OOF summary

Classification values use each outer fold's threshold selected exclusively from outer-training cross-fitted probabilities.

| design | n | ROC-AUC | AUPRC | MCC | sensitivity | specificity | Brier | ECE-10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(summary_lines)}

## Evidence to review first

1. `{relative(REQUEST_PATH)}`
2. `{relative(VALIDATION_REPORT_PATH)}`
3. `{relative(OUTPUT_MANIFEST_PATH)}`
4. `{relative(OOF_PATH)}`
5. `{relative(VALIDATION_METRICS_PATH)}`
6. `{relative(CALIBRATION_METRICS_PATH)}`
7. `{relative(SENSITIVITY_METRICS_PATH)}`
8. `{relative(model_manifest_path)}`
9. `{relative(ENVIRONMENT_PATH)}`
10. `{relative(MODEL_LOCK_PATH)}`

## Validation scope

- Exact protocol, Task 00 approval, E001, plan, Task 01 and split fingerprints were rechecked.
- All result JSONs and event rows were reconciled with the immutable 26,498-row plan.
- All 8,224 CPU jobs were restricted to the two sklearn whitelist classes; all 18,274 remaining jobs carried required CUDA evidence.
- All 4,612 XGBoost records contain a pre-prediction evidence snapshot (`prediction_device=not_yet_predicted`). The prediction implementation enforces CuPy/CUDA output and rejects warnings; no CPU fallback was observed, but Task 00 must review this disclosed audit-trace limitation.
- Five scenarios, two designs, five outer folds, five inner folds and all 14 members passed coverage and leakage checks.
- All 700 context-specific candidate selections and all 14 global-primary selections were independently recomputed from inner-fold metrics in the registered order.
- Nested outer-test OOF coverage, outer-training cross-fit coverage, calibration probabilities, branch-balanced ensemble probabilities and MCC thresholds were independently reproduced.
- Fourteen final base models and fourteen final CUDA Platt calibrators were hash-verified and load-tested.

## Authority boundary

`MODEL_LOCK.json` remains `status=DRAFT`, `locked=false`, and `usable_by_downstream_tasks=false`. External validation, MFH rescreening, manuscript editing, docking and dynamics remain prohibited. Only Task 00 may approve and lock the model or authorize Tasks 03/04.
"""
    atomic_text(HANDOFF_PATH, handoff)

    evidence_paths = [
        VALIDATION_REPORT_PATH, OUTPUT_MANIFEST_PATH, OOF_PATH, VALIDATION_METRICS_PATH,
        CALIBRATION_METRICS_PATH, SENSITIVITY_METRICS_PATH, ENVIRONMENT_PATH,
        model_manifest_path, MODEL_LOCK_PATH, STATUS_PATH, HANDOFF_PATH,
    ]
    request = {
        "artifact": "TASK02_E001_MODEL_LOCK_REVIEW_REQUEST",
        "status": "PENDING_TASK00_REVIEW",
        "submitted_at_utc": generated,
        "submitted_by": "Task 02",
        "requested_action": "Task 00 independently review the E001 validated candidate package and either lock MODEL-001 or return specific remediation requirements.",
        "request_is_approval": False,
        "task02_may_self_lock": False,
        "required_task02_completion_state": "VALIDATED_PENDING_TASK00_MODEL_LOCK_REVIEW",
        "training_accounting": {
            "planned_jobs": PLANNED_JOBS,
            "attempted_jobs": int(progress["formal_fit_jobs_attempted"]),
            "completed_jobs": int(progress["formal_fit_jobs_completed"]),
            "hard_ceiling": HARD_CEILING,
            "documented_extra_attempt": "E001-J11471 attempt 1 interrupted by user-requested reboot",
            "elapsed_hours": float(progress["elapsed_hours"]),
        },
        "immutable_hashes": {
            "protocol_v4": EXPECTED_PROTOCOL_SHA256,
            "go_full_training_v2": EXPECTED_GO_FULL_SHA256,
            "execution_supplement_e001": EXPECTED_SUPPLEMENT_SHA256,
            "execution_plan": EXPECTED_PLAN_SHA256,
            "split_manifest_index": EXPECTED_SPLIT_INDEX_SHA256,
        },
        "validation": {
            "status": "PASS",
            "check_count": len(validator.checks),
            "event_summary": event_summary,
            "device_evidence_counts": dict(sorted(device_counts.items())),
            "audit_findings": {
                "xgboost_prediction_device_snapshot_not_yet_predicted": int(audit_counts["xgboost_prediction_device_snapshot_not_yet_predicted"]),
                "interpretation": "Historical evidence was captured after CUDA fit but before CuPy/CUDA prediction. Prediction source guards and zero warnings provide indirect evidence; records were not rewritten and no rerun was performed.",
                "task00_review_required": True,
            },
            "primary_nested_oof_summary": summary,
        },
        "evidence": [
            {"path": relative(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in evidence_paths
        ],
        "model_lock_candidate": {
            "path": relative(MODEL_LOCK_PATH),
            "sha256": sha256_file(MODEL_LOCK_PATH),
            "status": "DRAFT",
            "locked": False,
            "usable_by_downstream_tasks": False,
        },
        "authorization_state": {
            "external_validation_authorized": False,
            "mfh_rescreening_authorized": False,
            "manuscript_revision_authorized": False,
            "docking_or_dynamics_authorized": False,
            "task01_mutation_authorized": False,
            "task02_model_lock_authorized": False,
        },
        "task00_review_questions": [
            "Do all immutable hashes, job/event accounting and backend-policy checks pass independently?",
            "Do nested OOF, calibration, ensemble and threshold checks establish no outer-test leakage?",
            "Are the selected 14 final members, model/calibrator hashes and environment candidate acceptable for MODEL-001 lock?",
            "Should Task 00 authorize Tasks 03 and 04 only after issuing a separate LOCKED model record?",
        ],
    }
    atomic_json(REQUEST_PATH, request)
    request_md = f"""# Task 00 Model-Lock Review Request — E001

Status: `PENDING_TASK00_REVIEW`  
Submitted by: Task 02  
Submitted at UTC: `{generated}`

Task 02 requests an independent model-lock review of the completed E001 package. This request is not an approval and does not authorize downstream use.

- Planned/completed jobs: `{PLANNED_JOBS:,} / {PLANNED_JOBS:,}`
- Attempted jobs: `{int(progress['formal_fit_jobs_attempted']):,} / {HARD_CEILING:,}`
- Integrity validation: `PASS`
- Candidate model lock: `DRAFT`, `locked=false`, `usable_by_downstream_tasks=false`
- Disclosed audit limitation: 4,612 XGBoost device-evidence snapshots were captured before prediction; historical records were not rewritten
- External validation/MFH rescreening/manuscript/docking/dynamics: not performed and not authorized

Machine-readable request: `{relative(REQUEST_PATH)}`  
Integrity report: `{relative(VALIDATION_REPORT_PATH)}`  
Output manifest: `{relative(OUTPUT_MANIFEST_PATH)}`  
Task 02 handoff: `{relative(HANDOFF_PATH)}`

Only Task 00 may issue the final lock decision and update central governance files.
"""
    atomic_text(REQUEST_MD_PATH, request_md)


def run_validation() -> int:
    validator = Validator()
    started = now_utc()
    try:
        progress = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
        validator.require(
            "runner_completed_before_validation",
            progress.get("state") == "COMPLETED_PENDING_TASK02_VALIDATION"
            and int(progress.get("formal_fit_jobs_completed", -1)) == PLANNED_JOBS
            and int(progress.get("formal_fit_jobs_attempted", -1)) == EXPECTED_ATTEMPTS,
            state=progress.get("state"), completed=progress.get("formal_fit_jobs_completed"), attempted=progress.get("formal_fit_jobs_attempted"),
        )
        training_processes = active_training_processes()
        validator.require("no_training_process_running", not training_processes, observed=training_processes)
        protocol, _, supplement, governance_hashes = load_governance(validator)
        _, data_by_scenario, split_by_context, split_validations = load_splits(protocol, validator)
        plan, plan_map, members, stage_counts, backend_counts = validate_plan(protocol, supplement, validator)
        event_summary = validate_events(plan_map, validator)
        inner_metrics, device_counts, audit_counts, job_hash_frame, folds = validate_results(
            plan=plan, plan_map=plan_map, data_by_scenario=data_by_scenario,
            split_by_context=split_by_context, validator=validator,
        )
        contexts = validate_candidate_selection(members=members, inner_metrics=inner_metrics, protocol=protocol, validator=validator)
        oof, validation_frame, calibration_frame, sensitivity_frame = validate_predictions_and_metrics(
            protocol=protocol, members=members, contexts=contexts,
            data_by_scenario=data_by_scenario, folds=folds, validator=validator,
        )
        _, global_payload, model_manifest, threshold = validate_final_primary(
            members=members, contexts=contexts, data_by_scenario=data_by_scenario, validator=validator,
        )
        checkpoint_paths = validate_checkpoints(validator)
        validate_feature_evidence(protocol, data_by_scenario, validator)
        validator.require("fit_ceiling_respected", int(progress["formal_fit_jobs_attempted"]) <= HARD_CEILING, attempted=int(progress["formal_fit_jobs_attempted"]), ceiling=HARD_CEILING)
        validator.require("runtime_limit_respected", float(progress["elapsed_hours"]) < 60.0, elapsed_hours=float(progress["elapsed_hours"]))
        validator.require("storage_limit_respected", float(progress["task02_storage_gib"]) < 60.0, storage_gib=float(progress["task02_storage_gib"]))
        current_lock = json.loads(MODEL_LOCK_PATH.read_text(encoding="utf-8"))
        validator.require("model_not_self_locked_before_package", current_lock.get("locked") is False and current_lock.get("usable_by_downstream_tasks") is False)

        atomic_csv(OOF_PATH, oof)
        atomic_csv(VALIDATION_METRICS_PATH, validation_frame)
        atomic_csv(CALIBRATION_METRICS_PATH, calibration_frame)
        atomic_csv(SENSITIVITY_METRICS_PATH, sensitivity_frame)
        atomic_csv(JOB_HASH_MANIFEST_PATH, job_hash_frame)
        atomic_json(ENVIRONMENT_PATH, environment_payload(progress, device_counts))

        report = {
            "artifact": "TASK02_E001_FULL_TRAINING_INTEGRITY_VALIDATION",
            "status": "PASS",
            "operation": "POST_TRAINING_VALIDATION_NO_PREDICTIVE_MODEL_FIT",
            "started_at_utc": started,
            "completed_at_utc": now_utc(),
            "required_completion_state": "VALIDATED_PENDING_TASK00_MODEL_LOCK_REVIEW",
            "governance_hashes": governance_hashes,
            "training_accounting": {
                "planned_jobs": PLANNED_JOBS,
                "attempted_jobs": int(progress["formal_fit_jobs_attempted"]),
                "completed_jobs": int(progress["formal_fit_jobs_completed"]),
                "hard_ceiling": HARD_CEILING,
                "elapsed_hours": float(progress["elapsed_hours"]),
                "storage_gib": float(progress["task02_storage_gib"]),
                "stage_counts": stage_counts,
                "backend_plan_counts": backend_counts,
                "event_summary": event_summary,
            },
            "coverage": {
                "scenarios": 5, "designs": 2, "outer_folds_per_design": 5,
                "inner_folds_per_outer": 5, "base_members": 14,
                "candidate_selection_contexts": 50,
                "candidate_member_contexts": 700,
                "nested_oof_rows_all_members": int(len(oof)),
                "final_primary_crossfit_rows_all_members": 1494 * 15,
                "checkpoint_count": len(checkpoint_paths),
            },
            "device_evidence_counts": dict(sorted(device_counts.items())),
            "audit_findings": {
                "xgboost_prediction_device_snapshot_not_yet_predicted": int(audit_counts["xgboost_prediction_device_snapshot_not_yet_predicted"]),
                "source_guard_evidence": "XGBoost predict_raw converts input to CuPy, rejects all prediction warnings, requires a CuPy prediction object and finite scores.",
                "interpretation": "The result evidence dictionary was serialized from a pre-prediction snapshot. This is an audit-trace timing limitation, not an observed CPU fallback. Historical result records were not rewritten and no rerun was performed.",
                "task00_review_required": True,
            },
            "split_validations": split_validations,
            "metric_method": {
                "performance_source": "nested outer-test OOF only",
                "classification_thresholds": "registered outer-training MCC threshold and fixed 0.5",
                "ece": "10 equal-width bins on [0,1]",
                "calibration_slope_intercept": "metric-only Newton estimation on cuda:0; not saved or used for any model decision",
                "confidence_intervals": "not added because v4/E001 did not preregister a bootstrap design",
            },
            "primary_nested_oof_summary": primary_summary(validation_frame, calibration_frame),
            "checks": validator.checks,
            "scope_compliance": {
                "external_validation_used": False,
                "mfh_rescreening_performed": False,
                "manuscript_modified": False,
                "docking_or_dynamics_performed": False,
                "task01_files_modified": False,
                "model_lock_set_locked_by_task02": False,
            },
            "generated_outputs": {
                relative(OOF_PATH): sha256_file(OOF_PATH),
                relative(VALIDATION_METRICS_PATH): sha256_file(VALIDATION_METRICS_PATH),
                relative(CALIBRATION_METRICS_PATH): sha256_file(CALIBRATION_METRICS_PATH),
                relative(SENSITIVITY_METRICS_PATH): sha256_file(SENSITIVITY_METRICS_PATH),
                relative(ENVIRONMENT_PATH): sha256_file(ENVIRONMENT_PATH),
                relative(JOB_HASH_MANIFEST_PATH): sha256_file(JOB_HASH_MANIFEST_PATH),
            },
            "known_process_limitation": "E001 started final-primary fitting after its built-in per-context checkpoints, but before a separate cross-scenario post-training validator existed. This report independently validates the completed package and does not backdate a separate pre-final gate.",
        }
        atomic_json(VALIDATION_REPORT_PATH, report)
        atomic_text(VALIDATION_LOG_PATH, json.dumps({
            "timestamp_utc": now_utc(),
            "event": "TASK02_E001_INTEGRITY_VALIDATION_COMPLETED",
            "status": "PASS",
            "checks": len(validator.checks),
            "planned_jobs": PLANNED_JOBS,
            "attempted_jobs": int(progress["formal_fit_jobs_attempted"]),
            "completed_jobs": int(progress["formal_fit_jobs_completed"]),
            "model_lock_locked": False,
        }, ensure_ascii=False, sort_keys=True) + "\n")
        output_manifest = build_output_manifest(authoritative_scientific_paths())
        atomic_csv(OUTPUT_MANIFEST_PATH, output_manifest)
        output_manifest_sha = sha256_file(OUTPUT_MANIFEST_PATH)
        assemble_governance_package(
            progress=progress, validator=validator, validation_frame=validation_frame,
            calibration_frame=calibration_frame, global_payload=global_payload,
            model_manifest=model_manifest, threshold=threshold,
            event_summary=event_summary, device_counts=device_counts, audit_counts=audit_counts,
            output_manifest_sha256=output_manifest_sha,
        )
        print(json.dumps({
            "status": "PASS",
            "state": "VALIDATED_PENDING_TASK00_MODEL_LOCK_REVIEW",
            "planned_jobs": PLANNED_JOBS,
            "attempted_jobs": int(progress["formal_fit_jobs_attempted"]),
            "completed_jobs": int(progress["formal_fit_jobs_completed"]),
            "check_count": len(validator.checks),
            "validation_report": relative(VALIDATION_REPORT_PATH),
            "output_manifest": relative(OUTPUT_MANIFEST_PATH),
            "task00_request": relative(REQUEST_PATH),
            "model_lock_locked": False,
        }, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "artifact": "TASK02_E001_FULL_TRAINING_INTEGRITY_VALIDATION",
            "status": "FAIL",
            "operation": "POST_TRAINING_VALIDATION_NO_PREDICTIVE_MODEL_FIT",
            "started_at_utc": started,
            "failed_at_utc": now_utc(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "checks": validator.checks,
            "model_lock_modified_to_locked": False,
        }
        atomic_json(VALIDATION_REPORT_PATH, failure)
        atomic_text(VALIDATION_LOG_PATH, json.dumps({
            "timestamp_utc": now_utc(),
            "event": "TASK02_E001_INTEGRITY_VALIDATION_FAILED",
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "model_lock_modified_to_locked": False,
        }, ensure_ascii=False, sort_keys=True) + "\n")
        try:
            status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
            status.update({
                "status": "REVISION_REQUIRED",
                "phase": "FULL_TRAINING_E001_VALIDATION_FAILED",
                "updated_at_utc": now_utc(),
                "next_gate": "TASK02_REMEDIATION_BEFORE_TASK00",
                "task02_validation": {
                    "status": "FAIL",
                    "report_path": relative(VALIDATION_REPORT_PATH),
                    "report_sha256": sha256_file(VALIDATION_REPORT_PATH),
                    "model_lock_modified_to_locked": False,
                },
            })
            atomic_json(STATUS_PATH, status)
        except Exception:
            pass
        print(json.dumps(failure, indent=2, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 1


def verify_package() -> int:
    validator = Validator()
    try:
        manifest = pd.read_csv(OUTPUT_MANIFEST_PATH, dtype=str, keep_default_na=False)
        validator.require("output_manifest_unique_paths", manifest["path"].is_unique)
        for row in manifest.itertuples(index=False):
            path = REPO_ROOT / row.path
            if not path.is_file() or path.stat().st_size != int(row.size_bytes) or sha256_file(path) != row.sha256:
                raise IntegrityError(f"Output manifest verification failed: {row.path}")
        job_hash_manifest = pd.read_csv(JOB_HASH_MANIFEST_PATH, dtype=str, keep_default_na=False)
        validator.require("job_hash_manifest_exact_count", len(job_hash_manifest) == PLANNED_JOBS, observed=int(len(job_hash_manifest)))
        validator.require("job_hash_manifest_unique_jobs_and_paths", job_hash_manifest["job_id"].is_unique and job_hash_manifest["path"].is_unique)
        for row in job_hash_manifest.itertuples(index=False):
            path = REPO_ROOT / row.path
            if path.stem != row.job_id or not path.is_file() or path.stat().st_size != int(row.size_bytes) or sha256_file(path) != row.sha256:
                raise IntegrityError(f"Raw job-result verification failed: {row.path}")
        validator.require("raw_job_results_reverified", True, files=PLANNED_JOBS)
        report = json.loads(VALIDATION_REPORT_PATH.read_text(encoding="utf-8"))
        lock_candidate = json.loads(MODEL_LOCK_PATH.read_text(encoding="utf-8"))
        status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        request = json.loads(REQUEST_PATH.read_text(encoding="utf-8"))
        validator.require("validation_report_pass", report.get("status") == "PASS")
        validator.require("model_lock_remains_draft", lock_candidate.get("status") == "DRAFT" and lock_candidate.get("validation_state") == "VALIDATED_PENDING_TASK00_MODEL_LOCK_REVIEW" and lock_candidate.get("locked") is False and lock_candidate.get("usable_by_downstream_tasks") is False)
        validator.require("status_at_task00_gate", status.get("phase") == "VALIDATED_PENDING_TASK00_MODEL_LOCK_REVIEW" and status.get("next_gate") == "TASK00_MODEL_LOCK_REVIEW")
        validator.require("request_pending_not_approval", request.get("status") == "PENDING_TASK00_REVIEW" and request.get("request_is_approval") is False and request.get("task02_may_self_lock") is False)
        for evidence in request["evidence"]:
            path = REPO_ROOT / evidence["path"]
            if not path.is_file() or sha256_file(path) != evidence["sha256"] or path.stat().st_size != int(evidence["size_bytes"]):
                raise IntegrityError(f"Request evidence verification failed: {evidence['path']}")
        validator.require("task01_still_unchanged", sha256_file(REPO_ROOT / "revision_management/01_data_audit/DATASET_LOCK.json") == EXPECTED_DATASET_LOCK_SHA256)
        payload = {
            "artifact": "TASK02_E001_FULL_TRAINING_PACKAGE_VERIFICATION",
            "status": "PASS",
            "verified_at_utc": now_utc(),
            "output_manifest_path": relative(OUTPUT_MANIFEST_PATH),
            "output_manifest_sha256": sha256_file(OUTPUT_MANIFEST_PATH),
            "output_manifest_entries": int(len(manifest)),
            "raw_job_result_entries_reverified": int(len(job_hash_manifest)),
            "validation_report_path": relative(VALIDATION_REPORT_PATH),
            "validation_report_sha256": sha256_file(VALIDATION_REPORT_PATH),
            "model_lock_candidate_path": relative(MODEL_LOCK_PATH),
            "model_lock_candidate_sha256": sha256_file(MODEL_LOCK_PATH),
            "task00_request_path": relative(REQUEST_PATH),
            "task00_request_sha256": sha256_file(REQUEST_PATH),
            "handoff_path": relative(HANDOFF_PATH),
            "handoff_sha256": sha256_file(HANDOFF_PATH),
            "checks": validator.checks,
            "model_lock_locked": False,
            "downstream_use_authorized": False,
        }
        atomic_json(PACKAGE_REPORT_PATH, payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        payload = {
            "artifact": "TASK02_E001_FULL_TRAINING_PACKAGE_VERIFICATION",
            "status": "FAIL",
            "failed_at_utc": now_utc(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "checks": validator.checks,
            "model_lock_modified_to_locked": False,
        }
        atomic_json(PACKAGE_REPORT_PATH, payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--verify-package", action="store_true")
    args = parser.parse_args()
    if args.repo_root.resolve() != REPO_ROOT.resolve():
        raise SystemExit(f"Unexpected repo root: {args.repo_root}")
    return verify_package() if args.verify_package else run_validation()


if __name__ == "__main__":
    raise SystemExit(main())
