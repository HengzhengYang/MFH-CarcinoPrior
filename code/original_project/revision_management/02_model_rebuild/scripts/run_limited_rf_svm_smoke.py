#!/usr/bin/env python3
"""Run only the P002-authorized RF/SVM primary outer-fold-0 smoke."""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import gc
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


SCRIPT_PATH = Path(__file__).resolve()
TASK_DIR = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]
sys.path.insert(0, str(TASK_DIR / "src"))

from gpu_models import TorchCudaPlattCalibrator
from hybrid_models import SklearnCpuRandomForestBinary, SklearnCpuRbfSvcBinary
from runtime_guards import cuda_evidence, require_cuda_runtime
from split_protocol import sha256_file


PROTOCOL_PATH = TASK_DIR / "configs" / "registered_protocol_v4_hybrid_restricted.json"
P002_PATH = TASK_DIR / "TASK00_PROTOCOL_AMENDMENT_P002.json"
SPLIT_REPORT_PATH = TASK_DIR / "reports" / "split_manifest_revalidation_p002.json"
OUTPUT_ROOT = TASK_DIR / "predictions" / "p002_rf_svm_limited_smoke"


def hash_ids(values) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()


def array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8).tobytes()).hexdigest()


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def indices_for_ids(data: pd.DataFrame, identifiers) -> np.ndarray:
    requested = set(map(str, identifiers))
    indices = np.flatnonzero(data["model_unit_id"].astype(str).isin(requested).to_numpy())
    if len(indices) != len(requested):
        raise RuntimeError("P002 split identifiers do not align one-to-one with locked input")
    return indices.astype(np.int64)


def load_fold(data: pd.DataFrame, design_id: str, outer_fold: int):
    path = TASK_DIR / "splits" / "primary_conflict_excluded" / design_id / "split_manifest.csv"
    manifest = pd.read_csv(path)
    outer = manifest[(manifest.level == "outer") & (manifest.outer_fold == outer_fold)]
    inner = manifest[(manifest.level == "inner") & (manifest.outer_fold == outer_fold)]
    outer_train = indices_for_ids(data, outer.loc[outer.subset == "train", "model_unit_id"])
    outer_test = indices_for_ids(data, outer.loc[outer.subset == "test", "model_unit_id"])
    if set(outer_train) & set(outer_test):
        raise RuntimeError("P002 outer train/test overlap")
    inner_splits = []
    for inner_fold in sorted(inner.inner_fold.unique()):
        fold = inner[inner.inner_fold == inner_fold]
        train = indices_for_ids(data, fold.loc[fold.subset == "train", "model_unit_id"])
        validation = indices_for_ids(data, fold.loc[fold.subset == "validation", "model_unit_id"])
        if set(train) & set(validation) or set(validation) & set(outer_test):
            raise RuntimeError("P002 inner split overlap")
        inner_splits.append((int(inner_fold), train, validation))
    if len(inner_splits) != 5:
        raise RuntimeError("P002 requires exactly five registered inner folds")
    return outer_train, outer_test, inner_splits


def load_registered_features(data: pd.DataFrame, protocol: dict):
    config = protocol["fingerprint_features"]
    cache_path = REPO_ROOT / config["cache_path"]
    if sha256_file(cache_path) != config["cache_sha256"]:
        raise RuntimeError("P002 registered fingerprint cache hash mismatch")
    # The already-registered internal NPZ stores string IDs as an object array.
    # Pickle loading is permitted only after the complete file hash above has
    # matched the preregistered value, and IDs are then checked one-to-one.
    with np.load(cache_path, allow_pickle=True) as cached:
        ids = cached["model_unit_ids"].astype(str)
        expected_ids = data["model_unit_id"].astype(str).to_numpy()
        if not np.array_equal(ids, expected_ids):
            raise RuntimeError("P002 fingerprint cache model_unit_id order mismatch")
        if hash_ids(ids) != config["ordered_model_unit_id_sha256"]:
            raise RuntimeError("P002 registered model_unit_id fingerprint mismatch")
        matrices = {
            "ECFP4": np.ascontiguousarray(cached["ECFP4"], dtype=np.float32),
            "MACCS": np.ascontiguousarray(cached["MACCS"], dtype=np.float32),
        }
    for name, matrix in matrices.items():
        spec = config[name]
        if list(matrix.shape) != spec["shape"] or array_hash(matrix) != spec["array_sha256"]:
            raise RuntimeError(f"P002 registered {name} feature identity mismatch")
        if not np.isfinite(matrix).all():
            raise RuntimeError(f"P002 {name} features contain non-finite values")
    return matrices


def member_specs() -> list[dict]:
    return [
        {"member": "RandomForest_ECFP4", "algorithm": "RandomForest", "feature": "ECFP4"},
        {"member": "RandomForest_MACCS", "algorithm": "RandomForest", "feature": "MACCS"},
        {"member": "RBF_SVM_ECFP4", "algorithm": "RBF_SVM", "feature": "ECFP4"},
        {"member": "RBF_SVM_MACCS", "algorithm": "RBF_SVM", "feature": "MACCS"},
    ]


def fit_member(spec: dict, train_indices: np.ndarray, seed: int, protocol: dict, y, features):
    candidate = protocol["restored_candidates"][spec["algorithm"]]
    limits = protocol["cpu_resource_limits"]
    if spec["algorithm"] == "RandomForest":
        model = SklearnCpuRandomForestBinary(candidate["smoke_parameters"], seed, limits)
    elif spec["algorithm"] == "RBF_SVM":
        model = SklearnCpuRbfSvcBinary(candidate["smoke_parameters"], seed, limits)
    else:
        raise RuntimeError(f"P002 unregistered member algorithm: {spec['algorithm']}")
    model.fit(features[spec["feature"]][train_indices], y[train_indices])
    return model


def run_one_design(repeat: int, design_index: int, design_id: str, data, protocol, y, features):
    outer_fold = int(protocol["p002_limited_smoke"]["outer_fold_zero_based"])
    outer_train, outer_test, inner_splits = load_fold(data, design_id, outer_fold)
    test_rows = []
    train_rows = []
    events = []
    for member_index, spec in enumerate(member_specs()):
        member_seed = int(protocol["base_seed"] + design_index * 100000 + member_index * 1000)
        cross_raw = np.full(len(data), np.nan, dtype=np.float64)
        for inner_fold, inner_train, inner_validation in inner_splits:
            seed = member_seed + inner_fold * 10
            print(
                f"[P002] repeat={repeat} design={design_id} member={spec['member']} "
                f"inner_fold={inner_fold}",
                flush=True,
            )
            model = fit_member(spec, inner_train, seed, protocol, y, features)
            raw = model.predict_raw(features[spec["feature"]][inner_validation])
            cross_raw[inner_validation] = raw
            evidence = model.evidence()
            events.append(
                {
                    "repeat": repeat,
                    "design_id": design_id,
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "stage": "inner_crossfit",
                    "member": spec["member"],
                    "family": "p002_cpu_whitelist_estimator",
                    "seed": seed,
                    "training_units": len(inner_train),
                    "prediction_units": len(inner_validation),
                    "training_ids_sha256": hash_ids(data.iloc[inner_train]["model_unit_id"]),
                    "prediction_ids_sha256": hash_ids(data.iloc[inner_validation]["model_unit_id"]),
                    "device_evidence": evidence,
                }
            )
            del model
            gc.collect()

        if not np.isfinite(cross_raw[outer_train]).all():
            raise RuntimeError(f"P002 incomplete outer-train cross-fit coverage: {design_id} {spec['member']}")
        if np.isfinite(cross_raw[outer_test]).any():
            raise RuntimeError("P002 outer-test prediction entered calibration input vector")

        calibration_seed = member_seed + 500000
        calibrator = TorchCudaPlattCalibrator(
            max_iter=int(protocol["calibration"]["max_iter"]), seed=calibration_seed
        ).fit(cross_raw[outer_train], y[outer_train])
        train_probability = calibrator.predict_proba(cross_raw[outer_train])
        calibration_ids = data.iloc[outer_train]["model_unit_id"].astype(str).tolist()
        outer_test_ids = set(data.iloc[outer_test]["model_unit_id"].astype(str))
        calibration_overlap = len(set(calibration_ids) & outer_test_ids)
        if calibration_overlap:
            raise RuntimeError("P002 calibrator training identifiers overlap outer test")
        events.append(
            {
                "repeat": repeat,
                "design_id": design_id,
                "outer_fold": outer_fold,
                "inner_fold": None,
                "stage": "outer_train_gpu_platt_calibration",
                "member": spec["member"],
                "family": "gpu_calibrator",
                "seed": calibration_seed,
                "training_units": len(outer_train),
                "training_ids_sha256": hash_ids(calibration_ids),
                "outer_test_overlap_count": calibration_overlap,
                "raw_prediction_source": "registered_inner_crossfit_outer_train_only",
                "device_evidence": calibrator.evidence(),
            }
        )

        final_seed = member_seed + 900000
        print(
            f"[P002] repeat={repeat} design={design_id} member={spec['member']} outer_final_fit",
            flush=True,
        )
        final_model = fit_member(spec, outer_train, final_seed, protocol, y, features)
        outer_raw = final_model.predict_raw(features[spec["feature"]][outer_test])
        outer_probability = calibrator.predict_proba(outer_raw)
        events.append(
            {
                "repeat": repeat,
                "design_id": design_id,
                "outer_fold": outer_fold,
                "inner_fold": None,
                "stage": "outer_final_fit",
                "member": spec["member"],
                "family": "p002_cpu_whitelist_estimator",
                "seed": final_seed,
                "training_units": len(outer_train),
                "prediction_units": len(outer_test),
                "training_ids_sha256": hash_ids(data.iloc[outer_train]["model_unit_id"]),
                "prediction_ids_sha256": hash_ids(data.iloc[outer_test]["model_unit_id"]),
                "device_evidence": final_model.evidence(),
            }
        )

        for local, index in enumerate(outer_train):
            train_rows.append(
                {
                    "repeat": repeat,
                    "scenario_id": "primary_conflict_excluded",
                    "design_id": design_id,
                    "outer_fold": outer_fold,
                    "member": spec["member"],
                    "feature": spec["feature"],
                    "model_unit_id": data.iloc[index]["model_unit_id"],
                    "cv_group_id": data.iloc[index]["cv_group_id"],
                    "final_label": int(y[index]),
                    "raw_crossfit_input": float(cross_raw[index]),
                    "calibrated_probability": float(train_probability[local]),
                }
            )
        for local, index in enumerate(outer_test):
            test_rows.append(
                {
                    "repeat": repeat,
                    "scenario_id": "primary_conflict_excluded",
                    "design_id": design_id,
                    "outer_fold": outer_fold,
                    "member": spec["member"],
                    "feature": spec["feature"],
                    "model_unit_id": data.iloc[index]["model_unit_id"],
                    "cv_group_id": data.iloc[index]["cv_group_id"],
                    "final_label": int(y[index]),
                    "raw_score": float(outer_raw[local]),
                    "calibrated_probability": float(outer_probability[local]),
                }
            )
        del final_model, calibrator
        gc.collect()
        torch.cuda.empty_cache()
    return test_rows, train_rows, events


def main() -> int:
    started = time.perf_counter()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    approval = json.loads(P002_PATH.read_text(encoding="utf-8"))
    if sha256_file(P002_PATH) != protocol["task00_amendment_sha256"]:
        raise RuntimeError("P002 approval hash mismatch")
    if approval["status"] != "CONDITIONAL_APPROVAL" or not approval["rf_svm_limited_smoke_authorized"]:
        raise RuntimeError("P002 did not authorize the RF/SVM limited smoke")
    for forbidden in (
        "full_training_authorized",
        "external_validation_authorized",
        "mfh_rescreening_authorized",
        "model_lock_authorized",
    ):
        if approval[forbidden] is not False:
            raise RuntimeError(f"P002 forbidden authorization unexpectedly true: {forbidden}")
    if protocol["hard_guards"]["full_training_authorized"] is not False:
        raise RuntimeError("Protocol v4 does not preserve the full-training block")
    if protocol["p002_limited_smoke"]["sensitivity_model_fitting"] is not False:
        raise RuntimeError("P002 sensitivity fitting is forbidden")

    split_report = json.loads(SPLIT_REPORT_PATH.read_text(encoding="utf-8"))
    if split_report["status"] != "PASS" or split_report["silent_regeneration_performed"] is not False:
        raise RuntimeError("P002 split revalidation evidence is missing")
    split_index_path = REPO_ROOT / protocol["split_policy"]["manifest_index_path"]
    if sha256_file(split_index_path) != protocol["split_policy"]["required_manifest_index_sha256"]:
        raise RuntimeError("P002 split index changed after read-only revalidation")

    require_cuda_runtime()
    runtime = cuda_evidence()
    primary = next(item for item in protocol["inputs"] if item["role"] == "primary")
    primary_path = REPO_ROOT / primary["path"]
    if sha256_file(primary_path) != primary["sha256"]:
        raise RuntimeError("P002 primary locked input hash mismatch")
    data = pd.read_csv(primary_path)
    if len(data) != primary["rows"]:
        raise RuntimeError("P002 primary locked input row-count mismatch")
    y = data["final_label"].to_numpy(dtype=np.int64)
    features = load_registered_features(data, protocol)

    specs = member_specs()
    if [item["member"] for item in specs] != protocol["p002_limited_smoke"]["members"]:
        raise RuntimeError("P002 smoke member set differs from the registered protocol")
    all_test_rows, all_train_rows, all_events = [], [], []
    designs = protocol["p002_limited_smoke"]["designs"]
    repeats = int(protocol["p002_limited_smoke"]["deterministic_repeats"])
    for repeat in range(1, repeats + 1):
        for design_index, design_id in enumerate(designs):
            test_rows, train_rows, events = run_one_design(
                repeat, design_index, design_id, data, protocol, y, features
            )
            all_test_rows.extend(test_rows)
            all_train_rows.extend(train_rows)
            all_events.extend(events)

    test_predictions = pd.DataFrame(all_test_rows).sort_values(
        ["repeat", "design_id", "member", "model_unit_id"], kind="mergesort"
    )
    train_predictions = pd.DataFrame(all_train_rows).sort_values(
        ["repeat", "design_id", "member", "model_unit_id"], kind="mergesort"
    )
    for frame in (test_predictions, train_predictions):
        probability = frame["calibrated_probability"]
        if not np.isfinite(probability).all() or not probability.between(0.0, 1.0).all():
            raise RuntimeError("P002 smoke produced invalid calibrated probabilities")

    oof_checks = []
    crossfit_checks = []
    expected_members = {item["member"] for item in specs}
    for repeat in range(1, repeats + 1):
        for design_id in designs:
            fold_test = test_predictions[
                (test_predictions.repeat == repeat) & (test_predictions.design_id == design_id)
            ]
            fold_train = train_predictions[
                (train_predictions.repeat == repeat) & (train_predictions.design_id == design_id)
            ]
            reference_test = set(fold_test[fold_test.member == specs[0]["member"]]["model_unit_id"])
            reference_train = set(fold_train[fold_train.member == specs[0]["member"]]["model_unit_id"])
            for member in expected_members:
                test_member = fold_test[fold_test.member == member]
                train_member = fold_train[fold_train.member == member]
                test_ok = set(test_member.model_unit_id) == reference_test and bool(
                    (test_member.model_unit_id.value_counts() == 1).all()
                )
                train_ok = set(train_member.model_unit_id) == reference_train and bool(
                    (train_member.model_unit_id.value_counts() == 1).all()
                )
                oof_checks.append(
                    {
                        "repeat": repeat,
                        "design_id": design_id,
                        "member": member,
                        "expected_units": len(reference_test),
                        "observed_rows": len(test_member),
                        "unique_and_complete": test_ok,
                    }
                )
                crossfit_checks.append(
                    {
                        "repeat": repeat,
                        "design_id": design_id,
                        "member": member,
                        "expected_units": len(reference_train),
                        "observed_rows": len(train_member),
                        "unique_and_complete": train_ok,
                    }
                )
                if not test_ok or not train_ok:
                    raise RuntimeError(f"P002 OOF/cross-fit coverage failure: {repeat} {design_id} {member}")

    keys = ["design_id", "member", "model_unit_id"]
    test_first = test_predictions[test_predictions.repeat == 1].set_index(keys).sort_index()
    test_second = test_predictions[test_predictions.repeat == 2].set_index(keys).sort_index()
    train_first = train_predictions[train_predictions.repeat == 1].set_index(keys).sort_index()
    train_second = train_predictions[train_predictions.repeat == 2].set_index(keys).sort_index()
    if not test_first.index.equals(test_second.index) or not train_first.index.equals(train_second.index):
        raise RuntimeError("P002 deterministic repeats have different prediction keys")
    test_difference = np.abs(
        test_first.calibrated_probability.to_numpy() - test_second.calibrated_probability.to_numpy()
    )
    train_difference = np.abs(
        train_first.calibrated_probability.to_numpy() - train_second.calibrated_probability.to_numpy()
    )
    maximum_difference = float(
        max(test_difference.max(initial=0.0), train_difference.max(initial=0.0))
    )
    tolerance = float(protocol["p002_limited_smoke"]["probability_absolute_tolerance"])
    if maximum_difference > tolerance:
        raise RuntimeError(f"P002 determinism failure: {maximum_difference} > {tolerance}")

    cpu_events = [event for event in all_events if event["family"] == "p002_cpu_whitelist_estimator"]
    calibration_events = [event for event in all_events if event["family"] == "gpu_calibrator"]
    allowed_fqns = {"sklearn.ensemble.RandomForestClassifier", "sklearn.svm.SVC"}
    observed_fqns = {event["device_evidence"]["estimator_fqn"] for event in cpu_events}
    whitelist_violations = [
        event
        for event in cpu_events
        if event["device_evidence"]["estimator_fqn"] not in allowed_fqns
        or event["device_evidence"]["whitelist_status"] != "PASS"
    ]
    if whitelist_violations or observed_fqns != allowed_fqns:
        raise RuntimeError("P002 CPU estimator whitelist audit failed")
    svc_probability_violations = [
        event
        for event in cpu_events
        if event["device_evidence"]["estimator_fqn"] == "sklearn.svm.SVC"
        and event["device_evidence"]["estimator_parameters"]["probability"] is not False
    ]
    if svc_probability_violations:
        raise RuntimeError("P002 observed SVC probability fitting")
    calibration_cuda_failures = [
        event
        for event in calibration_events
        if event["device_evidence"].get("device") != "cuda:0"
    ]
    if calibration_cuda_failures:
        raise RuntimeError("P002 calibration did not remain on cuda:0")
    calibration_overlap = sum(event.get("outer_test_overlap_count", 0) for event in calibration_events)
    if calibration_overlap:
        raise RuntimeError("P002 calibration isolation failed")

    metric_rows = []
    for (repeat, design_id, member), frame in test_predictions.groupby(
        ["repeat", "design_id", "member"], sort=True
    ):
        labels = frame.final_label.to_numpy(dtype=int)
        probability = frame.calibrated_probability.to_numpy(dtype=float)
        metric_rows.append(
            {
                "repeat": repeat,
                "design_id": design_id,
                "member": member,
                "n": len(frame),
                "roc_auc": float(roc_auc_score(labels, probability)),
                "auprc": float(average_precision_score(labels, probability)),
                "brier": float(brier_score_loss(labels, probability)),
                "interpretation": "P002_PIPELINE_SMOKE_NOT_FORMAL_PERFORMANCE",
            }
        )
    metrics = pd.DataFrame(metric_rows)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    test_path = OUTPUT_ROOT / "p002_rf_svm_outer_predictions.csv"
    train_path = OUTPUT_ROOT / "p002_rf_svm_outer_train_crossfit_predictions.csv"
    metrics_path = TASK_DIR / "metrics" / "p002_rf_svm_smoke_metrics.csv"
    events_path = TASK_DIR / "logs" / "p002_rf_svm_fit_events.jsonl"
    test_predictions.to_csv(test_path, index=False, lineterminator="\n")
    train_predictions.to_csv(train_path, index=False, lineterminator="\n")
    metrics.to_csv(metrics_path, index=False, lineterminator="\n")
    with events_path.open("w", encoding="utf-8") as handle:
        for event in all_events:
            handle.write(json.dumps(json_ready(event), sort_keys=True) + "\n")

    durations = [event["device_evidence"]["fit_duration_seconds"] for event in cpu_events]
    report = {
        "artifact": "TASK02_P002_LIMITED_RF_SVM_SMOKE_VALIDATION",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "authorization": "TASK00_P002_HYBRID_RF_SVM_LIMITED_SMOKE_ONLY",
        "protocol_path": PROTOCOL_PATH.relative_to(REPO_ROOT).as_posix(),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "p002_path": P002_PATH.relative_to(REPO_ROOT).as_posix(),
        "p002_sha256": sha256_file(P002_PATH),
        "split_revalidation_path": SPLIT_REPORT_PATH.relative_to(REPO_ROOT).as_posix(),
        "split_manifest_index_sha256": sha256_file(split_index_path),
        "runtime": runtime,
        "compute_policy": "HYBRID_RESTRICTED",
        "scenario": "primary_conflict_excluded",
        "designs": designs,
        "outer_fold_zero_based": protocol["p002_limited_smoke"]["outer_fold_zero_based"],
        "members": sorted(expected_members),
        "repeats": repeats,
        "cpu_fit_event_count": len(cpu_events),
        "gpu_platt_fit_event_count": len(calibration_events),
        "observed_cpu_estimator_fqns": sorted(observed_fqns),
        "cpu_whitelist_violation_count": len(whitelist_violations),
        "svc_probability_violation_count": len(svc_probability_violations),
        "gpu_calibration_failure_count": len(calibration_cuda_failures),
        "calibration_outer_test_overlap_count": int(calibration_overlap),
        "all_probabilities_finite_and_in_range": True,
        "outer_oof_checks": oof_checks,
        "outer_train_crossfit_checks": crossfit_checks,
        "determinism": {
            "status": "PASS",
            "absolute_tolerance": tolerance,
            "maximum_probability_difference": maximum_difference,
        },
        "timing": {
            "total_wall_seconds": float(time.perf_counter() - started),
            "cpu_estimator_fit_seconds_sum": float(sum(durations)),
            "cpu_estimator_fit_seconds_max": float(max(durations, default=0.0)),
        },
        "outputs": {
            "outer_predictions": {
                "path": test_path.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256_file(test_path),
                "rows": len(test_predictions),
            },
            "outer_train_crossfit_predictions": {
                "path": train_path.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256_file(train_path),
                "rows": len(train_predictions),
            },
            "metrics": {
                "path": metrics_path.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256_file(metrics_path),
                "rows": len(metrics),
            },
            "fit_events": {
                "path": events_path.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256_file(events_path),
                "rows": len(all_events),
            },
        },
        "formal_full_training_executed": False,
        "sensitivity_model_fitting_executed": False,
        "external_validation_used": False,
        "mfh_rescreening_performed": False,
        "manuscript_modified": False,
        "docking_or_dynamics_performed": False,
        "task01_files_modified": False,
        "model_lock_changed_to_locked": False,
    }
    report_path = TASK_DIR / "reports" / "p002_rf_svm_smoke_validation.json"
    report_path.write_text(json.dumps(json_ready(report), indent=2, sort_keys=True) + "\n")
    print(json.dumps(json_ready(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
