#!/usr/bin/env python3
"""Run only the Task 00-authorized primary outer-fold GPU smoke.

The script is deliberately non-configurable at the command line: scenario,
designs, fold, candidates, and repeat count come from the registered protocol.
It refuses to run if the Task 00 approval unexpectedly authorizes full training.
"""

from __future__ import annotations

import gc
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)


SCRIPT_PATH = Path(__file__).resolve()
TASK_DIR = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]
sys.path.insert(0, str(TASK_DIR / "src"))

from gpu_models import (
    LightGBMCudaBinary,
    TorchCudaLogisticHead,
    TorchCudaPlattCalibrator,
    XGBoostCudaBinary,
    build_fingerprint_matrices,
)
from graph_models import build_graph_records, predict_raw as predict_graph_raw, train_with_isolated_early_stop
from lm_embeddings import extract_or_load_embeddings
from runtime_guards import cuda_evidence, require_cuda_runtime, run_with_single_oom_retry
from split_protocol import sha256_file


PROTOCOL_PATH = TASK_DIR / "configs" / "registered_protocol_v3_gpu_only.json"
APPROVAL_PATH = TASK_DIR / "TASK00_PROTOCOL_APPROVAL.json"
OUTPUT_ROOT = TASK_DIR / "predictions" / "limited_smoke"


def hash_ids(values) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()


def array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8).tobytes()).hexdigest()


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def best_mcc_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], probability)))
    rows = []
    for threshold in candidates:
        score = matthews_corrcoef(y, (probability >= threshold).astype(int))
        rows.append((float(score), -abs(float(threshold) - 0.5), -float(threshold), float(threshold)))
    rows.sort(reverse=True)
    return rows[0][3]


def metric_row(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "positive": int(y.sum()),
        "negative": int(len(y) - y.sum()),
        "roc_auc": float(roc_auc_score(y, probability)),
        "auprc": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "threshold": float(threshold),
        "mcc": float(matthews_corrcoef(y, prediction)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def indices_for_ids(data: pd.DataFrame, identifiers) -> np.ndarray:
    requested = set(map(str, identifiers))
    indices = np.flatnonzero(data["model_unit_id"].astype(str).isin(requested).to_numpy())
    if len(indices) != len(requested):
        raise RuntimeError("Split identifiers do not align one-to-one with locked input")
    return indices.astype(np.int64)


def load_fold(data: pd.DataFrame, design_id: str, outer_fold: int):
    path = TASK_DIR / "splits" / "primary_conflict_excluded" / design_id / "split_manifest.csv"
    manifest = pd.read_csv(path)
    outer = manifest[(manifest.level == "outer") & (manifest.outer_fold == outer_fold)]
    inner = manifest[(manifest.level == "inner") & (manifest.outer_fold == outer_fold)]
    outer_train = indices_for_ids(data, outer.loc[outer.subset == "train", "model_unit_id"])
    outer_test = indices_for_ids(data, outer.loc[outer.subset == "test", "model_unit_id"])
    if set(outer_train) & set(outer_test):
        raise RuntimeError("Outer train/test index overlap")
    inner_splits = []
    for inner_fold in sorted(inner.inner_fold.unique()):
        fold = inner[inner.inner_fold == inner_fold]
        train = indices_for_ids(data, fold.loc[fold.subset == "train", "model_unit_id"])
        validation = indices_for_ids(data, fold.loc[fold.subset == "validation", "model_unit_id"])
        if set(train) & set(validation) or set(validation) & set(outer_test):
            raise RuntimeError("Inner split overlap")
        inner_splits.append((int(inner_fold), train, validation))
    split_group_map = (
        outer.drop_duplicates("model_unit_id").set_index("model_unit_id")["split_group_id"].astype(str).to_dict()
    )
    split_groups = np.asarray(
        [split_group_map[str(identifier)] for identifier in data["model_unit_id"]],
        dtype=object,
    )
    return manifest, outer_train, outer_test, inner_splits, split_groups


def prepare_features(data: pd.DataFrame, protocol: dict, protocol_hash: str, primary_sha256: str):
    fingerprint = build_fingerprint_matrices(data["canonical_isomeric_smiles"].astype(str).tolist())
    fingerprint_path = TASK_DIR / "features" / "primary_smoke_fingerprints.npz"
    np.savez_compressed(
        fingerprint_path,
        ECFP4=fingerprint["ECFP4"],
        MACCS=fingerprint["MACCS"],
        model_unit_ids=data["model_unit_id"].astype(str).to_numpy(),
    )
    fingerprint_meta = {
        "input_sha256": primary_sha256,
        "protocol_sha256": protocol_hash,
        "ordered_model_unit_id_sha256": hash_ids(data["model_unit_id"].astype(str)),
        "ECFP4_shape": list(fingerprint["ECFP4"].shape),
        "ECFP4_array_sha256": array_hash(fingerprint["ECFP4"]),
        "MACCS_shape": list(fingerprint["MACCS"].shape),
        "MACCS_array_sha256": array_hash(fingerprint["MACCS"]),
        "model_training_executed": False,
    }
    fingerprint_meta_path = TASK_DIR / "features" / "primary_smoke_fingerprints.metadata.json"
    fingerprint_meta_path.write_text(json.dumps(fingerprint_meta, indent=2, sort_keys=True) + "\n")

    embeddings = {}
    embedding_metadata = {}
    lm_config = protocol["approved_candidates"]["smiles_language_models"]
    for model_index, model_spec in enumerate(lm_config["models"]):
        matrix, metadata = extract_or_load_embeddings(
            data=data,
            model_spec=model_spec,
            input_sha256=primary_sha256,
            protocol_sha256=protocol_hash,
            cache_root=TASK_DIR / "features" / "lm_embeddings",
            max_token_length=int(lm_config["max_token_length"]),
            seed=int(protocol["base_seed"] + 900000 + model_index),
        )
        embeddings[model_spec["name"]] = matrix
        embedding_metadata[model_spec["name"]] = metadata
    graph_records = build_graph_records(data)
    return fingerprint, embeddings, graph_records, fingerprint_meta, embedding_metadata


def member_specs() -> list[dict]:
    members = []
    for feature in ("ECFP4", "MACCS"):
        for algorithm in ("XGBoost", "LightGBM"):
            members.append(
                {
                    "member": f"{algorithm}_{feature}",
                    "branch": "fingerprint",
                    "family": "fingerprint",
                    "algorithm": algorithm,
                    "feature": feature,
                }
            )
    for architecture in ("GCN", "GAT", "MPNN", "AttentiveFP"):
        members.append(
            {
                "member": architecture,
                "branch": "graph",
                "family": "graph",
                "architecture": architecture,
            }
        )
    for model_name in ("ChemBERTa", "MolFormer"):
        members.append(
            {
                "member": f"{model_name}_CUDA_Logistic",
                "branch": "language_model",
                "family": "language_model",
                "model_name": model_name,
            }
        )
    return members


def fit_member(
    spec: dict,
    train_indices: np.ndarray,
    split_groups: np.ndarray,
    seed: int,
    protocol: dict,
    y: np.ndarray,
    fingerprint: dict[str, np.ndarray],
    embeddings: dict[str, np.ndarray],
    graph_records,
):
    if spec["family"] == "fingerprint":
        matrix = fingerprint[spec["feature"]]
        config = protocol["approved_candidates"]["fingerprint_ml"]["algorithms"][spec["algorithm"]]
        if spec["algorithm"] == "XGBoost":
            model = XGBoostCudaBinary(config["smoke_parameters"], seed).fit(
                matrix[train_indices], y[train_indices]
            )
        else:
            model = LightGBMCudaBinary(config["smoke_parameters"], seed).fit(
                matrix[train_indices], y[train_indices]
            )
        evidence = asdict(model.evidence)
        return model, evidence, {"oom_reductions_used": 0, "attempted_batch_sizes": []}

    if spec["family"] == "language_model":
        matrix = embeddings[spec["model_name"]]
        config = protocol["approved_candidates"]["smiles_language_models"]
        model = TorchCudaLogisticHead(
            c_value=float(config["smoke_logistic_c"]),
            max_iter=int(config["smoke_head_max_iter"]),
            seed=seed,
        ).fit(matrix[train_indices], y[train_indices])
        return model, model.evidence(), {"oom_reductions_used": 0, "attempted_batch_sizes": []}

    graph_config = protocol["approved_candidates"]["graph_dl"]["smoke_parameters"]

    def operation(batch_size: int):
        parameters = dict(graph_config)
        parameters["batch_size"] = batch_size
        return train_with_isolated_early_stop(
            graph_records,
            train_indices,
            split_groups[train_indices],
            spec["architecture"],
            parameters,
            seed,
        )

    (model, evidence), retry = run_with_single_oom_retry(
        operation,
        int(graph_config["batch_size"]),
        int(graph_config["single_oom_reduced_batch_size"]),
    )
    return model, evidence, json_ready(asdict(retry))


def predict_member(spec, model, indices, protocol, fingerprint, embeddings, graph_records):
    if spec["family"] == "fingerprint":
        return model.predict_raw(fingerprint[spec["feature"]][indices])
    if spec["family"] == "language_model":
        return model.predict_raw(embeddings[spec["model_name"]][indices])
    batch_size = int(protocol["approved_candidates"]["graph_dl"]["smoke_parameters"]["batch_size"])
    return predict_graph_raw(model, graph_records, indices, batch_size)


def run_one_design(
    repeat: int,
    design_index: int,
    design_id: str,
    data: pd.DataFrame,
    protocol: dict,
    y: np.ndarray,
    fingerprint,
    embeddings,
    graph_records,
):
    outer_fold = int(protocol["limited_smoke_registration"]["outer_fold_zero_based"])
    manifest, outer_train, outer_test, inner_splits, split_groups = load_fold(data, design_id, outer_fold)
    member_test_rows = []
    member_train_rows = []
    fit_events = []
    train_probabilities = {}
    test_probabilities = {}

    for member_index, spec in enumerate(member_specs()):
        member_seed = int(protocol["base_seed"] + design_index * 100000 + member_index * 1000)
        cross_raw = np.full(len(data), np.nan, dtype=np.float64)
        for inner_fold, inner_train, inner_validation in inner_splits:
            seed = member_seed + inner_fold * 10
            print(
                f"[SMOKE] repeat={repeat} design={design_id} member={spec['member']} "
                f"inner_fold={inner_fold}",
                flush=True,
            )
            try:
                model, evidence, retry = fit_member(
                    spec,
                    inner_train,
                    split_groups,
                    seed,
                    protocol,
                    y,
                    fingerprint,
                    embeddings,
                    graph_records,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Limited smoke failed at repeat={repeat}, design={design_id}, "
                    f"member={spec['member']}, inner_fold={inner_fold}: {exc}"
                ) from exc
            raw = predict_member(
                spec, model, inner_validation, protocol, fingerprint, embeddings, graph_records
            )
            cross_raw[inner_validation] = raw
            if spec["family"] == "fingerprint":
                evidence = asdict(model.evidence)
            fit_events.append(
                {
                    "repeat": repeat,
                    "design_id": design_id,
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "stage": "inner_crossfit",
                    "member": spec["member"],
                    "family": spec["family"],
                    "seed": seed,
                    "training_units": len(inner_train),
                    "prediction_units": len(inner_validation),
                    "training_ids_sha256": hash_ids(data.iloc[inner_train]["model_unit_id"]),
                    "prediction_ids_sha256": hash_ids(data.iloc[inner_validation]["model_unit_id"]),
                    "device_evidence": evidence,
                    "oom_evidence": retry,
                }
            )
            del model
            gc.collect()
            torch.cuda.empty_cache()

        if not np.isfinite(cross_raw[outer_train]).all():
            raise RuntimeError(f"Incomplete inner cross-fit coverage for {design_id} {spec['member']}")
        if np.isfinite(cross_raw[outer_test]).any():
            raise RuntimeError("Outer-test prediction entered the cross-fit calibration vector")

        calibration_seed = member_seed + 500000
        calibrator = TorchCudaPlattCalibrator(
            max_iter=int(protocol["calibration"]["max_iter"]),
            seed=calibration_seed,
        ).fit(cross_raw[outer_train], y[outer_train])
        train_probability = calibrator.predict_proba(cross_raw[outer_train])
        calibration_ids = data.iloc[outer_train]["model_unit_id"].astype(str).tolist()
        test_ids = set(data.iloc[outer_test]["model_unit_id"].astype(str))
        calibration_overlap = len(set(calibration_ids) & test_ids)
        if calibration_overlap:
            raise RuntimeError("Calibrator training identifiers overlap outer test")
        fit_events.append(
            {
                "repeat": repeat,
                "design_id": design_id,
                "outer_fold": outer_fold,
                "inner_fold": None,
                "stage": "outer_train_calibration",
                "member": spec["member"],
                "family": "calibrator",
                "seed": calibration_seed,
                "training_units": len(outer_train),
                "prediction_units": len(outer_test),
                "training_ids_sha256": hash_ids(calibration_ids),
                "outer_test_overlap_count": calibration_overlap,
                "raw_prediction_source": "inner_crossfit_outer_train_only",
                "device_evidence": calibrator.evidence(),
                "oom_evidence": {"oom_reductions_used": 0, "attempted_batch_sizes": []},
            }
        )

        final_seed = member_seed + 900000
        print(
            f"[SMOKE] repeat={repeat} design={design_id} member={spec['member']} outer_final_fit",
            flush=True,
        )
        final_model, final_evidence, final_retry = fit_member(
            spec,
            outer_train,
            split_groups,
            final_seed,
            protocol,
            y,
            fingerprint,
            embeddings,
            graph_records,
        )
        outer_raw = predict_member(
            spec, final_model, outer_test, protocol, fingerprint, embeddings, graph_records
        )
        if spec["family"] == "fingerprint":
            final_evidence = asdict(final_model.evidence)
        outer_probability = calibrator.predict_proba(outer_raw)
        fit_events.append(
            {
                "repeat": repeat,
                "design_id": design_id,
                "outer_fold": outer_fold,
                "inner_fold": None,
                "stage": "outer_final_fit",
                "member": spec["member"],
                "family": spec["family"],
                "seed": final_seed,
                "training_units": len(outer_train),
                "prediction_units": len(outer_test),
                "training_ids_sha256": hash_ids(data.iloc[outer_train]["model_unit_id"]),
                "prediction_ids_sha256": hash_ids(data.iloc[outer_test]["model_unit_id"]),
                "device_evidence": final_evidence,
                "oom_evidence": final_retry,
            }
        )
        train_probabilities[spec["member"]] = train_probability
        test_probabilities[spec["member"]] = outer_probability
        for local, index in enumerate(outer_train):
            member_train_rows.append(
                {
                    "repeat": repeat,
                    "scenario_id": "primary_conflict_excluded",
                    "design_id": design_id,
                    "outer_fold": outer_fold,
                    "member": spec["member"],
                    "branch": spec["branch"],
                    "model_unit_id": data.iloc[index]["model_unit_id"],
                    "cv_group_id": data.iloc[index]["cv_group_id"],
                    "final_label": int(y[index]),
                    "raw_crossfit_score": float(cross_raw[index]),
                    "calibrated_probability": float(train_probability[local]),
                }
            )
        for local, index in enumerate(outer_test):
            member_test_rows.append(
                {
                    "repeat": repeat,
                    "scenario_id": "primary_conflict_excluded",
                    "design_id": design_id,
                    "outer_fold": outer_fold,
                    "member": spec["member"],
                    "branch": spec["branch"],
                    "model_unit_id": data.iloc[index]["model_unit_id"],
                    "cv_group_id": data.iloc[index]["cv_group_id"],
                    "final_label": int(y[index]),
                    "raw_score": float(outer_raw[local]),
                    "calibrated_probability": float(outer_probability[local]),
                    "threshold": 0.5,
                    "predicted_label": int(outer_probability[local] >= 0.5),
                }
            )
        del final_model, calibrator
        gc.collect()
        torch.cuda.empty_cache()

    branches = {spec["member"]: spec["branch"] for spec in member_specs()}
    train_branch = {}
    test_branch = {}
    for branch in ("fingerprint", "graph", "language_model"):
        names = [name for name, value in branches.items() if value == branch]
        train_branch[branch] = np.mean([train_probabilities[name] for name in names], axis=0)
        test_branch[branch] = np.mean([test_probabilities[name] for name in names], axis=0)
    ensemble_train = np.mean(list(train_branch.values()), axis=0)
    ensemble_test = np.mean(list(test_branch.values()), axis=0)
    threshold = best_mcc_threshold(y[outer_train], ensemble_train)
    for local, index in enumerate(outer_train):
        member_train_rows.append(
            {
                "repeat": repeat,
                "scenario_id": "primary_conflict_excluded",
                "design_id": design_id,
                "outer_fold": outer_fold,
                "member": "FINAL_ENSEMBLE",
                "branch": "ensemble",
                "model_unit_id": data.iloc[index]["model_unit_id"],
                "cv_group_id": data.iloc[index]["cv_group_id"],
                "final_label": int(y[index]),
                "raw_crossfit_score": np.nan,
                "calibrated_probability": float(ensemble_train[local]),
            }
        )
    for local, index in enumerate(outer_test):
        member_test_rows.append(
            {
                "repeat": repeat,
                "scenario_id": "primary_conflict_excluded",
                "design_id": design_id,
                "outer_fold": outer_fold,
                "member": "FINAL_ENSEMBLE",
                "branch": "ensemble",
                "model_unit_id": data.iloc[index]["model_unit_id"],
                "cv_group_id": data.iloc[index]["cv_group_id"],
                "final_label": int(y[index]),
                "raw_score": np.nan,
                "calibrated_probability": float(ensemble_test[local]),
                "threshold": float(threshold),
                "predicted_label": int(ensemble_test[local] >= threshold),
            }
        )
    return member_test_rows, member_train_rows, fit_events


def main() -> int:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    approval = json.loads(APPROVAL_PATH.read_text(encoding="utf-8"))
    if approval["status"] != "CONDITIONAL_APPROVAL":
        raise RuntimeError("Task 00 conditional approval is missing")
    if approval["full_training_authorized"] is not False:
        raise RuntimeError("This limited-smoke runner refuses a full-training authorization state")
    if not approval["one_outer_fold_smoke_authorized"]:
        raise RuntimeError("Task 00 did not authorize the limited smoke")
    if protocol["hard_guards"]["full_training_authorized"] is not False:
        raise RuntimeError("Registered protocol does not preserve the full-training block")
    if protocol["limited_smoke_registration"]["sensitivity_model_fitting"] is not False:
        raise RuntimeError("Sensitivity fitting is forbidden in the limited smoke")
    device = require_cuda_runtime()
    runtime = cuda_evidence()
    protocol_hash = sha256_file(PROTOCOL_PATH)

    primary_spec = next(item for item in protocol["inputs"] if item["role"] == "primary")
    primary_path = REPO_ROOT / primary_spec["path"]
    if sha256_file(primary_path) != primary_spec["sha256"]:
        raise RuntimeError("Primary locked input hash mismatch")
    data = pd.read_csv(primary_path)
    if len(data) != primary_spec["rows"]:
        raise RuntimeError("Primary row-count mismatch")
    y = data["final_label"].to_numpy(dtype=int)
    fingerprint, embeddings, graph_records, fingerprint_meta, embedding_meta = prepare_features(
        data, protocol, protocol_hash, primary_spec["sha256"]
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_test_rows = []
    all_train_rows = []
    all_fit_events = []
    designs = protocol["limited_smoke_registration"]["designs"]
    repeats = int(protocol["limited_smoke_registration"]["repeats_for_determinism"])
    for repeat in range(1, repeats + 1):
        for design_index, design_id in enumerate(designs):
            test_rows, train_rows, fit_events = run_one_design(
                repeat,
                design_index,
                design_id,
                data,
                protocol,
                y,
                fingerprint,
                embeddings,
                graph_records,
            )
            all_test_rows.extend(test_rows)
            all_train_rows.extend(train_rows)
            all_fit_events.extend(fit_events)

    test_predictions = pd.DataFrame(all_test_rows).sort_values(
        ["repeat", "design_id", "member", "model_unit_id"], kind="mergesort"
    )
    train_predictions = pd.DataFrame(all_train_rows).sort_values(
        ["repeat", "design_id", "member", "model_unit_id"], kind="mergesort"
    )
    if not np.isfinite(test_predictions["calibrated_probability"]).all():
        raise RuntimeError("Non-finite smoke probability")
    if not test_predictions["calibrated_probability"].between(0.0, 1.0).all():
        raise RuntimeError("Out-of-range smoke probability")

    expected_members = {spec["member"] for spec in member_specs()} | {"FINAL_ENSEMBLE"}
    oof_checks = []
    for repeat in range(1, repeats + 1):
        for design_id in designs:
            fold = test_predictions[
                (test_predictions.repeat == repeat) & (test_predictions.design_id == design_id)
            ]
            expected_ids = set(fold[fold.member == "FINAL_ENSEMBLE"]["model_unit_id"])
            for member in expected_members:
                rows = fold[fold.member == member]
                counts = rows["model_unit_id"].value_counts()
                ok = set(rows["model_unit_id"]) == expected_ids and bool((counts == 1).all())
                oof_checks.append(
                    {
                        "repeat": repeat,
                        "design_id": design_id,
                        "member": member,
                        "expected_units": len(expected_ids),
                        "observed_rows": len(rows),
                        "unique_and_complete": ok,
                    }
                )
                if not ok:
                    raise RuntimeError(f"Smoke OOF coverage failure: {repeat} {design_id} {member}")

    key_columns = ["design_id", "member", "model_unit_id"]
    first = test_predictions[test_predictions.repeat == 1].set_index(key_columns).sort_index()
    second = test_predictions[test_predictions.repeat == 2].set_index(key_columns).sort_index()
    if not first.index.equals(second.index):
        raise RuntimeError("Deterministic repeats have different prediction keys")
    difference = np.abs(
        first["calibrated_probability"].to_numpy() - second["calibrated_probability"].to_numpy()
    )
    tolerance = float(protocol["limited_smoke_registration"]["probability_absolute_tolerance"])
    maximum_difference = float(difference.max(initial=0.0))
    if maximum_difference > tolerance:
        raise RuntimeError(
            f"Determinism tolerance failed: {maximum_difference} > {tolerance}"
        )

    metric_rows = []
    for (repeat, design_id, member), frame in test_predictions.groupby(
        ["repeat", "design_id", "member"], sort=True
    ):
        threshold = float(frame["threshold"].iloc[0])
        metric_rows.append(
            {
                "repeat": repeat,
                "design_id": design_id,
                "member": member,
                **metric_row(
                    frame["final_label"].to_numpy(dtype=int),
                    frame["calibrated_probability"].to_numpy(dtype=float),
                    threshold,
                ),
            }
        )
    metrics = pd.DataFrame(metric_rows)

    test_path = OUTPUT_ROOT / "limited_smoke_outer_predictions.csv"
    train_path = OUTPUT_ROOT / "limited_smoke_outer_train_crossfit_predictions.csv"
    metrics_path = TASK_DIR / "metrics" / "limited_smoke_metrics.csv"
    events_path = TASK_DIR / "logs" / "limited_smoke_fit_events.jsonl"
    test_predictions.to_csv(test_path, index=False, lineterminator="\n")
    train_predictions.to_csv(train_path, index=False, lineterminator="\n")
    metrics.to_csv(metrics_path, index=False, lineterminator="\n")
    with events_path.open("w", encoding="utf-8") as handle:
        for event in all_fit_events:
            handle.write(json.dumps(json_ready(event), sort_keys=True) + "\n")

    calibration_events = [event for event in all_fit_events if event["stage"] == "outer_train_calibration"]
    device_failures = []
    for event in all_fit_events:
        evidence = event.get("device_evidence", {})
        serialized = json.dumps(json_ready(evidence)).lower()
        if "cuda" not in serialized:
            device_failures.append(
                f"{event['repeat']}:{event['design_id']}:{event['member']}:{event['stage']}"
            )
    if device_failures:
        raise RuntimeError(f"Fit events without CUDA evidence: {device_failures[:5]}")
    if any(event.get("outer_test_overlap_count", 0) != 0 for event in calibration_events):
        raise RuntimeError("Calibration isolation evidence failed")

    report = {
        "artifact": "TASK02_LIMITED_GPU_SMOKE_VALIDATION",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "authorization": "TASK00_P001_CONDITIONAL_ONE_OUTER_FOLD_SMOKE_ONLY",
        "protocol_path": PROTOCOL_PATH.relative_to(REPO_ROOT).as_posix(),
        "protocol_sha256": protocol_hash,
        "task00_approval_sha256": sha256_file(APPROVAL_PATH),
        "runtime": runtime,
        "scenario": "primary_conflict_excluded",
        "designs": designs,
        "outer_fold_zero_based": protocol["limited_smoke_registration"]["outer_fold_zero_based"],
        "repeats": repeats,
        "approved_member_count": len(member_specs()),
        "fit_event_count": len(all_fit_events),
        "calibrator_fit_event_count": len(calibration_events),
        "fit_events_without_cuda_evidence": device_failures,
        "calibration_outer_test_overlap_count": int(
            sum(event.get("outer_test_overlap_count", 0) for event in calibration_events)
        ),
        "all_probabilities_finite_and_in_range": True,
        "oof_checks": oof_checks,
        "determinism": {
            "status": "PASS",
            "absolute_tolerance": tolerance,
            "maximum_probability_difference": maximum_difference,
        },
        "fingerprint_cache": fingerprint_meta,
        "embedding_caches": embedding_meta,
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
                "rows": len(all_fit_events),
            },
        },
        "formal_full_training_executed": False,
        "sensitivity_model_fitting_executed": False,
        "external_validation_used": False,
        "mfh_rescreening_performed": False,
        "model_lock_changed_to_locked": False,
        "device": str(device),
    }
    report_path = TASK_DIR / "reports" / "limited_gpu_smoke_validation.json"
    report_path.write_text(json.dumps(json_ready(report), indent=2, sort_keys=True) + "\n")
    print(json.dumps(json_ready(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
