#!/usr/bin/env python3
"""Generate and validate the E001 no-fit execution plan.

This program performs no estimator, head, graph-model, or calibrator fit.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
TASK_DIR = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]
sys.path.insert(0, str(TASK_DIR / "src"))

from split_protocol import load_locked_scenario, sha256_file, validate_manifest


PROTOCOL_PATH = TASK_DIR / "configs" / "registered_protocol_v4_hybrid_restricted.json"
GO_FULL_PATH = TASK_DIR / "TASK00_GO_FULL_TRAINING_APPROVAL.json"
SUPPLEMENT_PATH = TASK_DIR / "TASK00_PROTOCOL_EXECUTION_SUPPLEMENT.json"
PLAN_PATH = TASK_DIR / "manifests" / "full_execution_plan_e001.csv"
REPORT_PATH = TASK_DIR / "reports" / "full_execution_plan_validation_e001.json"
PROGRESS_PATH = TASK_DIR / "runtime" / "full_training_progress.json"

EXPECTED_PROTOCOL_SHA256 = "2fc3cc0df70895643440dec40e27f51d045b88c62c7c50a8370af5688d592663"
EXPECTED_GO_FULL_SHA256 = "dae60ff147881db239e6b2f36dd0843f8f69687da8e4a9435fb9f9d4ff31dda4"
EXPECTED_SUPPLEMENT_SHA256 = "8ed6b3828c3acc53c2eca182833ad2b9e27fd6cfb099d358146f4ccba165c045"
EXPECTED_SPLIT_INDEX_SHA256 = "5ea19ab7c5d7c89bbbc79110bd2baa7fd86690c389127d2f88c9cba7aba41b8f"


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parameters_json(parameters: dict) -> str:
    return json.dumps(parameters, separators=(",", ":"), sort_keys=True)


def candidate_registry(protocol: dict, supplement: dict) -> list[dict]:
    members: list[dict] = []
    for feature in ("ECFP4", "MACCS"):
        for algorithm, prefix in (("RandomForest", "RandomForest"), ("RBF_SVM", "RBF_SVM")):
            registered = protocol["restored_candidates"][algorithm]
            fixed = dict(registered["fixed_parameters"])
            candidates = []
            for ordinal, item in enumerate(registered["full_candidates_per_feature"], start=1):
                if set(item) - {"id", "ordinal"} & set(fixed):
                    raise RuntimeError(f"Candidate overrides fixed parameters: {algorithm} {item['id']}")
                params = {**fixed, **{key: value for key, value in item.items() if key not in {"id", "ordinal"}}}
                candidates.append({"ordinal": ordinal, "id": item["id"], "parameters": params})
            members.append({
                "member": f"{prefix}_{feature}",
                "branch": "fingerprint",
                "family": algorithm,
                "feature": feature,
                "backend_policy": "cpu_p002_whitelist",
                "candidates": candidates,
            })

    fingerprint = supplement["gpu_candidate_registry"]["fingerprint_features"]
    for algorithm in ("XGBoost", "LightGBM"):
        for feature in ("ECFP4", "MACCS"):
            registered = fingerprint[feature][algorithm]
            fixed = dict(registered["fixed_parameters"])
            candidates = []
            for item in registered["candidates"]:
                params = {**fixed, **{key: value for key, value in item.items() if key not in {"id", "ordinal"}}}
                candidates.append({"ordinal": int(item["ordinal"]), "id": item["id"], "parameters": params})
            members.append({
                "member": f"{algorithm}_{feature}",
                "branch": "fingerprint",
                "family": algorithm,
                "feature": feature,
                "backend_policy": "cuda_required",
                "candidates": candidates,
            })

    graph_registry = supplement["gpu_candidate_registry"]["graph_architectures"]
    graph_fixed = graph_registry["shared_fixed_parameters"]
    for architecture in ("GCN", "GAT", "MPNN", "AttentiveFP"):
        candidates = []
        for item in graph_registry[architecture]:
            params = {
                "hidden_dim": item["hidden_dim"],
                "dropout": item["dropout"],
                "learning_rate": item["learning_rate"],
                "weight_decay": item["weight_decay"],
                "max_epochs": graph_fixed["max_epochs"],
                "patience": graph_fixed["patience"],
                "batch_size": graph_fixed["batch_size"],
                "single_oom_reduced_batch_size": graph_fixed["single_oom_reduced_batch_size"],
                "early_stop_validation_fraction": graph_fixed["early_stop_validation_fraction"],
                "early_stop_seed_candidates": graph_fixed["early_stop_seed_candidates"],
            }
            candidates.append({"ordinal": int(item["ordinal"]), "id": item["id"], "parameters": params})
        members.append({
            "member": architecture,
            "branch": "graph",
            "family": "graph",
            "architecture": architecture,
            "backend_policy": "cuda_required",
            "candidates": candidates,
        })

    for model_name in ("ChemBERTa", "MolFormer"):
        candidates = [
            {
                "ordinal": ordinal,
                "id": f"{model_name.lower()}_logistic_c{value:g}",
                "parameters": {"c_value": value, "max_iter": 50},
            }
            for ordinal, value in enumerate((0.01, 0.1, 1.0, 10.0, 100.0), start=1)
        ]
        members.append({
            "member": f"{model_name}_CUDA_Logistic",
            "branch": "language_model",
            "family": "language_model",
            "model_name": model_name,
            "backend_policy": "cuda_required",
            "candidates": candidates,
        })

    expected_order = supplement["final_primary_selection_and_refit"]["full_data_base_refit_jobs"]["member_order_zero_based"]
    by_name = {item["member"]: item for item in members}
    if set(by_name) != set(expected_order) or len(by_name) != 14:
        raise RuntimeError("Member registry does not match the E001 fourteen-member list")
    members = [by_name[name] for name in expected_order]
    for member in members:
        ids = [item["id"] for item in member["candidates"]]
        ordinals = [item["ordinal"] for item in member["candidates"]]
        if len(ids) != len(set(ids)) or ordinals != list(range(1, len(ids) + 1)):
            raise RuntimeError(f"Invalid candidate ID or ordinal registry: {member['member']}")
    expected_counts = {
        "RandomForest_ECFP4": 6,
        "RandomForest_MACCS": 6,
        "RBF_SVM_ECFP4": 8,
        "RBF_SVM_MACCS": 8,
        "XGBoost_ECFP4": 8,
        "XGBoost_MACCS": 8,
        "LightGBM_ECFP4": 8,
        "LightGBM_MACCS": 8,
        "GCN": 4,
        "GAT": 4,
        "MPNN": 4,
        "AttentiveFP": 4,
        "ChemBERTa_CUDA_Logistic": 5,
        "MolFormer_CUDA_Logistic": 5,
    }
    observed_counts = {item["member"]: len(item["candidates"]) for item in members}
    if observed_counts != expected_counts:
        raise RuntimeError(f"Candidate-count mismatch: {observed_counts}")
    return members


def make_row(counter: int, **values) -> dict:
    defaults = {
        "job_id": f"E001-J{counter:05d}",
        "planned_ordinal": counter,
        "phase": "",
        "scenario_id": "",
        "design_id": "",
        "outer_fold": "",
        "inner_fold": "",
        "stage": "",
        "member": "",
        "branch": "",
        "family": "",
        "candidate_id": "",
        "candidate_ordinal": "",
        "seed": "",
        "backend_policy": "",
        "parameters_json": "{}",
    }
    defaults.update(values)
    return defaults


def build_plan(protocol: dict, supplement: dict, members: list[dict]) -> list[dict]:
    rows: list[dict] = []
    counter = 0
    designs = protocol["split_policy"]["designs"]
    for scenario in protocol["inputs"]:
        scenario_id = scenario["scenario_id"]
        for design_index, design_id in enumerate(designs):
            for outer_fold in range(5):
                for member_index, member in enumerate(members):
                    member_seed = int(protocol["base_seed"] + design_index * 100000 + member_index * 1000)
                    for candidate in member["candidates"]:
                        for inner_fold in range(5):
                            counter += 1
                            rows.append(make_row(
                                counter,
                                phase="nested_cv",
                                scenario_id=scenario_id,
                                design_id=design_id,
                                outer_fold=outer_fold,
                                inner_fold=inner_fold,
                                stage="inner_candidate_fit",
                                member=member["member"],
                                branch=member["branch"],
                                family=member["family"],
                                candidate_id=candidate["id"],
                                candidate_ordinal=candidate["ordinal"],
                                seed=member_seed + inner_fold * 10,
                                backend_policy=member["backend_policy"],
                                parameters_json=parameters_json(candidate["parameters"]),
                            ))
                    for inner_fold in range(5):
                        counter += 1
                        rows.append(make_row(
                            counter,
                            phase="nested_cv",
                            scenario_id=scenario_id,
                            design_id=design_id,
                            outer_fold=outer_fold,
                            inner_fold=inner_fold,
                            stage="selected_member_crossfit",
                            member=member["member"],
                            branch=member["branch"],
                            family=member["family"],
                            candidate_id="__OUTER_SELECTED__",
                            seed=member_seed + inner_fold * 10,
                            backend_policy=member["backend_policy"],
                        ))
                    counter += 1
                    rows.append(make_row(
                        counter,
                        phase="nested_cv",
                        scenario_id=scenario_id,
                        design_id=design_id,
                        outer_fold=outer_fold,
                        stage="gpu_platt_calibration",
                        member=member["member"],
                        branch=member["branch"],
                        family="calibrator",
                        candidate_id="__OUTER_SELECTED__",
                        seed=member_seed + 500000,
                        backend_policy="cuda_required",
                        parameters_json=parameters_json({"max_iter": protocol["calibration"]["max_iter"]}),
                    ))
                    counter += 1
                    rows.append(make_row(
                        counter,
                        phase="nested_cv",
                        scenario_id=scenario_id,
                        design_id=design_id,
                        outer_fold=outer_fold,
                        stage="selected_member_outer_refit",
                        member=member["member"],
                        branch=member["branch"],
                        family=member["family"],
                        candidate_id="__OUTER_SELECTED__",
                        seed=member_seed + 900000,
                        backend_policy=member["backend_policy"],
                    ))

    final = supplement["final_primary_selection_and_refit"]
    scenario_id = final["scenario"]
    design_id = final["calibration_crossfit"]["design"]
    for member_index, member in enumerate(members):
        final_seed = int(protocol["base_seed"] + 2000000 + member_index * 1000)
        for fold in range(5):
            counter += 1
            rows.append(make_row(
                counter,
                phase="final_primary",
                scenario_id=scenario_id,
                design_id=design_id,
                outer_fold=fold,
                inner_fold=fold,
                stage="final_primary_crossfit",
                member=member["member"],
                branch=member["branch"],
                family=member["family"],
                candidate_id="__GLOBAL_SELECTED__",
                seed=final_seed + fold * 10,
                backend_policy=member["backend_policy"],
            ))
        counter += 1
        rows.append(make_row(
            counter,
            phase="final_primary",
            scenario_id=scenario_id,
            design_id=design_id,
            stage="final_primary_gpu_platt_calibration",
            member=member["member"],
            branch=member["branch"],
            family="calibrator",
            candidate_id="__GLOBAL_SELECTED__",
            seed=final_seed + 500000,
            backend_policy="cuda_required",
            parameters_json=parameters_json({"max_iter": protocol["calibration"]["max_iter"]}),
        ))
        counter += 1
        rows.append(make_row(
            counter,
            phase="final_primary",
            scenario_id=scenario_id,
            design_id=design_id,
            stage="final_primary_full_refit",
            member=member["member"],
            branch=member["branch"],
            family=member["family"],
            candidate_id="__GLOBAL_SELECTED__",
            seed=final_seed + 900000,
            backend_policy=member["backend_policy"],
        ))
    return rows


def validate_fingerprints(protocol: dict, supplement: dict) -> list[dict]:
    checks = []
    expected = {
        PROTOCOL_PATH: EXPECTED_PROTOCOL_SHA256,
        GO_FULL_PATH: EXPECTED_GO_FULL_SHA256,
        SUPPLEMENT_PATH: EXPECTED_SUPPLEMENT_SHA256,
    }
    for path, digest in expected.items():
        observed = sha256_file(path)
        checks.append({"path": path.relative_to(REPO_ROOT).as_posix(), "expected": digest, "observed": observed, "status": "PASS" if observed == digest else "FAIL"})
        if observed != digest:
            raise RuntimeError(f"Binding artifact SHA256 mismatch: {path}")
    index_path = REPO_ROOT / supplement["preserved_invariants"]["split_manifest_index_path"]
    if sha256_file(index_path) != EXPECTED_SPLIT_INDEX_SHA256:
        raise RuntimeError("Split-manifest index SHA256 mismatch")
    index = pd.read_csv(index_path)
    if len(index) != 10:
        raise RuntimeError("Expected ten split manifests")
    specs = {item["scenario_id"]: item for item in protocol["inputs"]}
    for spec in protocol["inputs"]:
        input_path = REPO_ROOT / spec["path"]
        observed = sha256_file(input_path)
        if observed != spec["sha256"]:
            raise RuntimeError(f"Locked input SHA256 mismatch: {spec['scenario_id']}")
        checks.append({"path": spec["path"], "expected": spec["sha256"], "observed": observed, "status": "PASS"})
    for row in index.itertuples(index=False):
        manifest_path = REPO_ROOT / row.manifest_path
        observed = sha256_file(manifest_path)
        if observed != row.manifest_sha256:
            raise RuntimeError(f"Split-manifest SHA256 mismatch: {row.manifest_path}")
        spec = specs[row.scenario_id]
        data = load_locked_scenario(REPO_ROOT / spec["path"], spec["scenario_id"], spec["sha256"], int(spec["rows"]))
        validation = validate_manifest(pd.read_csv(manifest_path), data, row.design_id)
        if validation["status"] != "PASS":
            raise RuntimeError(f"Split validation failed: {row.scenario_id} {row.design_id}")
        checks.append({"path": row.manifest_path, "expected": row.manifest_sha256, "observed": observed, "status": "PASS"})
    return checks


def main() -> int:
    started_at = datetime.now(timezone.utc).isoformat()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    approval = json.loads(GO_FULL_PATH.read_text(encoding="utf-8"))
    supplement = json.loads(SUPPLEMENT_PATH.read_text(encoding="utf-8"))
    if approval.get("full_training_authorized") is not True:
        raise RuntimeError("GO_FULL_TRAINING_V2 is not active")
    if supplement.get("status") != "APPROVED_EXECUTABLE_SUPPLEMENT":
        raise RuntimeError("E001 executable supplement is not approved")
    if supplement["authorization"]["formal_training_launch_authorized_after_no_fit_pass"] is not True:
        raise RuntimeError("E001 does not authorize launch after no-fit validation")

    fingerprint_checks = validate_fingerprints(protocol, supplement)
    members = candidate_registry(protocol, supplement)
    rows = build_plan(protocol, supplement, members)
    if len(rows) != 26498:
        raise RuntimeError(f"Whole-run planned count mismatch: {len(rows)}")
    if len({row["job_id"] for row in rows}) != len(rows):
        raise RuntimeError("Duplicate plan job IDs")
    counts = Counter((row["phase"], row["scenario_id"]) for row in rows)
    for scenario in (item["scenario_id"] for item in protocol["inputs"]):
        if counts[("nested_cv", scenario)] != 5280:
            raise RuntimeError(f"Scenario job count mismatch: {scenario}")
    if sum(value for (phase, _), value in counts.items() if phase == "nested_cv") != 26400:
        raise RuntimeError("Nested job total mismatch")
    if sum(value for (phase, _), value in counts.items() if phase == "final_primary") != 98:
        raise RuntimeError("Final-primary job total mismatch")
    if len(rows) > int(supplement["preserved_invariants"]["registered_fit_job_hard_ceiling"]):
        raise RuntimeError("Plan exceeds the registered hard ceiling")

    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PLAN_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plan_sha = sha256_file(PLAN_PATH)
    report = {
        "artifact": "TASK02_E001_NO_FIT_EXECUTION_PLAN_VALIDATION",
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "model_fit_executed": False,
        "formal_fit_jobs_attempted": 0,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "go_full_training_sha256": EXPECTED_GO_FULL_SHA256,
        "execution_supplement_sha256": EXPECTED_SUPPLEMENT_SHA256,
        "split_manifest_index_sha256": EXPECTED_SPLIT_INDEX_SHA256,
        "fingerprint_checks": fingerprint_checks,
        "candidate_counts": {item["member"]: len(item["candidates"]) for item in members},
        "member_count": len(members),
        "scenario_counts": {scenario: counts[("nested_cv", scenario)] for scenario in (item["scenario_id"] for item in protocol["inputs"])},
        "nested_cv_planned_jobs": 26400,
        "final_primary_planned_jobs": 98,
        "whole_run_planned_jobs": len(rows),
        "attempt_headroom": 27300 - len(rows),
        "hard_ceiling": 27300,
        "plan_manifest_path": PLAN_PATH.relative_to(REPO_ROOT).as_posix(),
        "plan_manifest_sha256": plan_sha,
        "plan_manifest_rows": len(rows),
        "task01_files_modified": False,
        "split_manifests_modified_or_regenerated": False,
        "external_validation_used": False,
        "mfh_rescreening_performed": False,
        "model_lock_changed_to_locked": False,
    }
    atomic_json(REPORT_PATH, report)
    progress = {
        "artifact_id": "TASK02-FULL-TRAINING-PROGRESS",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "state": "READY_TO_LAUNCH_NO_FIT_PLAN_PASS",
        "process_id": None,
        "training_started": False,
        "formal_fit_jobs_planned": 26498,
        "formal_fit_jobs_attempted": 0,
        "formal_fit_jobs_completed": 0,
        "fit_job_hard_ceiling": 27300,
        "attempt_headroom": 802,
        "progress_fraction": 0.0,
        "elapsed_hours": 0.0,
        "eta_hours": None,
        "planning_window_hours_after_start": [36, 60],
        "current_phase": None,
        "current_scenario": None,
        "current_design": None,
        "current_outer_fold": None,
        "current_member": None,
        "current_job_id": None,
        "last_completed_fit_at_utc": None,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "go_full_training_sha256": EXPECTED_GO_FULL_SHA256,
        "execution_supplement_sha256": EXPECTED_SUPPLEMENT_SHA256,
        "split_manifest_index_sha256": EXPECTED_SPLIT_INDEX_SHA256,
        "execution_plan_sha256": plan_sha,
        "no_fit_validation_report_path": REPORT_PATH.relative_to(REPO_ROOT).as_posix(),
        "no_fit_validation_status": "PASS",
        "blocker": None,
    }
    atomic_json(PROGRESS_PATH, progress)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
