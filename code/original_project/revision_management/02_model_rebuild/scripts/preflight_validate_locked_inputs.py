#!/usr/bin/env python3
"""Read-only validation of Task 01 locked model inputs.

The script reads only Task 01 artifacts and writes its report/log exclusively
under revision_management/02_model_rebuild. It does not generate splits or fit
models.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_COLUMNS = {
    "scenario_id",
    "model_unit_id",
    "cv_group_id",
    "final_label",
    "canonical_isomeric_smiles",
    "canonical_nonisomeric_smiles",
    "decision_status",
    "downstream_training_authorized",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    # Task 01 frozen CSVs are UTF-8 with a BOM. utf-8-sig is also safe for
    # ordinary UTF-8 and prevents the first field from becoming \ufeffscenario_id.
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def validate_model_input(path: Path, spec: dict, scenario_spec: dict) -> dict:
    columns, rows = load_csv(path)
    label_counts = defaultdict(int)
    group_labels: dict[str, set[str]] = defaultdict(set)
    model_units = []
    scenario_ids = set()
    unauthorized = 0
    unconfirmed = 0

    for row in rows:
        label_counts[row["final_label"]] += 1
        group_labels[row["cv_group_id"]].add(row["final_label"])
        model_units.append(row["model_unit_id"])
        scenario_ids.add(row["scenario_id"])
        unauthorized += row["downstream_training_authorized"] != "1"
        unconfirmed += row["decision_status"] != "CONFIRMED"

    multi_unit_groups = sum(1 for group in group_labels if sum(1 for row in rows if row["cv_group_id"] == group) > 1)
    max_group_size = max(
        (sum(1 for row in rows if row["cv_group_id"] == group) for group in group_labels),
        default=0,
    )
    opposite_label_groups = sum(len(labels) > 1 for labels in group_labels.values())

    checks = {
        "sha256_matches_lock": sha256(path) == spec["sha256"],
        "size_matches_lock": path.stat().st_size == spec["size_bytes"],
        "row_count_matches_lock": len(rows) == spec["row_count"],
        "column_count_matches_lock": len(columns) == spec["column_count"],
        "required_columns_present": REQUIRED_COLUMNS.issubset(columns),
        "single_expected_scenario": scenario_ids == {scenario_spec["scenario_id"]},
        "model_unit_id_unique": len(model_units) == len(set(model_units)),
        "no_opposite_label_cv_group": opposite_label_groups == 0,
        "all_rows_authorized": unauthorized == 0,
        "all_decisions_confirmed": unconfirmed == 0,
        "positive_count_matches": label_counts["1"] == scenario_spec["positive_model_units"],
        "negative_count_matches": label_counts["0"] == scenario_spec["negative_model_units"],
        "model_unit_count_matches": len(rows) == scenario_spec["final_model_units"],
    }

    return {
        "path": str(path),
        "scenario_id": next(iter(scenario_ids), None),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        "row_count": len(rows),
        "column_count": len(columns),
        "positive": label_counts["1"],
        "negative": label_counts["0"],
        "cv_groups": len(group_labels),
        "multi_unit_cv_groups": multi_unit_groups,
        "max_units_per_cv_group": max_group_size,
        "opposite_label_cv_groups": opposite_label_groups,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="MFH-CarcinoPrior-revision root",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    task01 = repo_root / "revision_management" / "01_data_audit"
    task02 = repo_root / "revision_management" / "02_model_rebuild"
    dataset_lock_path = task01 / "DATASET_LOCK.json"
    approval_path = task01 / "TASK00_LOCK_APPROVAL.json"
    manifest_path = task01 / "frozen" / "FROZEN_INPUT_MANIFEST.csv"

    dataset_lock = load_json(dataset_lock_path)
    approval = load_json(approval_path)
    scenario_specs = {item["scenario_id"]: item for item in dataset_lock["scenario_counts"]}
    model_specs = [
        item
        for item in dataset_lock["outputs"]
        if Path(item["path"]).name.startswith("model_units_")
    ]

    approval_checks = {
        "dataset_lock_status_locked": dataset_lock.get("status") == "LOCKED",
        "dataset_training_authorized": dataset_lock.get("downstream_training_authorized") is True,
        "task00_approval_status_locked": approval.get("status") == "LOCKED",
        "task02_use_authorized": approval.get("authorization", {}).get("task02_dataset_use_authorized") is True,
        "dataset_lock_sha256_matches_task00": sha256(dataset_lock_path)
        == approval["finalized_locks"]["dataset_lock_sha256"],
        "manifest_sha256_matches_task00": sha256(manifest_path)
        == approval["frozen_payload"]["manifest_sha256"],
        "exactly_five_model_inputs_registered": len(model_specs) == 5,
    }

    input_results = []
    for spec in model_specs:
        path = repo_root / spec["path"]
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            first = next(reader)
        scenario_id = first["scenario_id"]
        input_results.append(validate_model_input(path, spec, scenario_specs[scenario_id]))

    passed = all(approval_checks.values()) and all(item["status"] == "PASS" for item in input_results)
    completed_at = datetime.now(timezone.utc).isoformat()
    report = {
        "artifact_id": "MODEL-001-PREFLIGHT",
        "task": "02_model_rebuild",
        "completed_at_utc": completed_at,
        "status": "PASS" if passed else "FAIL",
        "training_executed": False,
        "task01_files_modified": False,
        "approval_checks": approval_checks,
        "inputs": input_results,
    }

    report_path = task02 / "reports" / "preflight_validation.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    log_path = task02 / "logs" / "preflight_validation.log"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"{completed_at} status={report['status']} "
            f"registered_inputs={len(model_specs)} training_executed=false\n"
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
