#!/usr/bin/env python3
"""Read-only P002 revalidation of the existing split manifests."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
TASK_DIR = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]
sys.path.insert(0, str(TASK_DIR / "src"))

from split_protocol import load_locked_scenario, sha256_file, validate_manifest


PROTOCOL_PATH = TASK_DIR / "configs" / "registered_protocol_v4_hybrid_restricted.json"
P002_PATH = TASK_DIR / "TASK00_PROTOCOL_AMENDMENT_P002.json"
REPORT_PATH = TASK_DIR / "reports" / "split_manifest_revalidation_p002.json"


def main() -> int:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    approval = json.loads(P002_PATH.read_text(encoding="utf-8"))
    if sha256_file(P002_PATH) != protocol["task00_amendment_sha256"]:
        raise RuntimeError("P002 approval hash mismatch")
    if approval["status"] != "CONDITIONAL_APPROVAL" or approval["full_training_authorized"] is not False:
        raise RuntimeError("P002 scope is not the expected conditional limited-smoke authorization")

    index_path = REPO_ROOT / protocol["split_policy"]["manifest_index_path"]
    before_index_hash = sha256_file(index_path)
    required_index_hash = protocol["split_policy"]["required_manifest_index_sha256"]
    if before_index_hash != required_index_hash:
        raise RuntimeError("P002 split-manifest index fingerprint changed; regeneration is forbidden")
    index = pd.read_csv(index_path)
    if len(index) != 10:
        raise RuntimeError("P002 expected exactly ten scenario-design manifests")

    specs = {item["scenario_id"]: item for item in protocol["inputs"]}
    validations = []
    for row in index.itertuples(index=False):
        manifest_path = REPO_ROOT / row.manifest_path
        observed_hash = sha256_file(manifest_path)
        if observed_hash != row.manifest_sha256:
            raise RuntimeError(f"Manifest hash changed: {row.manifest_path}")
        spec = specs[row.scenario_id]
        if row.input_sha256 != spec["sha256"]:
            raise RuntimeError(f"Index input hash mismatch: {row.scenario_id}")
        data = load_locked_scenario(
            REPO_ROOT / spec["path"], spec["scenario_id"], spec["sha256"], int(spec["rows"])
        )
        manifest = pd.read_csv(manifest_path)
        result = validate_manifest(manifest, data, row.design_id)
        if result["status"] != "PASS":
            raise RuntimeError(f"Split revalidation failed: {row.scenario_id} {row.design_id}")
        validations.append(
            {
                "scenario_id": row.scenario_id,
                "design_id": row.design_id,
                "manifest_path": row.manifest_path,
                "registered_sha256": row.manifest_sha256,
                "observed_sha256": observed_hash,
                **result,
            }
        )

    after_index_hash = sha256_file(index_path)
    if after_index_hash != before_index_hash:
        raise RuntimeError("Read-only P002 revalidation unexpectedly changed the split index")
    report = {
        "artifact": "TASK02_P002_SPLIT_MANIFEST_REVALIDATION",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "operation": "READ_ONLY_REVALIDATION_NO_REGENERATION",
        "protocol_path": PROTOCOL_PATH.relative_to(REPO_ROOT).as_posix(),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "p002_path": P002_PATH.relative_to(REPO_ROOT).as_posix(),
        "p002_sha256": sha256_file(P002_PATH),
        "manifest_index_path": index_path.relative_to(REPO_ROOT).as_posix(),
        "required_manifest_index_sha256": required_index_hash,
        "observed_manifest_index_sha256_before": before_index_hash,
        "observed_manifest_index_sha256_after": after_index_hash,
        "scenario_design_count": len(validations),
        "all_manifest_hashes_unchanged": True,
        "silent_regeneration_performed": False,
        "all_overlap_and_coverage_checks_passed": True,
        "validations": validations,
        "model_training_executed": False,
        "task01_files_modified": False,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
