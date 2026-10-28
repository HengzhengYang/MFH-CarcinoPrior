#!/usr/bin/env python3
"""Generate immutable shared nested-CV manifests for all locked scenarios."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
TASK_DIR = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]
sys.path.insert(0, str(TASK_DIR / "src"))

from split_protocol import generate_manifest, load_locked_scenario, sha256_file, validate_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=TASK_DIR / "configs" / "registered_protocol_v3_gpu_only.json",
    )
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["hard_guards"]["full_training_authorized"] is not False:
        raise RuntimeError("Limited-smoke protocol unexpectedly authorizes full training")

    index_rows = []
    validations = []
    for input_spec in protocol["inputs"]:
        input_path = REPO_ROOT / input_spec["path"]
        data = load_locked_scenario(
            input_path,
            input_spec["scenario_id"],
            input_spec["sha256"],
            input_spec["rows"],
        )
        for design in protocol["split_designs"]:
            design_id = design["id"]
            manifest = generate_manifest(
                data=data,
                scenario_id=input_spec["scenario_id"],
                design_id=design_id,
                base_seed=protocol["base_seed"],
                outer_folds=design["outer_folds"],
                inner_folds=design["inner_folds"],
            )
            validation = validate_manifest(manifest, data, design_id)
            if validation["status"] != "PASS":
                raise RuntimeError(f"Split validation failed: {input_spec['scenario_id']} {design_id}")
            output_dir = TASK_DIR / "splits" / input_spec["scenario_id"] / design_id
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "split_manifest.csv"
            manifest.to_csv(output_path, index=False, lineterminator="\n")
            manifest_hash = sha256_file(output_path)
            relative_path = output_path.relative_to(REPO_ROOT).as_posix()
            validation.update(
                {
                    "scenario_id": input_spec["scenario_id"],
                    "design_id": design_id,
                    "manifest_path": relative_path,
                    "manifest_sha256": manifest_hash,
                }
            )
            validations.append(validation)
            index_rows.append(
                {
                    "scenario_id": input_spec["scenario_id"],
                    "role": input_spec["role"],
                    "design_id": design_id,
                    "input_sha256": input_spec["sha256"],
                    "model_units": len(data),
                    "outer_folds": design["outer_folds"],
                    "inner_folds": design["inner_folds"],
                    "manifest_path": relative_path,
                    "manifest_rows": len(manifest),
                    "manifest_sha256": manifest_hash,
                    "validation_status": "PASS",
                }
            )

    index = pd.DataFrame(index_rows).sort_values(["role", "scenario_id", "design_id"])
    index_path = TASK_DIR / "splits" / "split_manifest_index.csv"
    index.to_csv(index_path, index=False, lineterminator="\n")
    report = {
        "artifact": "TASK02_SHARED_SPLIT_VALIDATION",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "protocol_path": args.protocol.relative_to(REPO_ROOT).as_posix(),
        "protocol_sha256": sha256_file(args.protocol),
        "manifest_index_path": index_path.relative_to(REPO_ROOT).as_posix(),
        "manifest_index_sha256": sha256_file(index_path),
        "scenario_design_count": len(validations),
        "validations": validations,
        "model_training_executed": False,
    }
    report_path = TASK_DIR / "reports" / "split_manifest_validation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
