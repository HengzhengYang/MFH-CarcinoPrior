#!/usr/bin/env python3
"""D113 V3.2 fail-closed production runner candidate.

Import is inert and standard-library-only. Execution requires a future exact
Task00 single-use GO. D113 authorizes preparation and synthetic/static tests
only; it does not authorize invoking this runner on the frozen MFH inputs.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve()
V3_2_ROOT = HERE.parents[1]
REPO_ROOT_DEFAULT = HERE.parents[4]


PROTOCOL_ID = "PROTOCOL_TASK04_MFH_SCREENING_V3_CANDIDATE"
MANIFEST_REL = "revision_management/04_mfh_rescreening/v3_2/manifests/TASK04_EXECUTION_MANIFEST_V3_2_CANDIDATE.json"
ENVIRONMENT_REL = "revision_management/04_mfh_rescreening/v3/environment/TASK04_RUNTIME_ENVIRONMENT_LOCK_V3_CANDIDATE.json"
CORE_REL = "revision_management/04_mfh_rescreening/v3_1/src/task04_v3_1_core.py"
AD_REFERENCE_SEAL = "AD_REFERENCE_SEAL.json"
OPERATION_LEDGER = "OPERATION_LEDGER.jsonl"
PREDICTION_SEAL = "PREDICTION_SEAL"
STAGING = "STAGING"
FAILURE_PACKAGE = "FAILURE_PACKAGE"
OUTPUT_MANIFEST = "OUTPUT_MANIFEST.json"


class ContractError(RuntimeError):
    """Fail-closed control-contract error usable before scientific imports."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_bytes(path, json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8") + b"\n")


def _publish_noreplace(staging: Path, destination: Path) -> None:
    if not staging.is_dir() or destination.exists():
        raise FileExistsError(str(destination))
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.renameat2(-100, os.fsencode(staging), -100, os.fsencode(destination), 1)
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))
    _fsync_directory(destination.parent)


def _append_hash_chained_event(path: Path, event: dict[str, Any]) -> dict[str, Any]:
    previous_sha = "0" * 64
    sequence = 1
    if path.exists():
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        previous_sha = events[-1]["event_sha256"]
        sequence = int(events[-1]["sequence"]) + 1
    body = {"sequence": sequence, "previous_event_sha256": previous_sha, **event}
    body["event_sha256"] = sha256_bytes(_canonical_json_bytes(body))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(_canonical_json_bytes(body))
        handle.flush()
        os.fsync(handle.fileno())
    return body


def _partial_staging_manifest(staging: Path) -> dict[str, Any]:
    files = []
    if staging.exists():
        for path in sorted(value for value in staging.rglob("*") if value.is_file()):
            files.append({"path": path.relative_to(staging).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"file_count": len(files), "files": files}


def _build_output_manifest(package_root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(value for value in package_root.rglob("*") if value.is_file() and value.name != OUTPUT_MANIFEST):
        files.append({"path": path.relative_to(package_root).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"manifest_version": "TASK04_ATOMIC_PACKAGE_V3_2", "file_count": len(files), "files": files}


def _verify_output_manifest(package_root: Path, manifest: dict[str, Any]) -> None:
    if manifest.get("file_count") != len(manifest.get("files", [])):
        raise ContractError("output manifest count mismatch")
    for item in manifest["files"]:
        path = package_root / item["path"]
        if not path.is_file() or path.stat().st_size != item["size_bytes"] or sha256_file(path) != item["sha256"]:
            raise ContractError(f"output manifest mismatch: {item['path']}")


def _seal_read_only_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        os.chmod(path, 0o555 if path.is_dir() else 0o444)
    os.chmod(root, 0o555)


def _authoritative_terminal_kind(run_root: Path) -> str | None:
    result_exists = (run_root / "RESULT_PACKAGE").is_dir()
    failure_exists = (run_root / "FAILURE_PACKAGE").is_dir()
    if result_exists and failure_exists:
        raise ContractError("contradictory result and failure terminal packages")
    if result_exists:
        return "RESULT"
    if failure_exists:
        return "FAILURE"
    return None


def atomic_claim_run_root(run_root: Path) -> None:
    """Atomically consume a fresh Task00 run ID before any fallible hard gate."""
    if not run_root.parent.is_dir():
        raise ContractError("canonical V3.2 runs parent must be pre-created by the locked package")
    os.mkdir(run_root)


def build_rdkit_version_evidence(expected_raw: str, module_raw: str, distribution_raw: str) -> dict[str, Any]:
    """Return D113/B18 raw and PEP 440-normalized RDKit version evidence."""
    from packaging.version import InvalidVersion, Version

    values = {"environment_lock_raw": expected_raw, "rdkit_module_raw": module_raw, "distribution_raw": distribution_raw}
    try:
        normalized = {key.replace("_raw", "_pep440_normalized"): str(Version(value)) for key, value in values.items()}
    except InvalidVersion as error:
        raise ContractError(f"invalid RDKit PEP 440 version: {values}") from error
    evidence = {**values, **normalized}
    evidence["normalized_all_equal"] = len(set(normalized.values())) == 1
    evidence["d113_allowed_equivalent_raw_triplet"] = (
        expected_raw == "2025.3.6" and module_raw == "2025.03.6" and distribution_raw == "2025.3.6"
    )
    evidence["gate_pass"] = bool(evidence["normalized_all_equal"] and evidence["d113_allowed_equivalent_raw_triplet"])
    return evidence


def validate_rdkit_version_evidence(evidence: dict[str, Any]) -> None:
    if evidence.get("gate_pass") is not True:
        raise ContractError(f"RDKit version hard gate failed: {evidence}")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ContractError(f"JSON object required: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def ordered_text_sha256(values: list[str]) -> str:
    return sha256_bytes("\n".join(values).encode("utf-8"))


def validate_preclaim_authorization(authorization: dict[str, Any], repo_root: Path) -> tuple[str, Path]:
    exact = {
        "authorization_type": "GO_TASK04_MFH_SCREENING",
        "status": "AUTHORIZED_SINGLE_USE_NO_RETRY",
        "issued_by": "Task00",
        "task_id": "04",
        "protocol_id": PROTOCOL_ID,
        "route_id": "E005_THREE_MEMBER_PARSIMONY_PRIMARY",
        "model_and_calibrator_loading_authorized": True,
        "feature_extraction_on_real_mfh_authorized": True,
        "real_mfh_inference_authorized": True,
        "applicability_domain_uncertainty_and_priority_authorized": True,
        "threshold_change_authorized": False,
        "screening_lock_creation_authorized": False,
        "retry_authorized": False,
        "automatic_result_acceptance": False,
        "external_or_isscan_access_authorized": False,
        "manuscript_editing_authorized": False,
        "docking_or_molecular_dynamics_authorized": False,
    }
    for key, expected in exact.items():
        if authorization.get(key) != expected:
            raise ContractError(f"authorization mismatch: {key}")
    if not isinstance(authorization.get("execution_manifest_sha256"), str) or len(authorization["execution_manifest_sha256"]) != 64:
        raise ContractError("Task00 execution manifest SHA256 required")
    if not isinstance(authorization.get("bound_hashes"), dict):
        raise ContractError("Task00 bound_hashes object required")
    run_id = authorization.get("run_id")
    if not isinstance(run_id, str) or not run_id.startswith("TASK04-MFH-SCREENING-V3-2-"):
        raise ContractError("Task00 exact V3.2 run ID required")
    output_value = authorization.get("output_root")
    if not isinstance(output_value, str) or not output_value:
        raise ContractError("Task00 exact output root required")
    output_root = (repo_root / output_value).resolve()
    allowed_parent = (repo_root / "revision_management/04_mfh_rescreening/v3_2/runs").resolve()
    if output_root.parent != allowed_parent or output_root.name != run_id:
        raise ContractError("output root must be the exact Task04 V3.2 run root")
    return run_id, output_root


def validate_postclaim_authorization_bindings(authorization: dict[str, Any], manifest: dict[str, Any], repo_root: Path) -> None:
    if authorization.get("execution_manifest_sha256") != sha256_file(repo_root / MANIFEST_REL):
        raise ContractError("execution manifest authorization hash mismatch")
    if authorization.get("bound_hashes") != manifest.get("execution_bindings"):
        raise ContractError("authorization bound_hashes differ from reviewed execution bindings")


def verify_all_bindings(manifest: dict[str, Any], repo_root: Path) -> None:
    for path_value, expected in manifest["execution_bindings"].items():
        path = Path(path_value) if Path(path_value).is_absolute() else repo_root / path_value
        if not path.is_file() or sha256_file(path) != expected:
            raise ContractError(f"execution binding mismatch: {path_value}")


def postclaim_environment_gate(
    environment: dict[str, Any], output_parent: Path, limits: dict[str, int],
    evidence_path: Path, evidence_out: dict[str, Any],
) -> tuple[Any, Any, Any, Any]:
    # Imports are package/runtime probes only; no scientific object is opened.
    import joblib
    import numpy as np
    import pandas as pd
    import psutil
    import rdkit
    import sklearn
    import torch
    import transformers

    expected_python = tuple(environment["python"]["version_info"])
    observed_versions = {
        "joblib": joblib.__version__, "numpy": np.__version__, "pandas": pd.__version__,
        "psutil": psutil.__version__, "rdkit": rdkit.__version__,
        "scikit_learn": sklearn.__version__, "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    rdkit_evidence = build_rdkit_version_evidence(
        str(environment["packages"]["rdkit"]), str(rdkit.__version__),
        str(importlib.metadata.version("rdkit")),
    )
    evidence_out.clear()
    evidence_out.update({
        "schema_version": "1.0",
        "evidence_type": "TASK04_V3_2_POSTCLAIM_ENVIRONMENT_VERSION_EVIDENCE",
        "python_expected": list(expected_python),
        "python_observed": list(sys.version_info[:3]),
        "python_executable_expected_sha256": environment["python"]["executable_sha256"],
        "python_executable_observed_sha256": sha256_file(Path(sys.executable)),
        "package_expected_raw": environment["packages"],
        "package_observed_raw": observed_versions,
        "rdkit": rdkit_evidence,
        "other_package_raw_exact": all(
            observed_versions[name] == expected for name, expected in environment["packages"].items() if name != "rdkit"
        ),
    })
    atomic_write_json(evidence_path, evidence_out)
    if tuple(sys.version_info[:3]) != expected_python:
        raise ContractError("Python version hard gate failed")
    if evidence_out["python_executable_observed_sha256"] != evidence_out["python_executable_expected_sha256"]:
        raise ContractError("Python executable hash hard gate failed")
    validate_rdkit_version_evidence(rdkit_evidence)
    if not evidence_out["other_package_raw_exact"]:
        raise ContractError(f"non-RDKit package version hard gate failed: {observed_versions}")
    cuda_available = bool(torch.cuda.is_available())
    evidence_out["cuda"] = {
        "available": cuda_available,
        "runtime_expected": environment["cuda"]["runtime"],
        "runtime_observed": torch.version.cuda,
    }
    atomic_write_json(evidence_path, evidence_out)
    if not cuda_available or torch.version.cuda != environment["cuda"]["runtime"]:
        raise ContractError("CUDA availability/runtime hard gate failed")
    observed_device_name = torch.cuda.get_device_name(0)
    observed_capability = list(torch.cuda.get_device_capability(0))
    properties = torch.cuda.get_device_properties(0)
    evidence_out["cuda"].update({
        "device_name_expected": environment["cuda"]["device_name"],
        "device_name_observed": observed_device_name,
        "compute_capability_expected": environment["cuda"]["compute_capability"],
        "compute_capability_observed": observed_capability,
        "total_memory_bytes_expected": int(environment["cuda"]["total_memory_bytes"]),
        "total_memory_bytes_observed": int(properties.total_memory),
    })
    atomic_write_json(evidence_path, evidence_out)
    if observed_device_name != environment["cuda"]["device_name"]:
        raise ContractError("GPU identity hard gate failed")
    if observed_capability != environment["cuda"]["compute_capability"]:
        raise ContractError("GPU compute capability hard gate failed")
    if int(properties.total_memory) != int(environment["cuda"]["total_memory_bytes"]):
        raise ContractError("GPU memory identity hard gate failed")
    torch.cuda.reset_peak_memory_stats(0)
    if int(torch.cuda.max_memory_allocated(0)) > limits["vram_bytes"]:
        raise ContractError("postclaim allocated-VRAM hard gate failed")
    free_storage = int(shutil.disk_usage(output_parent).free)
    evidence_out["resource_preflight"] = {
        "storage_free_bytes_observed": free_storage,
        "storage_free_bytes_required": int(limits["storage_bytes"]),
        "vram_capacity_bytes_observed": int(properties.total_memory),
        "vram_capacity_bytes_required": int(limits["vram_bytes"]),
    }
    atomic_write_json(evidence_path, evidence_out)
    if free_storage < limits["storage_bytes"]:
        raise ContractError("postclaim free-storage hard gate failed")
    if int(properties.total_memory) < limits["vram_bytes"]:
        raise ContractError("postclaim GPU capacity hard gate failed")
    return np, pd, torch, psutil


def publish_failure_package_v3_2(
    run_root: Path,
    staging: Path,
    ledger_path: Path,
    limits: dict[str, int],
    resource_snapshots: list[dict[str, Any]],
    claim_evidence: dict[str, Any],
    failed_stage: str,
    environment_evidence: dict[str, Any],
    error: BaseException,
    traceback_text: str,
) -> None:
    """Publish one complete immutable terminal for every postclaim exception."""
    terminal = _authoritative_terminal_kind(run_root)
    if terminal == "RESULT":
        return
    if terminal == "FAILURE":
        raise ContractError("failure terminal already committed")
    failure_stage = run_root / ".FAILURE_PACKAGE_STAGING"
    failure_stage.mkdir(exist_ok=False)
    atomic_write_json(failure_stage / "ATTEMPT_CLAIM_EVIDENCE.json", claim_evidence)
    atomic_write_json(failure_stage / "FAILURE.json", {
        "run_id": claim_evidence["run_id"],
        "state": "HARD_FAILED_RUN_ID_CONSUMED_NO_RETRY",
        "failed_stage": failed_stage,
        "error_type": type(error).__name__,
        "error": str(error),
        "screening_lock_created": False,
        "authoritative_terminal_kind": "FAILURE",
    })
    atomic_write_bytes(failure_stage / "TRACEBACK.txt", traceback_text.encode("utf-8"))
    atomic_write_json(failure_stage / "PARTIAL_STAGING_MANIFEST.json", _partial_staging_manifest(staging))
    if ledger_path.is_file():
        shutil.copy2(ledger_path, failure_stage / "OPERATION_LEDGER.jsonl")
    else:
        _append_hash_chained_event(failure_stage / "OPERATION_LEDGER.jsonl", {
            "operation": "postclaim_failure_before_operation_ledger_initialization",
            "run_id": claim_evidence["run_id"],
            "failed_stage": failed_stage,
        })
    atomic_write_json(failure_stage / "RESOURCE_LEDGER.json", {
        "limits": limits,
        "snapshots": resource_snapshots,
        "resource_guard_initialized": bool(resource_snapshots),
    })
    atomic_write_json(failure_stage / "ENVIRONMENT_VERSION_EVIDENCE.json", environment_evidence or {
        "status": "NOT_REACHED_OR_NOT_AVAILABLE",
        "failed_stage": failed_stage,
    })
    staging_resolved = staging.resolve()
    run_resolved = run_root.resolve()
    if run_resolved not in staging_resolved.parents or staging.name != STAGING:
        raise ContractError("refuse to remove non-registered staging directory")
    if staging.exists():
        shutil.rmtree(staging)
    if staging.exists():
        raise ContractError("validated STAGING tree remains after failure cleanup")
    atomic_write_json(failure_stage / "TERMINAL_STATUS.json", {
        "run_id": claim_evidence["run_id"],
        "state": "HARD_FAILED_RUN_ID_CONSUMED_NO_RETRY",
        "authoritative_terminal_kind": "FAILURE",
        "failed_stage": failed_stage,
        "whole_run_retry": 0,
        "screening_lock_created": False,
        "staging_exists_at_publication": False,
        "scientific_files_outside_failure_package": 0,
    })
    manifest = _build_output_manifest(failure_stage)
    atomic_write_json(failure_stage / OUTPUT_MANIFEST, manifest)
    _verify_output_manifest(failure_stage, manifest)
    _seal_read_only_tree(failure_stage)
    _publish_noreplace(failure_stage, run_root / FAILURE_PACKAGE)


def gpu_platt_apply(calibrator_path: Path, raw_scores: Any, member: str, torch: Any, np: Any, ledger_path: Path) -> Any:
    device = torch.device("cuda:0")
    checkpoint = torch.load(calibrator_path, map_location=device, weights_only=False)
    append_hash_chained_event(ledger_path, {"operation": "calibrator_load", "member": member, "device": "cuda:0"})
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict):
        raise ContractError(f"sealed Platt state missing: {member}")
    module = torch.nn.Linear(1, 1, device=device, dtype=torch.float32)
    module.load_state_dict(state, strict=True)
    if any(parameter.device.type != "cuda" for parameter in module.parameters()):
        raise ContractError("GPU Platt CPU fallback detected")
    tensor = torch.as_tensor(np.asarray(raw_scores, dtype=np.float32).reshape(-1, 1), device=device)
    with torch.no_grad():
        probability_tensor = torch.sigmoid(module(tensor).squeeze(1))
    if probability_tensor.device.type != "cuda" or not bool(torch.isfinite(probability_tensor).all().item()):
        raise ContractError("GPU Platt application failed device/finiteness gate")
    append_hash_chained_event(ledger_path, {"operation": "gpu_platt_application", "member": member, "device": "cuda:0", "rows": int(tensor.shape[0])})
    return probability_tensor.to(torch.float64).cpu().numpy()


def canonical_e005_probability(member_probabilities_in_registered_order: dict[str, Any], np: Any) -> Any:
    arrays = [np.asarray(member_probabilities_in_registered_order[name], dtype=np.float64) for name in MEMBER_ORDER]
    # Do not replace: D096/D097 exact registered-order float64 matrix mean.
    return np.mean(np.vstack(arrays), axis=0, dtype=np.float64)


def chemical_space_coordinates(training_matrix: Any, candidate_matrix: Any, np: Any) -> tuple[Any, Any, dict[str, Any]]:
    """Frozen training-only full-SVD PCA with deterministic component signs."""
    train = np.asarray(training_matrix, dtype=np.float64)
    candidate = np.asarray(candidate_matrix, dtype=np.float64)
    center = np.mean(train, axis=0, dtype=np.float64)
    centered = train - center
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].copy()
    for index in range(2):
        pivot = max(range(components.shape[1]), key=lambda column: (abs(float(components[index, column])), -column))
        if components[index, pivot] < 0:
            components[index] *= -1.0
    train_coordinates = centered @ components.T
    candidate_coordinates = (candidate - center) @ components.T
    seal = {
        "method": "TRAINING_ONLY_ECFP4_FULL_SVD_PCA_V1",
        "center_float64_c_sha256": float64_array_sha256(center.reshape(-1)),
        "components_float64_c_sha256": sha256_bytes(np.ascontiguousarray(components, dtype=np.float64).tobytes()),
        "singular_values_first_two": [float(value) for value in singular_values[:2]],
        "sign_rule": "largest-absolute-loading-positive",
    }
    return train_coordinates, candidate_coordinates, seal


def execute(repo_root: Path, authorization_path: Path) -> None:
    start = time.monotonic()
    authorization = read_json(authorization_path)
    run_id, run_root = validate_preclaim_authorization(authorization, repo_root)
    if run_root.exists():
        raise ContractError("single-use run root already exists; retry forbidden")

    claim_evidence = {
        "run_id": run_id,
        "state": "CLAIMED_SINGLE_USE_NO_RETRY",
        "authorization_path": str(authorization_path),
        "whole_run_retry": 0,
    }
    staging = run_root / STAGING
    result_stage = staging / "RESULT_PACKAGE"
    control = result_stage / "CONTROL"
    ledger_path = control / OPERATION_LEDGER
    guard = None
    limits: dict[str, int] = {}
    environment_evidence: dict[str, Any] = {}
    failed_stage = "atomic_run_claim"

    # D113/B19: the atomic directory creation is the irreversible claim. No
    # execution-manifest hash, execution binding, package, CUDA/GPU, storage,
    # scientific input, model or calibrator access occurs before this point.
    atomic_claim_run_root(run_root)
    try:
        failed_stage = "attempt_claim_evidence"
        atomic_write_json(run_root / "ATTEMPT_CLAIM.json", claim_evidence)

        failed_stage = "execution_manifest_authorization_binding"
        manifest = read_json(repo_root / MANIFEST_REL)
        limits = dict(manifest["resource_hard_limits"])
        claim_evidence.update({
            "authorization_sha256": sha256_file(authorization_path),
            "execution_manifest_sha256": sha256_file(repo_root / MANIFEST_REL),
        })
        atomic_write_json(run_root / "ATTEMPT_CLAIM.json", claim_evidence)
        validate_postclaim_authorization_bindings(authorization, manifest, repo_root)

        failed_stage = "execution_binding_hashes"
        verify_all_bindings(manifest, repo_root)

        failed_stage = "locked_scientific_core_import"
        sys.path.insert(0, str((repo_root / CORE_REL).parent))
        import task04_v3_1_core as locked_core
        for name in (
            "LOCKED_THRESHOLD", "MEMBER_ORDER", "ResourceGuard",
            "append_hash_chained_event", "build_output_manifest",
            "canonical_equal_member_probability", "float64_array_sha256",
            "publish_result_package_v3", "rank_source_verification_priority",
            "rank_structure_verification_priority", "seal_member_prediction",
            "verify_operation_ledger", "verify_output_manifest",
            "verify_prediction_seals", "uncertainty_warning_flags",
            "validate_molformer_extraction_metadata",
        ):
            globals()[name] = getattr(locked_core, name)

        failed_stage = "postclaim_environment_python_packages"
        environment = read_json(repo_root / ENVIRONMENT_REL)
        control.mkdir(parents=True)
        np, pd, torch, psutil = postclaim_environment_gate(
            environment, run_root.parent, limits,
            control / "ENVIRONMENT_VERSION_EVIDENCE.json", environment_evidence,
        )

        failed_stage = "resource_guard_initialization"
        guard = ResourceGuard(start, staging, limits, psutil.Process(), torch)
        append_hash_chained_event(ledger_path, {"operation": "run_claim", "run_id": run_id})
        guard.check("post_claim")
        failed_stage = "scientific_execution"
        paths = manifest["paths"]
        structures = pd.read_csv(repo_root / paths["structure_input"], dtype=str)
        pairs = pd.read_csv(repo_root / paths["pair_provenance"], dtype=str)
        training = pd.read_csv(repo_root / paths["training_overlap_reference"], dtype=str)
        if (len(structures), len(pairs), len(training)) != (5003, 11703, 1494):
            raise ContractError("accepted input counts changed")
        ordered_ids = structures["screening_structure_id"].tolist()
        ordered_id_sha = ordered_text_sha256(ordered_ids)
        append_hash_chained_event(ledger_path, {"operation": "accepted_input_read", "structures": 5003, "pairs": 11703, "training": 1494})

        from rdkit import Chem, DataStructs
        from rdkit.Chem import MACCSkeys, rdFingerprintGenerator

        mols = [Chem.MolFromSmiles(value) for value in structures["standardized_smiles"]]
        training_mols = [Chem.MolFromSmiles(value) for value in training["canonical_isomeric_smiles"]]
        if any(mol is None for mol in mols) or any(mol is None for mol in training_mols):
            raise ContractError("structure parsing failed")
        candidate_noniso = [Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False) for mol in mols]
        exact_map: dict[str, list[str]] = {}
        cv_map: dict[str, list[str]] = {}
        for row in training.itertuples(index=False):
            exact_map.setdefault(row.standardized_inchikey, []).append(row.model_unit_id)
            cv_map.setdefault(row.canonical_nonisomeric_smiles, []).append(row.cv_group_id)
        overlap_rows = []
        overlap_class = []
        for index, row in structures.iterrows():
            exact = sorted(exact_map.get(row["standardized_inchikey"], []))
            cv_groups = sorted(set(cv_map.get(candidate_noniso[index], [])))
            classification = "EXACT_STANDARDIZED_STRUCTURE_OVERLAP" if exact else "CV_GROUP_STRUCTURE_OVERLAP" if cv_groups else "STRUCTURE_UNSEEN"
            overlap_class.append(classification)
            if classification != "STRUCTURE_UNSEEN":
                overlap_rows.append({
                    "screening_structure_id": row["screening_structure_id"],
                    "training_overlap_class": classification,
                    "matched_training_model_unit_ids_json": json.dumps(exact, separators=(",", ":")),
                    "matched_training_cv_group_ids_json": json.dumps(cv_groups, separators=(",", ":")),
                })

        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        candidate_ecfp4 = [generator.GetFingerprint(mol) for mol in mols]
        training_ecfp4 = [generator.GetFingerprint(mol) for mol in training_mols]
        candidate_matrix = np.vstack([np.asarray(fp, dtype=np.uint8) for fp in candidate_ecfp4])
        training_matrix = np.vstack([np.asarray(fp, dtype=np.uint8) for fp in training_ecfp4])
        loo_nearest = []
        for index, fp in enumerate(training_ecfp4):
            similarity = list(DataStructs.BulkTanimotoSimilarity(fp, training_ecfp4))
            similarity[index] = -1.0
            loo_nearest.append(max(similarity))
        ad_threshold = float(np.quantile(np.asarray(loo_nearest, dtype=np.float64), 0.05, method="linear"))
        nearest_similarity = [max(DataStructs.BulkTanimotoSimilarity(fp, training_ecfp4)) for fp in candidate_ecfp4]
        train_xy, candidate_xy, chemical_seal = chemical_space_coordinates(training_matrix, candidate_matrix, np)

        ad_rows = [
            {
                "model_unit_id": row.model_unit_id,
                "cv_group_id": row.cv_group_id,
                "loo_nearest_ecfp4_tanimoto": format(float(loo_nearest[index]), ".17g"),
                "sealed_ad_threshold": format(ad_threshold, ".17g"),
            }
            for index, row in enumerate(training.itertuples(index=False))
        ]
        ad_path = result_stage / "applicability_domain_reference_candidate.csv"
        write_csv(ad_path, list(ad_rows[0]), ad_rows)
        chemical_rows = []
        for index, row in enumerate(training.itertuples(index=False)):
            chemical_rows.append({"population": "TRAINING_REFERENCE", "entity_id": row.model_unit_id, "pc1": format(float(train_xy[index, 0]), ".17g"), "pc2": format(float(train_xy[index, 1]), ".17g")})
        for index, row in enumerate(structures.itertuples(index=False)):
            chemical_rows.append({"population": "MFH_SCREENING_CANDIDATE", "entity_id": row.screening_structure_id, "pc1": format(float(candidate_xy[index, 0]), ".17g"), "pc2": format(float(candidate_xy[index, 1]), ".17g")})
        chemical_path = result_stage / "chemical_space_coordinates_candidate.csv"
        write_csv(chemical_path, list(chemical_rows[0]), chemical_rows)

        ad_seal_payload = {
            "seal_type": "TASK04_AD_REFERENCE_SEAL_V2",
            "created_before_any_member_model_load": True,
            "training_rows": 1494,
            "candidate_rows": 5003,
            "ecfp4": {"radius": 2, "n_bits": 2048, "similarity": "Tanimoto"},
            "quantile": {"q": 0.05, "method": "linear", "threshold_float64_hex": np.float64(ad_threshold).hex()},
            "training_loo_float64_c_sha256": float64_array_sha256(loo_nearest),
            "applicability_domain_reference_candidate_sha256": sha256_file(ad_path),
            "chemical_space_coordinates_candidate_sha256": sha256_file(chemical_path),
            "chemical_space_method": chemical_seal,
        }
        atomic_write_json(control / AD_REFERENCE_SEAL, ad_seal_payload)
        append_hash_chained_event(ledger_path, {"operation": "ad_reference_seal", "sha256": sha256_file(control / AD_REFERENCE_SEAL)})
        guard.check("AD_REFERENCE_SEAL_BEFORE_PREDICTION")

        protocol = read_json(repo_root / paths["protocol"])
        artifacts = protocol["route"]["artifacts"]
        sys.path.insert(0, str(repo_root / "revision_management/02_model_rebuild/src"))
        from graph_models import build_graph_records, build_model, predict_raw as predict_graph_raw
        from lm_embeddings import extract_or_load_embeddings
        import joblib

        raw: dict[str, Any] = {}
        probability: dict[str, Any] = {}
        seal_paths: dict[str, Path] = {}

        rf_member = MEMBER_ORDER[0]
        rf_payload = joblib.load(repo_root / artifacts[rf_member]["base_model_path"])
        append_hash_chained_event(ledger_path, {"operation": "base_model_load", "member": rf_member, "device": "cpu_whitelisted"})
        maccs_rows = []
        for mol in mols:
            fp = MACCSkeys.GenMACCSKeys(mol)
            array = np.zeros((167,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, array)
            maccs_rows.append(array.astype(np.float32, copy=False))
        raw[rf_member] = np.asarray(rf_payload["model"].predict_proba(np.vstack(maccs_rows))[:, 1], dtype=np.float64)
        probability[rf_member] = gpu_platt_apply(repo_root / artifacts[rf_member]["calibrator_path"], raw[rf_member], rf_member, torch, np, ledger_path)
        event = append_hash_chained_event(ledger_path, {"operation": "member_prediction_complete", "member": rf_member, "rows": 5003})
        seal_paths[rf_member] = control / f"{PREDICTION_SEAL}_{rf_member}.json"
        seal_member_prediction(seal_paths[rf_member], rf_member, ordered_id_sha, raw[rf_member], probability[rf_member], artifacts[rf_member]["base_model_sha256"], artifacts[rf_member]["calibrator_sha256"], "cuda:0", event["sequence"])
        guard.check("RandomForest_MACCS_prediction_sealed")

        mpnn_member = MEMBER_ORDER[1]
        inference_cv_ids = ["INFER-CV-" + hashlib.sha256(value.encode()).hexdigest().upper()[:20] for value in candidate_noniso]
        graph_frame = pd.DataFrame({
            "model_unit_id": ordered_ids,
            "cv_group_id": inference_cv_ids,
            "final_label": np.full(len(structures), -1, dtype=np.int8),
            "canonical_isomeric_smiles": structures["standardized_smiles"],
        })
        graph_records = build_graph_records(graph_frame)
        mpnn_checkpoint = torch.load(repo_root / artifacts[mpnn_member]["base_model_path"], map_location="cuda:0", weights_only=False)
        append_hash_chained_event(ledger_path, {"operation": "base_model_load", "member": mpnn_member, "device": "cuda:0"})
        mpnn = build_model("MPNN", hidden_dim=256, dropout=0.30)
        mpnn.load_state_dict(mpnn_checkpoint["state_dict"], strict=True)
        raw[mpnn_member] = predict_graph_raw(mpnn, graph_records, np.arange(len(graph_records)), batch_size=128)
        probability[mpnn_member] = gpu_platt_apply(repo_root / artifacts[mpnn_member]["calibrator_path"], raw[mpnn_member], mpnn_member, torch, np, ledger_path)
        event = append_hash_chained_event(ledger_path, {"operation": "member_prediction_complete", "member": mpnn_member, "rows": 5003})
        seal_paths[mpnn_member] = control / f"{PREDICTION_SEAL}_{mpnn_member}.json"
        seal_member_prediction(seal_paths[mpnn_member], mpnn_member, ordered_id_sha, raw[mpnn_member], probability[mpnn_member], artifacts[mpnn_member]["base_model_sha256"], artifacts[mpnn_member]["calibrator_sha256"], "cuda:0", event["sequence"])
        guard.check("MPNN_prediction_sealed")

        lm_member = MEMBER_ORDER[2]
        append_hash_chained_event(ledger_path, {"operation": "fixed_feature_extractor_load_begin", "member": "MolFormer", "device": "cuda:0"})
        lm_frame = pd.DataFrame({"model_unit_id": ordered_ids, "canonical_isomeric_smiles": structures["standardized_smiles"]})
        embeddings, metadata = extract_or_load_embeddings(
            data=lm_frame,
            model_spec=protocol["feature_contracts"]["MolFormer"],
            input_sha256=sha256_file(repo_root / paths["structure_input"]),
            protocol_sha256=sha256_file(repo_root / paths["protocol"]),
            cache_root=staging / "FEATURE_CACHE",
            max_token_length=256,
            seed=20260805,
        )
        molformer_evidence = validate_molformer_extraction_metadata(
            metadata,
            int(protocol["feature_contracts"]["MolFormer"]["smoke_batch_size"]),
            int(protocol["feature_contracts"]["MolFormer"]["single_oom_reduced_batch_size"]),
        )
        molformer_metadata_path = control / "MOLFORMER_EXTRACTION_METADATA.json"
        atomic_write_json(molformer_metadata_path, metadata)
        molformer_evidence["metadata_file_sha256"] = sha256_file(molformer_metadata_path)
        append_hash_chained_event(ledger_path, {
            "operation": "fixed_feature_extractor_load_complete",
            "member": "MolFormer",
            "device": "cuda:0",
            "rows": 5003,
            "whole_run_retry": 0,
            "attempted_batch_sizes": molformer_evidence["attempted_batch_sizes"],
            "used_batch_size": molformer_evidence["used_batch_size"],
            "oom_reductions_used": molformer_evidence["oom_reductions_used"],
            "cpu_fallback_used": False,
            "metadata_file_sha256": molformer_evidence["metadata_file_sha256"],
            "metadata_canonical_sha256": molformer_evidence["metadata_canonical_sha256"],
        })
        lm_checkpoint = torch.load(repo_root / artifacts[lm_member]["base_model_path"], map_location="cuda:0", weights_only=False)
        append_hash_chained_event(ledger_path, {"operation": "base_model_load", "member": lm_member, "device": "cuda:0"})
        device = torch.device("cuda:0")
        head = torch.nn.Linear(embeddings.shape[1], 1, device=device)
        head.load_state_dict(lm_checkpoint["state_dict"], strict=True)
        with torch.no_grad():
            raw[lm_member] = head((torch.as_tensor(embeddings, device=device) - lm_checkpoint["mean"].to(device)) / lm_checkpoint["scale"].to(device)).squeeze(1).to(torch.float64).cpu().numpy()
        probability[lm_member] = gpu_platt_apply(repo_root / artifacts[lm_member]["calibrator_path"], raw[lm_member], lm_member, torch, np, ledger_path)
        event = append_hash_chained_event(ledger_path, {"operation": "member_prediction_complete", "member": lm_member, "rows": 5003})
        seal_paths[lm_member] = control / f"{PREDICTION_SEAL}_{lm_member}.json"
        seal_member_prediction(seal_paths[lm_member], lm_member, ordered_id_sha, raw[lm_member], probability[lm_member], artifacts[lm_member]["base_model_sha256"], artifacts[lm_member]["calibrator_sha256"], "cuda:0", event["sequence"])
        guard.check("MolFormer_prediction_sealed")

        verify_prediction_seals(seal_paths, probability, ordered_id_sha)
        ensemble = canonical_e005_probability(probability, np)
        if not np.array_equal(ensemble, canonical_equal_member_probability(probability)):
            raise ContractError("canonical E005 aggregation implementation mismatch")
        append_hash_chained_event(ledger_path, {"operation": "canonical_float64_matrix_mean", "rows": 5003, "ensemble_sha256": float64_array_sha256(ensemble)})

        result_rows = []
        for index, row in structures.iterrows():
            member_values = np.asarray([probability[name][index] for name in MEMBER_ORDER], dtype=np.float64)
            spread = float(np.max(member_values) - np.min(member_values))
            warnings = uncertainty_warning_flags(float(ensemble[index]), member_values)
            result_rows.append({
                **row.to_dict(),
                "training_overlap_class": overlap_class[index],
                "ecfp4_nearest_training_tanimoto": float(nearest_similarity[index]),
                "ecfp4_ad_reference_threshold": float(ad_threshold),
                "ecfp4_ad_warning": int(float(nearest_similarity[index]) < ad_threshold),
                "RandomForest_MACCS_probability": float(probability[MEMBER_ORDER[0]][index]),
                "MPNN_probability": float(probability[MEMBER_ORDER[1]][index]),
                "MolFormer_CUDA_Logistic_probability": float(probability[MEMBER_ORDER[2]][index]),
                "E005_ensemble_probability": float(ensemble[index]),
                "locked_threshold": float(LOCKED_THRESHOLD),
                "model_signal_positive": int(ensemble[index] >= LOCKED_THRESHOLD),
                "member_probability_sd_population": float(np.std(member_values, ddof=0)),
                "member_probability_range": spread,
                "member_probability_iqr_linear": float(np.quantile(member_values, 0.75, method="linear") - np.quantile(member_values, 0.25, method="linear")),
                "absolute_margin_to_locked_threshold": abs(float(ensemble[index] - LOCKED_THRESHOLD)),
                "near_threshold_warning": warnings["near_threshold_warning"],
                "moderate_disagreement_warning": warnings["moderate_disagreement_warning"],
                "high_disagreement_warning": warnings["high_disagreement_warning"],
                "uncertainty_warning_claim_limit": "diagnostic only; not a confidence interval or calibrated epistemic uncertainty",
            })
        ranked_structures = rank_structure_verification_priority(result_rows)
        structure_rank_fields = (
            "structure_verification_priority_rank",
            "structure_priority_tie_group_rank",
            "structure_priority_within_tie_order",
        )
        structure_rank_map = {
            row["screening_structure_id"]: {field: row[field] for field in structure_rank_fields}
            for row in ranked_structures
        }
        for row in result_rows:
            row.update(structure_rank_map[row["screening_structure_id"]])
        all_path = result_stage / "mfh_screening_all_candidate.csv"
        write_csv(all_path, list(result_rows[0]), result_rows)
        novel_rows = [row for row in result_rows if row["training_overlap_class"] == "STRUCTURE_UNSEEN"]
        write_csv(result_stage / "mfh_screening_novel_only_candidate.csv", list(result_rows[0]), novel_rows)
        overlap_fields = ["screening_structure_id", "training_overlap_class", "matched_training_model_unit_ids_json", "matched_training_cv_group_ids_json"]
        write_csv(result_stage / "mfh_training_overlaps_candidate.csv", overlap_fields, overlap_rows)

        result_frame = pd.DataFrame(result_rows)
        numeric_probability_columns = [
            "RandomForest_MACCS_probability",
            "MPNN_probability",
            "MolFormer_CUDA_Logistic_probability",
            "E005_ensemble_probability",
        ]
        result_frame[numeric_probability_columns] = result_frame[numeric_probability_columns].apply(pd.to_numeric, errors="raise")
        pair_result = pairs.merge(result_frame, on="screening_structure_id", how="left", validate="many_to_one")
        pair_result.to_csv(result_stage / "mfh_pair_results_candidate.csv", index=False, lineterminator="\n")
        source = pair_result.groupby(["mfh_id", "official_name"], as_index=False).agg(
            unique_mfh_structure_pairs=("mapping_pair_id", "nunique"),
            unique_standardized_structures=("screening_structure_id", "nunique"),
            model_signal_positive_structures=("model_signal_positive", "sum"),
            ad_warning_structures=("ecfp4_ad_warning", "sum"),
            high_disagreement_structures=("high_disagreement_warning", "sum"),
            maximum_e005_model_signal=("E005_ensemble_probability", "max"),
            mean_e005_model_signal=("E005_ensemble_probability", "mean"),
        )
        source["ad_warning_fraction"] = source["ad_warning_structures"] / source["unique_standardized_structures"]
        source["high_disagreement_fraction"] = source["high_disagreement_structures"] / source["unique_standardized_structures"]
        source_rows = rank_source_verification_priority(source.to_dict(orient="records"))
        for row in source_rows:
            row["claim_limit"] = "verification priority only; not occurrence, concentration, exposure, bioavailability, or human risk"
        write_csv(result_stage / "mfh_source_verification_priority_candidate.csv", list(source_rows[0]), source_rows)

        required_outputs = manifest["required_candidate_outputs"]
        missing = [name for name in required_outputs if not (result_stage / name).is_file()]
        if missing:
            raise ContractError(f"required candidate outputs missing: {missing}")
        append_hash_chained_event(ledger_path, {"operation": "result_package_complete", "required_outputs": len(required_outputs)})
        guard.check("pre_atomic_result_publication")
        atomic_write_json(control / "RESOURCE_LEDGER.json", {"limits": limits, "snapshots": guard.ledger})
        ledger_events = verify_operation_ledger(ledger_path)
        operation_counts = {name: sum(event.get("operation") == name for event in ledger_events) for name in {
            "base_model_load", "calibrator_load", "gpu_platt_application", "member_prediction_complete",
            "canonical_float64_matrix_mean", "ad_reference_seal", "fixed_feature_extractor_load_complete",
        }}
        expected_counts = {
            "base_model_load": 3, "calibrator_load": 3, "gpu_platt_application": 3,
            "member_prediction_complete": 3, "canonical_float64_matrix_mean": 1, "ad_reference_seal": 1,
            "fixed_feature_extractor_load_complete": 1,
        }
        if operation_counts != expected_counts:
            raise ContractError(f"operation ledger exact-count failure: {operation_counts}")
        atomic_write_json(control / "OPERATION_SUMMARY.json", {
            "counts": operation_counts,
            "model_fit": 0, "calibrator_fit": 0, "threshold_search": 0,
            "cpu_platt_fallback": 0,
            "whole_run_retry": 0,
            "retry": {
                "whole_run_retry": 0,
                "molformer_local_oom_batch_reduction": molformer_evidence["oom_reductions_used"],
                "local_oom_batch_reduction_is_not_whole_run_retry": True,
            },
            "molformer": molformer_evidence,
            "cpu_fallback": 0,
            "screening_lock_creation": 0,
            "ledger_sha256": sha256_file(ledger_path),
        })
        feature_cache = staging / "FEATURE_CACHE"
        if feature_cache.exists():
            shutil.rmtree(feature_cache)
        atomic_write_json(result_stage / "TERMINAL_STATUS.json", {
            "run_id": run_id,
            "state": "RESULT_PACKAGE_COMPLETE_PENDING_TASK00_REVIEW_NOT_SCREENING_LOCK",
            "authoritative_terminal_kind": "RESULT",
            "whole_run_retry": 0,
            "molformer_local_oom_batch_reduction": molformer_evidence["oom_reductions_used"],
            "screening_lock_created": False,
        })
        manifest_payload = build_output_manifest(result_stage)
        atomic_write_json(result_stage / OUTPUT_MANIFEST, manifest_payload)
        verify_output_manifest(result_stage, manifest_payload)
        publish_result_package_v3(result_stage, run_root)
        return
    except BaseException as error:
        # RESULT_PACKAGE directory presence is the authoritative success commit,
        # including the narrow case where parent-directory fsync raises after
        # renameat2 succeeded. Never publish a contradictory failure terminal.
        snapshots = [] if guard is None else guard.ledger
        publish_failure_package_v3_2(
            run_root=run_root,
            staging=staging,
            ledger_path=ledger_path,
            limits=limits,
            resource_snapshots=snapshots,
            claim_evidence=claim_evidence,
            failed_stage=failed_stage,
            environment_evidence=environment_evidence,
            error=error,
            traceback_text=traceback.format_exc(),
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT_DEFAULT)
    parser.add_argument("--authorization", required=True, type=Path)
    args = parser.parse_args()
    execute(args.repo_root.resolve(), args.authorization.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
