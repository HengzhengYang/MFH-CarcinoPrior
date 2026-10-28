"""Pure control and numerical primitives for Task04 D111 V3.1.

No function in this module opens a scientific input, loads a model, extracts a
feature, or performs a prediction.  Production and synthetic tests share these
exact boundary semantics.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


MEMBER_ORDER = (
    "RandomForest_MACCS",
    "MPNN",
    "MolFormer_CUDA_Logistic",
)
LOCKED_THRESHOLD = np.float64(0.42534557978312176)
NEAR_THRESHOLD_MAX_MARGIN = np.float64(0.05)
MODERATE_DISAGREEMENT_MIN_RANGE = np.float64(0.15)
HIGH_DISAGREEMENT_MIN_RANGE = np.float64(0.30)


class ContractError(RuntimeError):
    pass


class ResourceLimitError(ContractError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def canonical_float64(values: Any) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ContractError("finite one-dimensional float64 array required")
    return array


def float64_array_sha256(values: Any) -> str:
    return sha256_bytes(canonical_float64(values).tobytes(order="C"))


def canonical_equal_member_probability(member_probabilities: dict[str, Any]) -> np.ndarray:
    """Frozen E002/E005 registered-order float64 matrix mean."""
    if tuple(member_probabilities) != MEMBER_ORDER or set(member_probabilities) != set(MEMBER_ORDER):
        raise ContractError("member probability order differs from registered E005 order")
    arrays = [canonical_float64(member_probabilities[name]) for name in MEMBER_ORDER]
    if len({len(array) for array in arrays}) != 1:
        raise ContractError("member prediction lengths differ")
    result = np.mean(np.vstack(arrays), axis=0, dtype=np.float64)
    if np.any((result < 0.0) | (result > 1.0)):
        raise ContractError("ensemble probability outside [0,1]")
    return result


def fsync_directory(path: Path) -> None:
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
    fsync_directory(path.parent)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_bytes(path, json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8") + b"\n")


def publish_noreplace(staging: Path, destination: Path) -> None:
    """Atomically publish a directory with Linux RENAME_NOREPLACE."""
    if not staging.is_dir() or destination.exists():
        raise FileExistsError(str(destination))
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.renameat2(-100, os.fsencode(staging), -100, os.fsencode(destination), 1)
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))
    fsync_directory(destination.parent)


def append_hash_chained_event(path: Path, event: dict[str, Any]) -> dict[str, Any]:
    previous_sha = "0" * 64
    sequence = 1
    if path.exists():
        events = verify_operation_ledger(path)
        previous_sha = events[-1]["event_sha256"]
        sequence = int(events[-1]["sequence"]) + 1
    body = {"sequence": sequence, "previous_event_sha256": previous_sha, **event}
    body["event_sha256"] = sha256_bytes(canonical_json_bytes(body))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(canonical_json_bytes(body))
        handle.flush()
        os.fsync(handle.fileno())
    return body


def verify_operation_ledger(path: Path) -> list[dict[str, Any]]:
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    previous_sha = "0" * 64
    for expected_sequence, event in enumerate(events, 1):
        if event.get("sequence") != expected_sequence or event.get("previous_event_sha256") != previous_sha:
            raise ContractError("operation ledger sequence/hash chain mismatch")
        observed_sha = event.get("event_sha256")
        body = dict(event)
        body.pop("event_sha256", None)
        if observed_sha != sha256_bytes(canonical_json_bytes(body)):
            raise ContractError("operation ledger event hash mismatch")
        previous_sha = observed_sha
    return events


def seal_member_prediction(
    seal_path: Path,
    member: str,
    ordered_id_sha256: str,
    raw_scores: Any,
    probabilities: Any,
    base_model_sha256: str,
    calibrator_sha256: str,
    device: str,
    ledger_sequence: int,
) -> dict[str, Any]:
    if member not in MEMBER_ORDER:
        raise ContractError("unregistered E005 member")
    raw = canonical_float64(raw_scores)
    probability = canonical_float64(probabilities)
    if len(raw) != len(probability) or np.any((probability < 0.0) | (probability > 1.0)):
        raise ContractError("invalid member prediction arrays")
    payload = {
        "seal_type": "TASK04_MEMBER_PREDICTION_SEAL_V3",
        "member": member,
        "registered_ordinal": MEMBER_ORDER.index(member),
        "row_count": len(probability),
        "ordered_screening_structure_id_sha256": ordered_id_sha256,
        "raw_score_float64_c_sha256": float64_array_sha256(raw),
        "calibrated_probability_float64_c_sha256": float64_array_sha256(probability),
        "base_model_sha256": base_model_sha256,
        "calibrator_sha256": calibrator_sha256,
        "application_device": device,
        "ledger_sequence": ledger_sequence,
        "complete": True,
    }
    atomic_write_json(seal_path, payload)
    return payload


def verify_prediction_seals(
    seal_paths: dict[str, Path],
    probabilities: dict[str, Any],
    ordered_id_sha256: str,
) -> None:
    if tuple(seal_paths) != MEMBER_ORDER or tuple(probabilities) != MEMBER_ORDER:
        raise ContractError("three registered-order prediction seals required")
    for member in MEMBER_ORDER:
        payload = json.loads(seal_paths[member].read_text(encoding="utf-8"))
        if payload.get("member") != member or payload.get("complete") is not True:
            raise ContractError(f"incomplete prediction seal: {member}")
        if payload.get("ordered_screening_structure_id_sha256") != ordered_id_sha256:
            raise ContractError(f"prediction ID seal mismatch: {member}")
        if payload.get("calibrated_probability_float64_c_sha256") != float64_array_sha256(probabilities[member]):
            raise ContractError(f"prediction probability seal mismatch: {member}")


def build_output_manifest(package_root: Path, exclude: Iterable[str] = ("OUTPUT_MANIFEST.json",)) -> dict[str, Any]:
    excluded = set(exclude)
    files = []
    for path in sorted(value for value in package_root.rglob("*") if value.is_file() and value.name not in excluded):
        files.append({
            "path": path.relative_to(package_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return {"manifest_version": "TASK04_ATOMIC_PACKAGE_V3_1", "file_count": len(files), "files": files}


def verify_output_manifest(package_root: Path, manifest: dict[str, Any]) -> None:
    if manifest.get("file_count") != len(manifest.get("files", [])):
        raise ContractError("output manifest count mismatch")
    for item in manifest["files"]:
        path = package_root / item["path"]
        if not path.is_file() or path.stat().st_size != item["size_bytes"] or sha256_file(path) != item["sha256"]:
            raise ContractError(f"output manifest mismatch: {item['path']}")


class ResourceGuard:
    def __init__(self, start_monotonic: float, staging_root: Path, limits: dict[str, int], process: Any, torch: Any):
        self.start = float(start_monotonic)
        self.staging_root = staging_root
        self.limits = limits
        self.process = process
        self.torch = torch
        self.ledger: list[dict[str, Any]] = []

    def check(self, stage: str) -> dict[str, Any]:
        wall = time.monotonic() - self.start
        rss = int(self.process.memory_info().rss)
        vram = int(self.torch.cuda.max_memory_allocated(0)) if self.torch.cuda.is_available() else 0
        storage = sum(path.stat().st_size for path in self.staging_root.rglob("*") if path.is_file()) if self.staging_root.exists() else 0
        snapshot = {"stage": stage, "wall_seconds": wall, "rss_bytes": rss, "vram_bytes": vram, "staging_bytes": storage}
        self.ledger.append(snapshot)
        for field, key in (
            (wall, "wall_seconds"), (rss, "rss_bytes"),
            (vram, "vram_bytes"), (storage, "storage_bytes"),
        ):
            if field > self.limits[key]:
                raise ResourceLimitError(f"{key} ceiling exceeded at {stage}: {field}>{self.limits[key]}")
        return snapshot


def safe_remove_staging(staging: Path, run_root: Path) -> None:
    staging_resolved = staging.resolve()
    run_resolved = run_root.resolve()
    if run_resolved not in staging_resolved.parents or staging.name != "STAGING":
        raise ContractError("refuse to remove non-registered staging directory")
    if staging.exists():
        shutil.rmtree(staging)


def uncertainty_warning_flags(ensemble_probability: float, member_probabilities: Any) -> dict[str, int]:
    """Apply the three V1 preregistered diagnostic warning boundaries exactly."""
    members = canonical_float64(member_probabilities)
    if len(members) != len(MEMBER_ORDER):
        raise ContractError("exactly three member probabilities required")
    probability = np.float64(ensemble_probability)
    if not np.isfinite(probability) or probability < 0.0 or probability > 1.0:
        raise ContractError("finite ensemble probability in [0,1] required")
    margin = np.abs(probability - LOCKED_THRESHOLD)
    member_range = np.max(members) - np.min(members)
    return {
        "near_threshold_warning": int(margin <= NEAR_THRESHOLD_MAX_MARGIN),
        "moderate_disagreement_warning": int(member_range >= MODERATE_DISAGREEMENT_MIN_RANGE),
        "high_disagreement_warning": int(member_range >= HIGH_DISAGREEMENT_MIN_RANGE),
    }


def validate_molformer_extraction_metadata(metadata: dict[str, Any], initial_batch: int, reduced_batch: int) -> dict[str, Any]:
    """Normalize and seal the permitted local OOM reduction evidence."""
    if not isinstance(metadata, dict):
        raise ContractError("MolFormer extractor metadata object required")
    reductions = int(metadata.get("oom_reductions_used", -1))
    attempted = [int(value) for value in metadata.get("attempted_batch_sizes", [])]
    cpu_fallback = bool(metadata.get("cpu_fallback_used", True))
    used_batch = int(metadata.get("used_batch_size", -1))
    expected_attempted = [int(initial_batch)] if reductions == 0 else [int(initial_batch), int(reduced_batch)]
    if reductions not in (0, 1):
        raise ContractError("MolFormer permits at most one local OOM batch reduction")
    if attempted != expected_attempted or used_batch != expected_attempted[-1]:
        raise ContractError("MolFormer attempted/used batch-size evidence mismatch")
    if cpu_fallback or str(metadata.get("device")) != "cuda:0":
        raise ContractError("MolFormer CPU fallback or non-cuda:0 execution detected")
    if metadata.get("cache_hit") is not False:
        raise ContractError("fresh single-use run must not consume a prior embedding cache")
    return {
        "whole_run_retry": 0,
        "attempted_batch_sizes": attempted,
        "used_batch_size": used_batch,
        "oom_reductions_used": reductions,
        "cpu_fallback_used": False,
        "device": "cuda:0",
        "metadata_canonical_sha256": sha256_bytes(canonical_json_bytes(metadata)),
    }


def _rank_with_ties(rows: list[dict[str, Any]], scientific_key: Any, tie_break_key: Any, rank_field: str, tie_field: str, order_field: str) -> list[dict[str, Any]]:
    ordered = sorted((dict(row) for row in rows), key=lambda row: scientific_key(row) + tie_break_key(row))
    previous = None
    tie_rank = 0
    within_tie = 0
    for ordinal, row in enumerate(ordered, 1):
        current = scientific_key(row)
        if current != previous:
            tie_rank += 1
            within_tie = 1
            previous = current
        else:
            within_tie += 1
        row[rank_field] = ordinal
        row[tie_field] = tie_rank
        row[order_field] = within_tie
    return ordered


def rank_structure_verification_priority(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Frozen numeric structure hierarchy; identifiers are deterministic final tie-breaks."""
    def scientific(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            -int(row["model_signal_positive"]),
            -float(row["E005_ensemble_probability"]),
            int(row["ecfp4_ad_warning"]),
            int(row["high_disagreement_warning"]),
            int(row["moderate_disagreement_warning"]),
            int(row["near_threshold_warning"]),
            float(row["member_probability_range"]),
        )

    return _rank_with_ties(
        rows,
        scientific,
        lambda row: (str(row["screening_structure_id"]),),
        "structure_verification_priority_rank",
        "structure_priority_tie_group_rank",
        "structure_priority_within_tie_order",
    )


def rank_source_verification_priority(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Frozen numeric source hierarchy; source identifiers are final tie-breaks."""
    def scientific(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            -int(row["model_signal_positive_structures"]),
            -float(row["maximum_e005_model_signal"]),
            -float(row["mean_e005_model_signal"]),
            float(row["ad_warning_fraction"]),
            float(row["high_disagreement_fraction"]),
            -int(row["unique_standardized_structures"]),
        )

    return _rank_with_ties(
        rows,
        scientific,
        lambda row: (str(row["mfh_id"]), str(row["official_name"])),
        "source_verification_priority_rank",
        "source_priority_tie_group_rank",
        "source_priority_within_tie_order",
    )


def authoritative_terminal_kind(run_root: Path) -> str | None:
    result_exists = (run_root / "RESULT_PACKAGE").is_dir()
    failure_exists = (run_root / "FAILURE_PACKAGE").is_dir()
    if result_exists and failure_exists:
        raise ContractError("contradictory result and failure terminal packages")
    if result_exists:
        return "RESULT"
    if failure_exists:
        return "FAILURE"
    return None


def partial_staging_manifest(staging: Path) -> dict[str, Any]:
    files = []
    if staging.exists():
        for path in sorted(value for value in staging.rglob("*") if value.is_file()):
            files.append({
                "path": path.relative_to(staging).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    return {"file_count": len(files), "files": files}


def publish_failure_package_v3(
    run_root: Path,
    staging: Path,
    ledger_path: Path,
    limits: dict[str, int],
    resource_snapshots: list[dict[str, Any]],
    claim_evidence: dict[str, Any],
    error: BaseException,
    traceback_text: str,
) -> None:
    """Publish the sole failure terminal for any exception after run-root claim."""
    if authoritative_terminal_kind(run_root) is not None:
        raise ContractError("terminal package already committed; refusing second terminal")
    failure_stage = run_root / ".FAILURE_PACKAGE_STAGING"
    failure_stage.mkdir(exist_ok=False)
    atomic_write_json(failure_stage / "ATTEMPT_CLAIM_EVIDENCE.json", claim_evidence)
    atomic_write_json(failure_stage / "FAILURE.json", {
        "run_id": claim_evidence["run_id"],
        "state": "HARD_FAILED_RUN_ID_CONSUMED_NO_RETRY",
        "error_type": type(error).__name__,
        "error": str(error),
        "screening_lock_created": False,
        "authoritative_terminal_kind": "FAILURE",
    })
    atomic_write_bytes(failure_stage / "TRACEBACK.txt", traceback_text.encode("utf-8"))
    atomic_write_json(failure_stage / "PARTIAL_STAGING_MANIFEST.json", partial_staging_manifest(staging))
    if ledger_path.is_file():
        shutil.copy2(ledger_path, failure_stage / "OPERATION_LEDGER.jsonl")
    else:
        append_hash_chained_event(failure_stage / "OPERATION_LEDGER.jsonl", {
            "operation": "postclaim_failure_before_operation_ledger_initialization",
            "run_id": claim_evidence["run_id"],
        })
    atomic_write_json(failure_stage / "RESOURCE_LEDGER.json", {
        "limits": limits,
        "snapshots": resource_snapshots,
        "resource_guard_initialized": bool(resource_snapshots),
    })
    # D111/B17: the complete partial-staging manifest and operation ledger are
    # already sealed in failure_stage. Remove only the exact validated
    # run_root/STAGING tree before the authoritative failure terminal is
    # published, so no mutable partial scientific file remains outside it.
    safe_remove_staging(staging, run_root)
    if staging.exists():
        raise ContractError("validated STAGING tree remains after failure cleanup")
    atomic_write_json(failure_stage / "TERMINAL_STATUS.json", {
        "run_id": claim_evidence["run_id"],
        "state": "HARD_FAILED_RUN_ID_CONSUMED_NO_RETRY",
        "authoritative_terminal_kind": "FAILURE",
        "whole_run_retry": 0,
        "screening_lock_created": False,
    })
    manifest = build_output_manifest(failure_stage)
    atomic_write_json(failure_stage / "OUTPUT_MANIFEST.json", manifest)
    verify_output_manifest(failure_stage, manifest)
    publish_noreplace(failure_stage, run_root / "FAILURE_PACKAGE")


def publish_result_package_v3(result_stage: Path, run_root: Path) -> None:
    """Commit the sole success terminal; callers must perform no later writes."""
    if authoritative_terminal_kind(run_root) is not None:
        raise ContractError("terminal package already committed; refusing result publication")
    status_path = result_stage / "TERMINAL_STATUS.json"
    if not status_path.is_file():
        raise ContractError("result terminal status must be inside staged result package")
    manifest = json.loads((result_stage / "OUTPUT_MANIFEST.json").read_text(encoding="utf-8"))
    verify_output_manifest(result_stage, manifest)
    publish_noreplace(result_stage, run_root / "RESULT_PACKAGE")


def settle_postclaim_exception_v3(
    run_root: Path,
    staging: Path,
    ledger_path: Path,
    limits: dict[str, int],
    resource_snapshots: list[dict[str, Any]],
    claim_evidence: dict[str, Any],
    error: BaseException,
    traceback_text: str,
) -> str:
    """Resolve a claimed-run exception without ever creating dual terminals."""
    terminal = authoritative_terminal_kind(run_root)
    if terminal == "RESULT":
        return "RESULT_ALREADY_COMMITTED_NO_FAILURE_PUBLISHED"
    if terminal == "FAILURE":
        raise ContractError("failure terminal already committed")
    publish_failure_package_v3(
        run_root, staging, ledger_path, limits, resource_snapshots,
        claim_evidence, error, traceback_text,
    )
    return "FAILURE_PUBLISHED"
