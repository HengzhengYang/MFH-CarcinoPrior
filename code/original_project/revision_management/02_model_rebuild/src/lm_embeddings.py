"""Pinned, local-only CUDA embedding extraction with content-addressed cache checks."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from runtime_guards import (
    assert_cuda_module,
    assert_cuda_tensor,
    require_cuda_runtime,
    run_with_single_oom_retry,
    set_reproducible_seed,
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _ordered_text_hash(values: list[str]) -> str:
    return _sha256_bytes("\n".join(values).encode("utf-8"))


def _array_hash(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return _sha256_bytes(contiguous.view(np.uint8).tobytes())


def extract_or_load_embeddings(
    data: pd.DataFrame,
    model_spec: dict,
    input_sha256: str,
    protocol_sha256: str,
    cache_root: Path,
    max_token_length: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    model_unit_ids = data["model_unit_id"].astype(str).tolist()
    smiles = data["canonical_isomeric_smiles"].astype(str).tolist()
    expected = {
        "cache_version": "TASK02_LM_EMBEDDING_V1",
        "name": model_spec["name"],
        "model_id": model_spec["model_id"],
        "revision": model_spec["revision"],
        "input_sha256": input_sha256,
        "protocol_sha256": protocol_sha256,
        "ordered_model_unit_id_sha256": _ordered_text_hash(model_unit_ids),
        "ordered_smiles_sha256": _ordered_text_hash(smiles),
        "pooling": "attention_mask_weighted_mean_last_hidden_state",
        "max_token_length": int(max_token_length),
        "row_count": len(data),
        "local_files_only": True,
    }
    cache_dir = cache_root / protocol_sha256[:16] / model_spec["name"] / model_spec["revision"]
    array_path = cache_dir / "embeddings.npz"
    metadata_path = cache_dir / "metadata.json"
    if array_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise RuntimeError(f"Embedding cache metadata mismatch for {model_spec['name']}: {key}")
        payload = np.load(array_path, allow_pickle=False)
        cached_ids = payload["model_unit_ids"].astype(str).tolist()
        embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
        if cached_ids != model_unit_ids:
            raise RuntimeError("Embedding cache ordered model_unit_id mismatch")
        if metadata.get("embedding_array_sha256") != _array_hash(embeddings):
            raise RuntimeError("Embedding cache array SHA256 mismatch")
        metadata = dict(metadata)
        metadata["cache_hit"] = True
        return embeddings, metadata

    device = require_cuda_runtime()
    set_reproducible_seed(seed)
    trust_remote_code = model_spec["name"] == "MolFormer"
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec["model_id"],
        revision=model_spec["revision"],
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )
    token_lengths = [
        len(ids)
        for ids in tokenizer(
            smiles,
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )["input_ids"]
    ]

    def operation(batch_size: int):
        model = AutoModel.from_pretrained(
            model_spec["model_id"],
            revision=model_spec["revision"],
            local_files_only=True,
            trust_remote_code=trust_remote_code,
            **({"deterministic_eval": True} if trust_remote_code else {}),
        ).eval().to(device)
        assert_cuda_module(model, model_spec["name"])
        outputs = []
        torch.cuda.reset_peak_memory_stats(device)
        try:
            with torch.no_grad():
                for start in range(0, len(smiles), batch_size):
                    encoded = tokenizer(
                        smiles[start : start + batch_size],
                        padding=True,
                        truncation=True,
                        max_length=max_token_length,
                        return_tensors="pt",
                    )
                    encoded = {key: value.to(device) for key, value in encoded.items()}
                    for key, value in encoded.items():
                        assert_cuda_tensor(value, f"{model_spec['name']}.{key}")
                    hidden = model(**encoded).last_hidden_state
                    assert_cuda_tensor(hidden, f"{model_spec['name']}.last_hidden_state")
                    mask = encoded.get("attention_mask")
                    if mask is None:
                        mask = torch.ones(hidden.shape[:2], dtype=hidden.dtype, device=device)
                    mask = mask.to(hidden.dtype).unsqueeze(-1)
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                    assert_cuda_tensor(pooled, f"{model_spec['name']}.pooled_embedding")
                    if not bool(torch.isfinite(pooled).all().item()):
                        raise RuntimeError(f"{model_spec['name']} produced non-finite embeddings")
                    outputs.append(pooled.detach().cpu().numpy().astype(np.float32))
            torch.cuda.synchronize(device)
            embeddings = np.vstack(outputs).astype(np.float32, copy=False)
            peak_mib = round(torch.cuda.max_memory_allocated(device) / 1024**2, 1)
            return embeddings, peak_mib, batch_size
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()

    result, retry_evidence = run_with_single_oom_retry(
        operation,
        int(model_spec["smoke_batch_size"]),
        int(model_spec["single_oom_reduced_batch_size"]),
    )
    embeddings, peak_mib, used_batch_size = result
    if embeddings.shape[0] != len(data):
        raise RuntimeError("Embedding row count mismatch")
    metadata = {
        **expected,
        "embedding_dimension": int(embeddings.shape[1]),
        "embedding_array_sha256": _array_hash(embeddings),
        "maximum_observed_token_length": int(max(token_lengths)),
        "truncated_sequence_count": int(sum(length > max_token_length for length in token_lengths)),
        "used_batch_size": int(used_batch_size),
        "attempted_batch_sizes": list(retry_evidence.attempted_batch_sizes),
        "oom_reductions_used": retry_evidence.reductions_used,
        "cpu_fallback_used": retry_evidence.cpu_fallback_used,
        "device": str(device),
        "torch": torch.__version__,
        "peak_memory_mib": peak_mib,
        "cache_hit": False,
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        array_path,
        embeddings=embeddings,
        model_unit_ids=np.asarray(model_unit_ids, dtype=str),
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return embeddings, metadata
