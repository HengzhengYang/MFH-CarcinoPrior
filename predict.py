#!/usr/bin/env python3
"""Local-only E005 inference. Run --help for SMILES and CSV examples."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
ROOT = Path(__file__).resolve().parent
THRESHOLD = 0.42534557978312176
MEMBERS = ("RandomForest_MACCS", "MPNN", "MolFormer_CUDA_Logistic")
FIELDS = ["row_id", "input_smiles", "canonical_smiles", "status", "error",
          *["p_" + name for name in MEMBERS], "p_E005", "classification_threshold",
          "signal_positive", "molformer_token_count", "molformer_truncated"]


def read_inputs(args):
    if args.smiles is not None:
        return [(str(i + 1), text) for i, text in enumerate(args.smiles)]
    with args.input.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = [args.smiles_column] + ([args.id_column] if args.id_column else [])
        if not reader.fieldnames or any(key not in reader.fieldnames for key in required):
            raise ValueError("CSV is missing a requested SMILES or ID column")
        rows = [(str(row[args.id_column]) if args.id_column else str(i + 1),
                 row[args.smiles_column] or "") for i, row in enumerate(reader)]
    if not rows:
        raise ValueError("Input CSV contains no data rows")
    return rows


def prepare_rows(inputs):
    from rdkit import Chem
    rows = []
    for row_id, text in inputs:
        row = dict.fromkeys(FIELDS, "")
        row.update(row_id=row_id, input_smiles=text, status="invalid_smiles")
        mol = Chem.MolFromSmiles(text.strip()) if text.strip() else None
        if mol is None or mol.GetNumAtoms() == 0:
            row["error"] = "Empty or invalid SMILES"
        else:
            row.update(canonical_smiles=Chem.MolToSmiles(mol, isomericSmiles=True),
                       status="pending")
        rows.append(row)
    return rows


def portable_collate(records, device):
    """Same graph assembly as the original collate; explicit CPU/CUDA device."""
    import numpy as np
    import torch
    from graph_models import GraphBatch, BOND_DIM
    xs, edges, attributes, batches = [], [], [], []
    offset = 0
    for i, record in enumerate(records):
        xs.append(record.x)
        if record.edge_index.shape[1]:
            edges.append(record.edge_index + offset)
            attributes.append(record.edge_attr)
        batches.append(np.full(len(record.x), i, dtype=np.int64))
        offset += len(record.x)
    return GraphBatch(
        torch.as_tensor(np.vstack(xs), dtype=torch.float32, device=device),
        torch.as_tensor(np.concatenate(edges, axis=1), dtype=torch.long, device=device)
        if edges else torch.zeros((2, 0), dtype=torch.long, device=device),
        torch.as_tensor(np.vstack(attributes), dtype=torch.float32, device=device)
        if attributes else torch.zeros((0, BOND_DIM), dtype=torch.float32, device=device),
        torch.as_tensor(np.concatenate(batches), dtype=torch.long, device=device),
        torch.as_tensor([r.y for r in records], dtype=torch.float32, device=device),
    )


def calibrate(raw, name, device):
    import numpy as np
    import torch
    checkpoint = torch.load(ROOT / "weights/e005" / (name + "_platt.pt"),
                            map_location=device, weights_only=False)
    module = torch.nn.Linear(1, 1, device=device, dtype=torch.float32)
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    with torch.no_grad():
        tensor = torch.as_tensor(np.asarray(raw, dtype=np.float32).reshape(-1, 1),
                                 device=device)
        return torch.sigmoid(module(tensor).squeeze(1)).to(torch.float64).cpu().numpy()


def predict_probabilities(smiles, device_name="auto", batch_size=32):
    import joblib
    import numpy as np
    import pandas as pd
    import torch
    from rdkit import Chem, DataStructs
    from rdkit.Chem import MACCSkeys
    from transformers import AutoModel, AutoTokenizer

    source = ROOT / "code/original_project/revision_management/02_model_rebuild/src"
    sys.path.insert(0, str(source))
    from graph_models import MPNN, build_graph_records
    from runtime_guards import set_reproducible_seed

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    device = torch.device("cuda:0" if device_name == "cuda" else "cpu")
    # Small molecular batches do not benefit from excessive CPU thread counts.
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    set_reproducible_seed(20260805)
    print(f"E005: {len(smiles)} molecules, device={device}, batch_size={batch_size}",
          file=sys.stderr)
    weights = ROOT / "weights/e005"
    rf = joblib.load(weights / "RandomForest_MACCS.joblib")["model"]
    if list(rf.classes_) != [0, 1]:
        raise ValueError("Unexpected random-forest class order")
    features = []
    for text in smiles:
        array = np.zeros(167, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(text)), array)
        features.append(array.astype(np.float32))
    raw_rf = np.asarray(rf.predict_proba(np.vstack(features))[:, 1], dtype=np.float64)

    frame = pd.DataFrame({"model_unit_id": [str(i) for i in range(len(smiles))],
                          "cv_group_id": "inference", "final_label": -1,
                          "canonical_isomeric_smiles": smiles})
    records = build_graph_records(frame)
    checkpoint = torch.load(weights / "MPNN.pt", map_location=device, weights_only=False)
    model = MPNN(hidden_dim=256, dropout=0.30).to(device).eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    with torch.no_grad():
        raw_mpnn = np.concatenate([
            model(portable_collate(records[start:start + batch_size], device)).cpu().numpy()
            for start in range(0, len(records), batch_size)
        ]).astype(np.float64)
    del model

    set_reproducible_seed(20260805)
    backbone = ROOT / "weights/molformer_backbone"
    tokenizer = AutoTokenizer.from_pretrained(str(backbone), local_files_only=True,
                                              trust_remote_code=True)
    lengths = [len(ids) for ids in tokenizer(smiles, add_special_tokens=True,
                                            truncation=False, padding=False)["input_ids"]]
    encoder = AutoModel.from_pretrained(str(backbone), local_files_only=True,
                                       trust_remote_code=True, deterministic_eval=True).eval().to(device)
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(smiles), batch_size):
            encoded = tokenizer(smiles[start:start + batch_size], padding=True,
                                truncation=True, max_length=256, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = encoder(**encoded).last_hidden_state
            mask = encoded["attention_mask"].to(hidden.dtype).unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)
            embeddings.append(pooled.cpu().numpy().astype(np.float32))
    del encoder
    embeddings = np.vstack(embeddings)
    checkpoint = torch.load(weights / "MolFormer_CUDA_Logistic.pt", map_location=device,
                            weights_only=False)
    head = torch.nn.Linear(embeddings.shape[1], 1, device=device)
    head.load_state_dict(checkpoint["state_dict"], strict=True)
    with torch.no_grad():
        x = torch.as_tensor(embeddings, device=device)
        standardized = (x - checkpoint["mean"].to(device)) / checkpoint["scale"].to(device)
        raw_lm = head(standardized).squeeze(1).to(torch.float64).cpu().numpy()
    raw = (raw_rf, raw_mpnn, raw_lm)
    if not all(np.isfinite(value).all() for value in raw):
        raise RuntimeError("A member produced non-finite raw outputs")
    probabilities = {name: calibrate(value, name, device) for name, value in zip(MEMBERS, raw)}
    probabilities["E005"] = np.mean(np.vstack([probabilities[n] for n in MEMBERS]),
                                    axis=0, dtype=np.float64)
    if not all(np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()
               for p in probabilities.values()):
        raise RuntimeError("Invalid calibrated probabilities")
    return probabilities, lengths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smiles", nargs="+", help="One or more quoted SMILES")
    group.add_argument("--input", type=Path, help="UTF-8 CSV input")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--id-column", help="Optional CSV identifier column; duplicates retained")
    parser.add_argument("--output", required=True, type=Path, help="New output CSV path")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    try:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be positive")
        if args.output.exists():
            raise ValueError("Output already exists; choose a new path")
        rows = prepare_rows(read_inputs(args))
        valid = [row for row in rows if row["status"] == "pending"]
        if valid:
            probabilities, lengths = predict_probabilities(
                [r["canonical_smiles"] for r in valid], args.device, args.batch_size)
            for i, row in enumerate(valid):
                row.update(status="ok", classification_threshold=THRESHOLD,
                           signal_positive=int(probabilities["E005"][i] >= THRESHOLD),
                           molformer_token_count=lengths[i], molformer_truncated=lengths[i] > 256)
                row.update({"p_" + name: float(values[i]) for name, values in probabilities.items()})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        invalid = len(rows) - len(valid)
        print(f"Wrote {args.output}: {len(valid)} predicted, {invalid} invalid", file=sys.stderr)
        return 2 if invalid else 0
    except Exception as error:
        print(f"Prediction failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
