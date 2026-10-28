"""Deterministic grouped nested-CV manifest generation and validation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import StratifiedGroupKFold


REQUIRED_COLUMNS = {
    "scenario_id",
    "model_unit_id",
    "cv_group_id",
    "final_label",
    "canonical_nonisomeric_smiles",
    "downstream_training_authorized",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_id(prefix: str, value: str) -> str:
    return f"{prefix}-{hashlib.sha256(value.encode('utf-8')).hexdigest()[:20].upper()}"


def scaffold_identity(smiles: str, cv_group_id: str) -> tuple[str, str]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Locked model unit has invalid canonical SMILES: {smiles!r}")
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is not None and scaffold.GetNumAtoms() > 0:
        scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=False)
        return _stable_id("SCF", scaffold_smiles), scaffold_smiles
    return f"ACY-{cv_group_id}", ""


def load_locked_scenario(path: Path, expected_scenario: str, expected_sha256: str, expected_rows: int) -> pd.DataFrame:
    if sha256_file(path) != expected_sha256:
        raise RuntimeError(f"Locked input SHA256 mismatch: {path}")
    data = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise RuntimeError(f"Missing required locked columns: {sorted(missing)}")
    if len(data) != expected_rows:
        raise RuntimeError(f"Locked row count mismatch: {len(data)} != {expected_rows}")
    if set(data["scenario_id"].astype(str)) != {expected_scenario}:
        raise RuntimeError("Scenario identifier mismatch")
    if data["model_unit_id"].duplicated().any():
        raise RuntimeError("model_unit_id is not unique")
    if not data["downstream_training_authorized"].astype(bool).all():
        raise RuntimeError("Locked input contains an unauthorized model unit")
    label_counts = data.groupby("cv_group_id")["final_label"].nunique()
    if int((label_counts > 1).sum()) != 0:
        raise RuntimeError("A cv_group_id contains opposite labels")
    scaffolds = [
        scaffold_identity(smiles, group)
        for smiles, group in zip(
            data["canonical_nonisomeric_smiles"].astype(str),
            data["cv_group_id"].astype(str),
        )
    ]
    data = data.copy()
    data["scaffold_group_id"] = [item[0] for item in scaffolds]
    data["scaffold_smiles"] = [item[1] for item in scaffolds]
    data["final_label"] = data["final_label"].astype(int)
    return data


def _assert_two_classes(frame: pd.DataFrame, context: str) -> None:
    if set(frame["final_label"].astype(int)) != {0, 1}:
        raise RuntimeError(f"{context} does not contain both labels")


def _choose_grouped_splits(
    indices: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    base_seed: int,
    candidate_seed_count: int = 256,
    minimum_evaluation_size: int = 20,
) -> tuple[int, list[tuple[np.ndarray, np.ndarray]]]:
    """Select a deterministic grouped split using label/size balance only.

    This is registered split construction, not model selection: no features,
    predictions, or performance metrics are available to this function.
    """
    for offset in range(candidate_seed_count):
        seed = int(base_seed + offset)
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = [(train.copy(), test.copy()) for train, test in splitter.split(indices, labels, groups)]
        valid = True
        for train_pos, test_pos in splits:
            train_labels = labels[train_pos]
            test_labels = labels[test_pos]
            if len(test_pos) < minimum_evaluation_size:
                valid = False
                break
            if set(train_labels.tolist()) != {0, 1} or set(test_labels.tolist()) != {0, 1}:
                valid = False
                break
        if valid:
            return seed, splits
    raise RuntimeError(
        f"No eligible {n_splits}-fold grouped split found across {candidate_seed_count} "
        f"registered seeds starting at {base_seed}"
    )


def generate_manifest(
    data: pd.DataFrame,
    scenario_id: str,
    design_id: str,
    base_seed: int,
    outer_folds: int = 5,
    inner_folds: int = 5,
) -> pd.DataFrame:
    if design_id == "group_random_nested_cv":
        split_groups = data["cv_group_id"].astype(str).to_numpy()
    elif design_id == "scaffold_group_nested_cv":
        split_groups = data["scaffold_group_id"].astype(str).to_numpy()
    else:
        raise ValueError(design_id)

    indices = np.arange(len(data), dtype=np.int64)
    labels = data["final_label"].to_numpy(dtype=int)
    scenario_offset = int(hashlib.sha256(scenario_id.encode()).hexdigest()[:6], 16) % 100000
    design_offset = 0 if design_id == "group_random_nested_cv" else 500000
    outer_seed_base = int(base_seed + scenario_offset + design_offset)
    outer_seed = None
    outer_splits = None
    registered_inner_solutions: dict[int, tuple[int, list[tuple[np.ndarray, np.ndarray]]]] = {}
    for outer_offset in range(256):
        candidate_outer_seed = outer_seed_base + outer_offset
        try:
            _, candidate_outer_splits = _choose_grouped_splits(
                indices,
                labels,
                split_groups,
                n_splits=outer_folds,
                base_seed=candidate_outer_seed,
                candidate_seed_count=1,
            )
        except RuntimeError:
            continue
        candidate_inner_solutions = {}
        feasible = True
        for candidate_outer_fold, (candidate_train_pos, _) in enumerate(candidate_outer_splits):
            candidate_outer_train = indices[candidate_train_pos]
            candidate_inner_indices = np.arange(len(candidate_outer_train), dtype=np.int64)
            candidate_inner_labels = labels[candidate_outer_train]
            candidate_inner_groups = split_groups[candidate_outer_train]
            candidate_inner_base = candidate_outer_seed + 100000 + candidate_outer_fold * 1000
            try:
                candidate_inner_solutions[candidate_outer_fold] = _choose_grouped_splits(
                    candidate_inner_indices,
                    candidate_inner_labels,
                    candidate_inner_groups,
                    n_splits=inner_folds,
                    base_seed=candidate_inner_base,
                )
            except RuntimeError:
                feasible = False
                break
        if feasible:
            outer_seed = candidate_outer_seed
            outer_splits = candidate_outer_splits
            registered_inner_solutions = candidate_inner_solutions
            break
    if outer_seed is None or outer_splits is None:
        raise RuntimeError(
            f"No jointly eligible outer/inner grouped split found for {scenario_id} {design_id}"
        )
    rows: list[dict] = []

    def add_rows(
        selected: np.ndarray,
        level: str,
        outer_fold: int,
        inner_fold: int,
        subset: str,
        inner_seed: int | None,
    ) -> None:
        for idx in selected:
            record = data.iloc[int(idx)]
            rows.append(
                {
                    "manifest_version": "TASK02_SPLIT_V1",
                    "scenario_id": scenario_id,
                    "design_id": design_id,
                    "level": level,
                    "outer_fold": int(outer_fold),
                    "inner_fold": int(inner_fold),
                    "subset": subset,
                    "model_unit_id": str(record["model_unit_id"]),
                    "cv_group_id": str(record["cv_group_id"]),
                    "scaffold_group_id": str(record["scaffold_group_id"]),
                    "scaffold_smiles": str(record["scaffold_smiles"]),
                    "split_group_id": (
                        str(record["cv_group_id"])
                        if design_id == "group_random_nested_cv"
                        else str(record["scaffold_group_id"])
                    ),
                    "final_label": int(record["final_label"]),
                    "outer_seed": outer_seed,
                    "inner_seed": "" if inner_seed is None else int(inner_seed),
                }
            )

    for outer_fold, (outer_train_pos, outer_test_pos) in enumerate(outer_splits):
        outer_train = indices[outer_train_pos]
        outer_test = indices[outer_test_pos]
        add_rows(outer_train, "outer", outer_fold, -1, "train", None)
        add_rows(outer_test, "outer", outer_fold, -1, "test", None)

        inner_indices = np.arange(len(outer_train), dtype=np.int64)
        inner_seed, inner_splits = registered_inner_solutions[outer_fold]
        for inner_fold, (inner_train_pos, inner_valid_pos) in enumerate(inner_splits):
            add_rows(outer_train[inner_train_pos], "inner", outer_fold, inner_fold, "train", inner_seed)
            add_rows(outer_train[inner_valid_pos], "inner", outer_fold, inner_fold, "validation", inner_seed)

    manifest = pd.DataFrame(rows)
    level_order = pd.Categorical(manifest["level"], ["outer", "inner"], ordered=True)
    subset_order = pd.Categorical(manifest["subset"], ["train", "validation", "test"], ordered=True)
    manifest = (
        manifest.assign(_level_order=level_order, _subset_order=subset_order)
        .sort_values(
            ["_level_order", "outer_fold", "inner_fold", "_subset_order", "model_unit_id"],
            kind="mergesort",
        )
        .drop(columns=["_level_order", "_subset_order"])
        .reset_index(drop=True)
    )
    return manifest


def validate_manifest(manifest: pd.DataFrame, data: pd.DataFrame, design_id: str) -> dict:
    expected_ids = set(data["model_unit_id"].astype(str))
    atomic_key = "cv_group_id"
    split_key = "cv_group_id" if design_id == "group_random_nested_cv" else "scaffold_group_id"
    checks: dict[str, bool | int] = {
        "outer_cv_group_overlap_count": 0,
        "inner_cv_group_overlap_count": 0,
        "outer_split_group_overlap_count": 0,
        "inner_split_group_overlap_count": 0,
        "folds_missing_a_class": 0,
        "outer_test_missing_or_duplicate_model_units": 0,
        "inner_validation_missing_or_duplicate_model_units": 0,
    }

    outer = manifest[manifest["level"] == "outer"]
    outer_test_counts = outer.loc[outer["subset"] == "test", "model_unit_id"].value_counts()
    checks["outer_test_missing_or_duplicate_model_units"] = int(
        len(expected_ids - set(outer_test_counts.index)) + int((outer_test_counts != 1).sum())
    )

    for outer_fold in sorted(outer["outer_fold"].unique()):
        fold = outer[outer["outer_fold"] == outer_fold]
        train = fold[fold["subset"] == "train"]
        test = fold[fold["subset"] == "test"]
        checks["outer_cv_group_overlap_count"] += len(set(train[atomic_key]) & set(test[atomic_key]))
        checks["outer_split_group_overlap_count"] += len(set(train[split_key]) & set(test[split_key]))
        for frame in (train, test):
            if set(frame["final_label"].astype(int)) != {0, 1}:
                checks["folds_missing_a_class"] += 1

        inner = manifest[(manifest["level"] == "inner") & (manifest["outer_fold"] == outer_fold)]
        outer_train_ids = set(train["model_unit_id"])
        validation_counts = inner.loc[inner["subset"] == "validation", "model_unit_id"].value_counts()
        checks["inner_validation_missing_or_duplicate_model_units"] += int(
            len(outer_train_ids - set(validation_counts.index)) + int((validation_counts != 1).sum())
        )
        if set(inner["model_unit_id"]) - outer_train_ids:
            raise RuntimeError("Inner manifest contains an outer-test model unit")
        for inner_fold in sorted(inner["inner_fold"].unique()):
            inner_part = inner[inner["inner_fold"] == inner_fold]
            inner_train = inner_part[inner_part["subset"] == "train"]
            inner_valid = inner_part[inner_part["subset"] == "validation"]
            checks["inner_cv_group_overlap_count"] += len(
                set(inner_train[atomic_key]) & set(inner_valid[atomic_key])
            )
            checks["inner_split_group_overlap_count"] += len(
                set(inner_train[split_key]) & set(inner_valid[split_key])
            )
            for frame in (inner_train, inner_valid):
                if set(frame["final_label"].astype(int)) != {0, 1}:
                    checks["folds_missing_a_class"] += 1

    numeric_failures = sum(int(value) for key, value in checks.items() if key.endswith("count") or "missing" in key)
    checks["status"] = "PASS" if numeric_failures == 0 else "FAIL"
    checks["manifest_rows"] = int(len(manifest))
    checks["model_units"] = int(len(expected_ids))
    checks["outer_folds"] = int(outer["outer_fold"].nunique())
    checks["inner_folds_per_outer"] = int(
        manifest[manifest["level"] == "inner"].groupby("outer_fold")["inner_fold"].nunique().min()
    )
    return checks
