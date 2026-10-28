"""Native-PyTorch CUDA molecular graph models for the Task 02 rebuild."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from runtime_guards import (
    assert_cuda_module,
    assert_cuda_tensor,
    require_cuda_runtime,
    set_reproducible_seed,
)


ATOM_NUMBERS = [1, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 19, 20, 26, 29, 30, 34, 35, 53]
DEGREES = [0, 1, 2, 3, 4, 5]
FORMAL_CHARGES = [-2, -1, 0, 1, 2]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
TOTAL_HS = [0, 1, 2, 3, 4]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
STEREO_TYPES = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
]


def _one_hot(value, choices: list) -> list[float]:
    return [float(value == choice) for choice in choices] + [float(value not in choices)]


def atom_features(atom: Chem.Atom) -> np.ndarray:
    values = []
    values += _one_hot(atom.GetAtomicNum(), ATOM_NUMBERS)
    values += _one_hot(atom.GetDegree(), DEGREES)
    values += _one_hot(atom.GetFormalCharge(), FORMAL_CHARGES)
    values += _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
    values += _one_hot(atom.GetTotalNumHs(), TOTAL_HS)
    values += [
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        float(atom.GetMass() * 0.01),
    ]
    return np.asarray(values, dtype=np.float32)


def bond_features(bond: Chem.Bond | None) -> np.ndarray:
    if bond is None:
        return np.zeros((len(BOND_TYPES) + 1 + len(STEREO_TYPES) + 1 + 2,), dtype=np.float32)
    values = []
    values += _one_hot(bond.GetBondType(), BOND_TYPES)
    values += _one_hot(bond.GetStereo(), STEREO_TYPES)
    values += [float(bond.GetIsConjugated()), float(bond.IsInRing())]
    return np.asarray(values, dtype=np.float32)


NODE_DIM = len(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))
BOND_DIM = len(bond_features(None))


@dataclass
class GraphRecord:
    model_unit_id: str
    cv_group_id: str
    y: int
    x: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray


@dataclass
class GraphBatch:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    batch: torch.Tensor
    y: torch.Tensor


def build_graph_records(data: pd.DataFrame) -> list[GraphRecord]:
    records = []
    for row in data.itertuples(index=False):
        mol = Chem.MolFromSmiles(str(row.canonical_isomeric_smiles))
        if mol is None:
            raise ValueError(f"Invalid locked SMILES for {row.model_unit_id}")
        x = np.vstack([atom_features(atom) for atom in mol.GetAtoms()]).astype(np.float32)
        edges = []
        attributes = []
        for bond in mol.GetBonds():
            begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            feature = bond_features(bond)
            edges.extend([(begin, end), (end, begin)])
            attributes.extend([feature, feature])
        if edges:
            edge_index = np.asarray(edges, dtype=np.int64).T
            edge_attr = np.vstack(attributes).astype(np.float32)
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_attr = np.zeros((0, BOND_DIM), dtype=np.float32)
        records.append(
            GraphRecord(
                model_unit_id=str(row.model_unit_id),
                cv_group_id=str(row.cv_group_id),
                y=int(row.final_label),
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
            )
        )
    return records


def collate(records: list[GraphRecord], indices: np.ndarray, device: torch.device) -> GraphBatch:
    xs, edge_indices, edge_attrs, batch_ids, labels = [], [], [], [], []
    offset = 0
    for graph_index, record_index in enumerate(np.asarray(indices, dtype=np.int64)):
        record = records[int(record_index)]
        node_count = len(record.x)
        xs.append(record.x)
        if record.edge_index.shape[1]:
            edge_indices.append(record.edge_index + offset)
            edge_attrs.append(record.edge_attr)
        batch_ids.append(np.full((node_count,), graph_index, dtype=np.int64))
        labels.append(record.y)
        offset += node_count
    x = torch.as_tensor(np.vstack(xs), dtype=torch.float32, device=device)
    if edge_indices:
        edge_index = torch.as_tensor(np.concatenate(edge_indices, axis=1), dtype=torch.long, device=device)
        edge_attr = torch.as_tensor(np.vstack(edge_attrs), dtype=torch.float32, device=device)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        edge_attr = torch.zeros((0, BOND_DIM), dtype=torch.float32, device=device)
    batch = torch.as_tensor(np.concatenate(batch_ids), dtype=torch.long, device=device)
    y = torch.as_tensor(np.asarray(labels), dtype=torch.float32, device=device)
    for name, value in (("x", x), ("edge_index", edge_index), ("edge_attr", edge_attr), ("batch", batch), ("y", y)):
        assert_cuda_tensor(value, f"graph_batch.{name}")
    return GraphBatch(x, edge_index, edge_attr, batch, y)


def iter_batches(indices: np.ndarray, batch_size: int, shuffle: bool, seed: int):
    order = np.asarray(indices, dtype=np.int64).copy()
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


def mean_max_pool(h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    graph_count = int(batch.max().item()) + 1
    sums = h.new_zeros((graph_count, h.shape[1]))
    sums.index_add_(0, batch, h)
    counts = torch.bincount(batch, minlength=graph_count).clamp_min(1).to(h.dtype).unsqueeze(1)
    means = sums / counts
    maxima = torch.stack([h[batch == index].max(dim=0).values for index in range(graph_count)])
    return torch.cat([means, maxima], dim=1)


class GCNLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, edge_index, edge_attr):
        output = self.self_linear(h)
        if edge_index.numel():
            source, target = edge_index
            aggregate = h.new_zeros(h.shape)
            aggregate.index_add_(0, target, self.neighbor_linear(h[source]))
            degree = h.new_zeros((h.shape[0], 1))
            degree.index_add_(0, target, torch.ones((len(target), 1), device=h.device))
            output = output + aggregate / degree.clamp_min(1.0)
        return self.dropout(F.relu(self.norm(output)))


class GATLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attention_source = nn.Parameter(torch.empty(hidden_dim))
        self.attention_target = nn.Parameter(torch.empty(hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.attention_source.unsqueeze(0))
        nn.init.xavier_uniform_(self.attention_target.unsqueeze(0))

    def forward(self, h, edge_index, edge_attr):
        node_count = len(h)
        loops = torch.arange(node_count, device=h.device)
        if edge_index.numel():
            source = torch.cat([edge_index[0], loops])
            target = torch.cat([edge_index[1], loops])
        else:
            source = target = loops
        transformed = self.linear(h)
        score = F.leaky_relu(
            (transformed[source] * self.attention_source).sum(-1)
            + (transformed[target] * self.attention_target).sum(-1),
            negative_slope=0.2,
        )
        maximum = h.new_full((node_count,), -1e9)
        maximum.scatter_reduce_(0, target, score, reduce="amax", include_self=True)
        exponent = torch.exp(score - maximum[target])
        denominator = h.new_zeros((node_count,))
        denominator.index_add_(0, target, exponent)
        attention = self.dropout(exponent / denominator[target].clamp_min(1e-9))
        aggregate = h.new_zeros(transformed.shape)
        aggregate.index_add_(0, target, transformed[source] * attention.unsqueeze(1))
        return F.elu(self.norm(aggregate))


class MPNNLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden_dim + BOND_DIM, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.GRUCell(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, edge_index, edge_attr):
        aggregate = h.new_zeros(h.shape)
        if edge_index.numel():
            source, target = edge_index
            messages = self.message(torch.cat([h[source], edge_attr], dim=1))
            aggregate.index_add_(0, target, messages)
        return self.norm(self.update(aggregate, h))


class PooledGraphClassifier(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def classify(self, h, batch):
        return self.classifier(mean_max_pool(h, batch)).squeeze(1)


class GCN(PooledGraphClassifier):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__(hidden_dim, dropout)
        self.encoder = nn.Sequential(nn.Linear(NODE_DIM, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.layers = nn.ModuleList([GCNLayer(hidden_dim, dropout) for _ in range(2)])

    def forward(self, graph: GraphBatch):
        h = self.encoder(graph.x)
        for layer in self.layers:
            h = layer(h, graph.edge_index, graph.edge_attr)
        return self.classify(h, graph.batch)


class GAT(PooledGraphClassifier):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__(hidden_dim, dropout)
        self.encoder = nn.Sequential(nn.Linear(NODE_DIM, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.layers = nn.ModuleList([GATLayer(hidden_dim, dropout) for _ in range(2)])

    def forward(self, graph: GraphBatch):
        h = self.encoder(graph.x)
        for layer in self.layers:
            h = layer(h, graph.edge_index, graph.edge_attr)
        return self.classify(h, graph.batch)


class MPNN(PooledGraphClassifier):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__(hidden_dim, dropout)
        self.encoder = nn.Sequential(nn.Linear(NODE_DIM, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.layers = nn.ModuleList([MPNNLayer(hidden_dim, dropout) for _ in range(3)])

    def forward(self, graph: GraphBatch):
        h = self.encoder(graph.x)
        for layer in self.layers:
            h = F.relu(layer(h, graph.edge_index, graph.edge_attr))
        return self.classify(h, graph.batch)


class AttentiveFP(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(NODE_DIM, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.layers = nn.ModuleList([MPNNLayer(hidden_dim, dropout) for _ in range(2)])
        self.gate = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, graph: GraphBatch):
        h = self.encoder(graph.x)
        for layer in self.layers:
            h = F.relu(layer(h, graph.edge_index, graph.edge_attr))
        graph_count = int(graph.batch.max().item()) + 1
        weights = torch.sigmoid(self.gate(h))
        numerator = h.new_zeros((graph_count, h.shape[1]))
        denominator = h.new_zeros((graph_count, 1))
        numerator.index_add_(0, graph.batch, h * weights)
        denominator.index_add_(0, graph.batch, weights)
        pooled = numerator / denominator.clamp_min(1e-6)
        return self.classifier(pooled).squeeze(1)


def build_model(architecture: str, hidden_dim: int, dropout: float) -> nn.Module:
    models = {"GCN": GCN, "GAT": GAT, "MPNN": MPNN, "AttentiveFP": AttentiveFP}
    if architecture not in models:
        raise ValueError(architecture)
    model = models[architecture](hidden_dim, dropout).to(require_cuda_runtime())
    assert_cuda_module(model, architecture)
    return model


def predict_raw(
    model: nn.Module,
    records: list[GraphRecord],
    indices: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    device = require_cuda_runtime()
    assert_cuda_module(model, "graph_model")
    outputs = []
    model.eval()
    with torch.no_grad():
        for batch_indices in iter_batches(indices, batch_size, False, 0):
            logits = model(collate(records, batch_indices, device))
            assert_cuda_tensor(logits, "graph_logits")
            outputs.append(logits.detach().cpu().numpy())
    output = np.concatenate(outputs).astype(np.float64, copy=False)
    if not np.isfinite(output).all():
        raise RuntimeError("Graph model produced non-finite logits")
    return output


def _train_fixed_epochs(
    records: list[GraphRecord],
    train_indices: np.ndarray,
    architecture: str,
    hidden_dim: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    seed: int,
) -> nn.Module:
    device = require_cuda_runtime()
    set_reproducible_seed(seed)
    model = build_model(architecture, hidden_dim, dropout)
    labels = np.asarray([records[int(index)].y for index in train_indices], dtype=np.float32)
    positives = max(float(labels.sum()), 1.0)
    negatives = max(float(len(labels) - labels.sum()), 1.0)
    positive_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch in range(1, max(1, epochs) + 1):
        model.train()
        for batch_indices in iter_batches(train_indices, batch_size, True, seed + epoch):
            batch = collate(records, batch_indices, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = loss_function(logits, batch.y)
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("Graph training loss is non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    assert_cuda_module(model, architecture)
    return model


def train_with_isolated_early_stop(
    records: list[GraphRecord],
    train_indices: np.ndarray,
    split_group_ids: np.ndarray,
    architecture: str,
    parameters: dict,
    seed: int,
) -> tuple[nn.Module, dict]:
    labels = np.asarray([records[int(index)].y for index in train_indices], dtype=int)
    local = np.arange(len(train_indices), dtype=np.int64)
    groups = np.asarray(split_group_ids, dtype=object)
    if len(groups) != len(train_indices):
        raise ValueError("Early-stop group vector length mismatch")
    selected_early_seed = None
    fit_pos = stop_pos = None
    minimum_stop_units = min(20, max(5, len(train_indices) // 10))
    for offset in range(int(parameters.get("early_stop_seed_candidates", 256))):
        candidate_seed = seed + 700000 + offset
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=float(parameters.get("early_stop_validation_fraction", 0.2)),
            random_state=candidate_seed,
        )
        candidate_fit, candidate_stop = next(splitter.split(local, labels, groups))
        if len(candidate_stop) < minimum_stop_units:
            continue
        if set(labels[candidate_fit]) != {0, 1} or set(labels[candidate_stop]) != {0, 1}:
            continue
        selected_early_seed = candidate_seed
        fit_pos, stop_pos = candidate_fit, candidate_stop
        break
    if selected_early_seed is None or fit_pos is None or stop_pos is None:
        raise RuntimeError("No eligible group-disjoint early-stop partition found")
    fit_indices = train_indices[fit_pos]
    stop_indices = train_indices[stop_pos]
    if set(groups[fit_pos]) & set(groups[stop_pos]):
        raise RuntimeError("Early-stop split group leakage")
    if set(labels[fit_pos]) != {0, 1} or set(labels[stop_pos]) != {0, 1}:
        raise RuntimeError("Early-stop partition lacks a class")

    device = require_cuda_runtime()
    set_reproducible_seed(seed)
    selection_model = build_model(
        architecture,
        int(parameters["hidden_dim"]),
        float(parameters["dropout"]),
    )
    fit_labels = np.asarray([records[int(index)].y for index in fit_indices], dtype=np.float32)
    positives = max(float(fit_labels.sum()), 1.0)
    negatives = max(float(len(fit_labels) - fit_labels.sum()), 1.0)
    positive_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(
        selection_model.parameters(),
        lr=float(parameters["learning_rate"]),
        weight_decay=float(parameters["weight_decay"]),
    )
    best_auc = -np.inf
    best_epoch = 1
    no_improvement = 0
    history = []
    for epoch in range(1, int(parameters["max_epochs"]) + 1):
        selection_model.train()
        for batch_indices in iter_batches(
            fit_indices,
            int(parameters["batch_size"]),
            True,
            seed + epoch,
        ):
            batch = collate(records, batch_indices, device)
            optimizer.zero_grad(set_to_none=True)
            logits = selection_model(batch)
            loss = loss_function(logits, batch.y)
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("Graph early-stop training loss is non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(selection_model.parameters(), 5.0)
            optimizer.step()
        stop_raw = predict_raw(selection_model, records, stop_indices, int(parameters["batch_size"]))
        stop_labels = np.asarray([records[int(index)].y for index in stop_indices], dtype=int)
        auc = float(roc_auc_score(stop_labels, stop_raw))
        history.append({"epoch": epoch, "early_stop_roc_auc": auc})
        if auc > best_auc + 1e-6:
            best_auc = auc
            best_epoch = epoch
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= int(parameters["patience"]):
            break
    del selection_model
    torch.cuda.empty_cache()
    final_model = _train_fixed_epochs(
        records,
        train_indices,
        architecture,
        int(parameters["hidden_dim"]),
        float(parameters["dropout"]),
        float(parameters["learning_rate"]),
        float(parameters["weight_decay"]),
        best_epoch,
        int(parameters["batch_size"]),
        seed + 1,
    )
    evidence = {
        "backend": "native_pytorch_cuda",
        "architecture": architecture,
        "device": str(next(final_model.parameters()).device),
        "seed": seed,
        "best_epoch": best_epoch,
        "early_stop_best_roc_auc": best_auc,
        "early_stop_group_overlap_count": 0,
        "early_stop_seed": selected_early_seed,
        "early_stop_fit_units": int(len(fit_indices)),
        "early_stop_validation_units": int(len(stop_indices)),
        "early_stop_minimum_validation_units": int(minimum_stop_units),
        "early_stop_validation_ids_sha256": hashlib.sha256(
            "\n".join(records[int(index)].model_unit_id for index in sorted(stop_indices)).encode()
        ).hexdigest(),
        "history": history,
    }
    return final_model, evidence
