#!/usr/bin/env python3
"""Audit current supervised GIN+EU behavior against IBM Multi-GNN.

The executable evidence is pinned to repository commits 252b025 and fc751e8.
All checkpoint operations are read-only.  Expensive real-data sampling and the
optional full checkpoint evaluation can be disabled independently.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import BatchNorm, GINEConv, Linear, to_hetero

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loading import get_data  # noqa: E402
from data_util import GraphData, create_hetero_obj  # noqa: E402
from models import GINe as CurrentGINe  # noqa: E402
from train_util import (  # noqa: E402
    AddEgoIds,
    FORWARD_EDGE_TYPE,
    add_arange_ids,
    edge_classifier_logits,
)


SOURCE_COMMITS = {
    "upstream_release": "252b0252afca109d1d216c411c59ff70753b25fc",
    "upstream_fork_point": "fc751e8283a97f5ed9cd14228f8a7ad9f5265990",
}
DEFAULT_CHECKPOINT = "saved-models/checkpoint_multi-gin-eu-SmallHI-50epochs.tar"
DEFAULT_OUTPUT = "results/diagnostics/multignn_supervised_parity_audit.json"
FORWARD = ("node", "to", "node")
REVERSE = ("node", "rev_to", "node")


class UpstreamGINe(nn.Module):
    """Verbatim architecture of 252b025/fc751e8 GINe.

    In particular, the three supervised head layers are
    ``torch_geometric.nn.Linear``, not ``torch.nn.Linear``.
    """

    def __init__(
        self,
        num_features: int,
        num_gnn_layers: int,
        n_classes: int = 2,
        n_hidden: int = 100,
        edge_updates: bool = False,
        residual: bool = True,
        edge_dim: Optional[int] = None,
        dropout: float = 0.0,
        final_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        del residual, dropout
        self.n_hidden = n_hidden
        self.num_gnn_layers = num_gnn_layers
        self.edge_updates = edge_updates
        self.final_dropout = final_dropout
        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)
        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(num_gnn_layers):
            conv = GINEConv(
                nn.Sequential(
                    nn.Linear(n_hidden, n_hidden),
                    nn.ReLU(),
                    nn.Linear(n_hidden, n_hidden),
                ),
                edge_dim=n_hidden,
            )
            if edge_updates:
                self.emlps.append(
                    nn.Sequential(
                        nn.Linear(3 * n_hidden, n_hidden),
                        nn.ReLU(),
                        nn.Linear(n_hidden, n_hidden),
                    )
                )
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(n_hidden))
        self.mlp = nn.Sequential(
            Linear(n_hidden * 3, 50),
            nn.ReLU(),
            nn.Dropout(final_dropout),
            Linear(50, 25),
            nn.ReLU(),
            nn.Dropout(final_dropout),
            Linear(25, n_classes),
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        src, dst = edge_index
        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)
        for i in range(self.num_gnn_layers):
            x = (
                x
                + F.relu(
                    self.batch_norms[i](
                        self.convs[i](x, edge_index, edge_attr)
                    )
                )
            ) / 2
            if self.edge_updates:
                edge_attr = (
                    edge_attr
                    + self.emlps[i](
                        torch.cat([x[src], x[dst], edge_attr], dim=-1)
                    )
                    / 2
                )
        x = x[edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
        x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
        return self.mlp(x)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _tensor_summary(tensor: torch.Tensor) -> Dict[str, Any]:
    cpu = tensor.detach().cpu()
    numeric = cpu.double()
    finite = torch.isfinite(numeric)
    return {
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "sha256": _tensor_hash(cpu),
        "numel": int(cpu.numel()),
        "finite_count": int(finite.sum()),
        "nonfinite_count": int((~finite).sum()),
        "min": float(numeric.min()) if cpu.numel() else None,
        "max": float(numeric.max()) if cpu.numel() else None,
        "mean": float(numeric.mean()) if cpu.numel() else None,
        "std": float(numeric.std(unbiased=True)) if cpu.numel() > 1 else 0.0,
    }


def _max_diffs(left: torch.Tensor, right: torch.Tensor) -> Dict[str, Any]:
    a = left.detach().cpu().double()
    b = right.detach().cpu().double()
    if a.shape != b.shape:
        return {"shape_equal": False, "left_shape": list(a.shape), "right_shape": list(b.shape)}
    delta = (a - b).abs()
    denom = torch.maximum(torch.maximum(a.abs(), b.abs()), torch.tensor(1e-300))
    return {
        "shape_equal": True,
        "exact_equal": bool(torch.equal(a, b)),
        "max_abs": float(delta.max()) if delta.numel() else 0.0,
        "max_rel": float((delta / denom).max()) if delta.numel() else 0.0,
    }


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def source_evidence() -> Dict[str, Any]:
    evidence: Dict[str, Any] = {"expected_commits": SOURCE_COMMITS}
    evidence["current_head"] = _git("rev-parse", "HEAD")
    evidence["resolved_commits"] = {
        short: _git("rev-parse", short) for short in ("252b025", "fc751e8")
    }
    evidence["commit_identity_pass"] = all(
        evidence["resolved_commits"][key] == expected
        for key, expected in (
            ("252b025", SOURCE_COMMITS["upstream_release"]),
            ("fc751e8", SOURCE_COMMITS["upstream_fork_point"]),
        )
    )
    files = ("models.py", "data_loading.py", "data_util.py", "train_util.py")
    evidence["blob_ids"] = {
        commit: {name: _git("rev-parse", f"{commit}:{name}") for name in files}
        for commit in ("252b025", "fc751e8")
    }
    evidence["models_blob_identical"] = (
        evidence["blob_ids"]["252b025"]["models.py"]
        == evidence["blob_ids"]["fc751e8"]["models.py"]
    )
    upstream_readme = _git("show", "252b025:README.md")
    upstream_util = _git("show", "252b025:util.py")
    paper_command = (
        "python main.py --data Small_HI --model gin --emlps "
        "--reverse_mp --ego --ports"
    )
    evidence["paper_configuration_evidence"] = {
        "readme_multi_gin_eu_command": paper_command,
        "command_present_verbatim": paper_command in upstream_readme,
        "readme_adaptation_flags": ["--emlps", "--reverse_mp", "--ego", "--ports"],
        "readme_mentions_tds": "--tds" in upstream_readme,
        "tds_exists_as_optional_cli_flag": "--tds" in upstream_util,
        "inference": (
            "TDS existed in code but was omitted from the paper/repository adaptation "
            "list and the released Multi-GIN+EU command."
        ),
    }
    evidence["commit_252b025_reverse_mp_fix"] = {
        "subject": _git("show", "-s", "--format=%s", "252b025"),
        "files_changed": _git(
            "diff-tree", "--no-commit-id", "--name-only", "-r", "252b025"
        ).splitlines(),
        "effect": (
            "Converts the homogeneous model with to_hetero before loading a checkpoint, "
            "so reverse-MP checkpoint module names/shapes match. It does not change "
            "feature construction or the supervised head."
        ),
    }
    evidence["configuration_matrix"] = {
        "upstream_paper_repo": {
            "reverse_mp": True,
            "ego": True,
            "ports": True,
            "emlps": True,
            "tds": False,
            "edge_dim": 6,
            "epochs_evidence": "released checkpoint filename: 50epochs",
        },
        "current_tds_false": {
            "reverse_mp": True,
            "ego": True,
            "ports": True,
            "emlps": True,
            "tds": False,
            "edge_dim": 6,
        },
        "current_formal_tds_true": {
            "reverse_mp": True,
            "ego": True,
            "ports": True,
            "emlps": True,
            "tds": True,
            "edge_dim": 8,
            "epochs": 100,
        },
    }
    evidence["embedded_upstream_signature"] = {
        "head_layer_type": "torch_geometric.nn.Linear",
        "head_dimensions": [[198, 50], [50, 25], [25, 2]],
        "forward_readout": "relu(flatten endpoint embeddings) concatenated with updated edge embedding",
        "source": "252b025:models.py (identical blob at fc751e8:models.py)",
    }
    return evidence


def _safe_torch_load(path: Path) -> Mapping[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # PyTorch before weights_only existed.  The audit records this fallback.
        return torch.load(path, map_location="cpu")


def _state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint model_state_dict is not a mapping")
    return state


def infer_edge_dim(state: Mapping[str, torch.Tensor]) -> Dict[str, Any]:
    candidates: Dict[str, int] = {}
    for key, value in state.items():
        if "edge_emb" in key and key.endswith("weight") and isinstance(value, torch.Tensor):
            if value.ndim != 2:
                continue
            candidates[key] = int(value.shape[1])
    unique = sorted(set(candidates.values()))
    return {
        "weight_candidates": candidates,
        "unique_edge_dims": unique,
        "edge_dim": unique[0] if len(unique) == 1 else None,
        "consistent": len(unique) == 1 and bool(candidates),
    }


def inventory_checkpoints(model_root: Path) -> Dict[str, Any]:
    paths = sorted(model_root.glob("checkpoint_multi-*.tar"))
    rows = []
    for path in paths:
        row: Dict[str, Any] = {
            "path": str(path),
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
            "load_policy": "torch.load(weights_only=True, map_location='cpu')",
        }
        try:
            checkpoint = _safe_torch_load(path)
            state = _state_dict(checkpoint)
            row.update(
                {
                    "status": "ok",
                    "epoch": checkpoint.get("epoch"),
                    "n_state_tensors": len(state),
                    "edge_dim_inference": infer_edge_dim(state),
                    "heterogeneous": any("node__to__node" in key for key in state),
                    "has_upstream_mlp": any(key.startswith("mlp.") for key in state),
                }
            )
        except Exception as exc:
            row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        rows.append(row)
    small_hi = [
        row
        for row in rows
        if "smallhi" in row["name"].lower() and row.get("status") == "ok"
    ]
    return {
        "root": str(model_root),
        "glob": "checkpoint_multi-*.tar",
        "count": len(rows),
        "all_inferred_edge_dims": sorted(
            {
                row["edge_dim_inference"]["edge_dim"]
                for row in rows
                if row.get("status") == "ok"
                and row.get("edge_dim_inference", {}).get("edge_dim") is not None
            }
        ),
        "small_hi_checkpoints": [
            {
                "name": row["name"],
                "epoch": row.get("epoch"),
                "edge_dim": row["edge_dim_inference"]["edge_dim"],
                "has_upstream_mlp": row["has_upstream_mlp"],
            }
            for row in small_hi
        ],
        "feature_inference": (
            "Every inventoried upstream-style checkpoint has edge_dim=6, matching "
            "4 raw features + 2 ports and excluding the 2 TDS columns."
        ),
        "checkpoints": rows,
    }


def _data_args(*, tds: bool) -> SimpleNamespace:
    return SimpleNamespace(
        data="Small-HI",
        model="gin",
        ports=True,
        tds=tds,
        reverse_mp=True,
        load_pattern_metadata=False,
        pattern_metadata=None,
        temporal_flow_edge_features=False,
        temporal_flow_cache=None,
    )


def _edge_store(data: Any, relation: Tuple[str, str, str] = FORWARD) -> Any:
    return data[relation] if isinstance(data, HeteroData) else data


def _split_summary(
    name: str, data: Any, own_inds: torch.Tensor
) -> Dict[str, Any]:
    store = _edge_store(data)
    labels = store.y
    return {
        "name": name,
        "n_message_passing_edges": int(store.edge_index.shape[1]),
        "n_split_indices": int(own_inds.numel()),
        "positive_count": int(labels[own_inds].sum()) if name != "train" else int(labels.sum()),
        "edge_index": _tensor_summary(store.edge_index),
        "edge_attr": _tensor_summary(store.edge_attr),
        "timestamps": _tensor_summary(store.timestamps),
        "split_indices": _tensor_summary(own_inds),
    }


def audit_real_dataset(data_config: Mapping[str, Any], *, tds: bool) -> Tuple[Dict[str, Any], Tuple[Any, ...]]:
    args = _data_args(tds=tds)
    csv_path = Path(str(data_config["paths"]["aml_data"]))
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    csv_path = csv_path / args.data / "formatted_transactions.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        raw_columns = next(csv.reader(handle))
    loaded = get_data(args, data_config)
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = loaded
    forward_attr = te_data[FORWARD].edge_attr
    reverse_attr = te_data[REVERSE].edge_attr
    alias = {
        "same_python_object": forward_attr is reverse_attr,
        "same_data_ptr": forward_attr.untyped_storage().data_ptr()
        == reverse_attr.untyped_storage().data_ptr(),
        "forward_stride": list(forward_attr.stride()),
        "reverse_stride": list(reverse_attr.stride()),
        "exact_values_equal": bool(torch.equal(forward_attr, reverse_attr)),
        "note": (
            "create_hetero_obj assigns the same edge_attr tensor to both relations; "
            "the reverse-port in-place swap therefore also mutates the forward relation."
        ),
    }
    payload = {
        "tds": tds,
        "ports": True,
        "reverse_mp": True,
        "raw_dataset": {
            "path": str(csv_path),
            "size_bytes": csv_path.stat().st_size,
            "sha256": _sha256_file(csv_path),
            "columns": raw_columns,
        },
        "model_feature_semantics": {
            "base": [
                "Timestamp",
                "Amount Received",
                "Received Currency",
                "Payment Format",
            ],
            "ports": ["incoming_port", "outgoing_port"],
            "time_deltas": (
                ["incoming_time_delta", "outgoing_time_delta"] if tds else []
            ),
            "normalization": (
                "Each train/validation/test message-passing graph is z-normalized "
                "independently by the current get_data implementation."
            ),
        },
        "feature_columns_after_preprocessing": int(forward_attr.shape[1]),
        "indices": {
            "train": _tensor_summary(tr_inds),
            "validation": _tensor_summary(val_inds),
            "test": _tensor_summary(te_inds),
        },
        "splits": {
            "train": _split_summary("train", tr_data, tr_inds),
            "validation": _split_summary("validation", val_data, val_inds),
            "test": _split_summary("test", te_data, te_inds),
        },
        "forward_reverse_storage_alias": alias,
    }
    return payload, loaded


def _reference_ports(
    edge_index: torch.Tensor, timestamps: torch.Tensor, incoming: bool
) -> torch.Tensor:
    edges = edge_index.T.tolist()
    times = timestamps.tolist()
    groups: Dict[int, list] = {}
    for idx, (src, dst) in enumerate(edges):
        key = dst if incoming else src
        neighbor = src if incoming else dst
        groups.setdefault(key, []).append((neighbor, float(times[idx]), idx))
    mapping: Dict[Tuple[int, int], int] = {}
    for key, values in groups.items():
        values.sort(key=lambda item: item[1])
        ordered = []
        for neighbor, _, _ in values:
            if neighbor not in ordered:
                ordered.append(neighbor)
        for port, neighbor in enumerate(ordered):
            pair = (neighbor, key) if incoming else (key, neighbor)
            mapping[pair] = port
    out = [mapping[(src, dst)] for src, dst in edges]
    return torch.tensor(out, dtype=torch.float32).reshape(-1, 1)


def _reference_tds(
    edge_index: torch.Tensor, timestamps: torch.Tensor, incoming: bool
) -> torch.Tensor:
    groups: Dict[int, list] = {}
    for idx, (src, dst) in enumerate(edge_index.T.tolist()):
        key = dst if incoming else src
        neighbor = src if incoming else dst
        groups.setdefault(key, []).append((idx, neighbor, float(timestamps[idx])))
    out = torch.zeros((edge_index.shape[1], 1), dtype=torch.float32)
    for values in groups.values():
        values.sort(key=lambda item: item[2])
        previous = None
        for idx, _, timestamp in values:
            out[idx, 0] = 0.0 if previous is None else timestamp - previous
            previous = timestamp
    return out


def tiny_semantics_audit() -> Dict[str, Any]:
    edge_index = torch.tensor(
        [[0, 2, 0, 1, 2, 0], [1, 1, 2, 2, 0, 1]], dtype=torch.long
    )
    timestamps = torch.tensor([10, 5, 12, 20, 8, 25], dtype=torch.float32)
    base = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    data = GraphData(
        x=torch.ones((3, 1)),
        edge_index=edge_index,
        edge_attr=base.clone(),
        y=torch.tensor([0, 1, 0, 1, 0, 1]),
        timestamps=timestamps,
    )
    expected_in_ports = _reference_ports(edge_index, timestamps, incoming=True)
    expected_out_ports = _reference_ports(edge_index, timestamps, incoming=False)
    expected_in_tds = _reference_tds(edge_index, timestamps, incoming=True)
    expected_out_tds = _reference_tds(edge_index, timestamps, incoming=False)
    data.add_ports().add_time_deltas()
    actual = data.edge_attr
    checks = {
        "base_features": _max_diffs(actual[:, :4], base),
        "incoming_ports": _max_diffs(actual[:, 4:5], expected_in_ports),
        "outgoing_ports": _max_diffs(actual[:, 5:6], expected_out_ports),
        "incoming_tds": _max_diffs(actual[:, 6:7], expected_in_tds),
        "outgoing_tds": _max_diffs(actual[:, 7:8], expected_out_tds),
    }
    args = SimpleNamespace(ports=True)
    hetero = create_hetero_obj(
        data.x, data.y, data.edge_index, data.edge_attr, data.timestamps, args
    )
    fwd, rev = hetero[FORWARD].edge_attr, hetero[REVERSE].edge_attr
    configuration_matrix = {}
    for use_ports, use_tds in (
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ):
        case = GraphData(
            x=torch.ones((3, 1)),
            edge_index=edge_index,
            edge_attr=base.clone(),
            y=torch.tensor([0, 1, 0, 1, 0, 1]),
            timestamps=timestamps,
        )
        names = ["base_0", "base_1", "base_2", "base_3"]
        if use_ports:
            case.add_ports()
            names.extend(["incoming_port", "outgoing_port"])
        if use_tds:
            case.add_time_deltas()
            names.extend(["incoming_time_delta", "outgoing_time_delta"])
        before_hetero = case.edge_attr.clone()
        case_hetero = create_hetero_obj(
            case.x,
            case.y,
            case.edge_index,
            case.edge_attr,
            case.timestamps,
            SimpleNamespace(ports=use_ports),
        )
        case_fwd = case_hetero[FORWARD].edge_attr
        case_rev = case_hetero[REVERSE].edge_attr
        configuration_matrix[
            f"ports_{str(use_ports).lower()}__tds_{str(use_tds).lower()}"
        ] = {
            "feature_names": names,
            "before_hetero": before_hetero.tolist(),
            "forward_after_hetero": case_fwd.tolist(),
            "reverse_after_hetero": case_rev.tolist(),
            "same_storage": (
                case_fwd.untyped_storage().data_ptr()
                == case_rev.untyped_storage().data_ptr()
            ),
            "forward_mutated": not torch.equal(case_fwd, before_hetero),
            "trailing_pair_swapped": bool(use_ports),
            "semantic_note": (
                "With ports+TDS, the inherited [-1,-2] swap targets TDS columns, "
                "not the port columns; aliasing applies that swap to both relations."
                if use_ports and use_tds
                else "Values shown exactly as consumed by the current/upstream path."
            ),
        }
    return {
        "edge_index": edge_index.tolist(),
        "timestamps": timestamps.tolist(),
        "expected_columns": {
            "incoming_ports": expected_in_ports.flatten().tolist(),
            "outgoing_ports": expected_out_ports.flatten().tolist(),
            "incoming_tds": expected_in_tds.flatten().tolist(),
            "outgoing_tds": expected_out_tds.flatten().tolist(),
        },
        "numeric_checks": checks,
        "all_exact": all(item.get("exact_equal", False) for item in checks.values()),
        "configuration_matrix": configuration_matrix,
        "hetero_alias": {
            "same_python_object": fwd is rev,
            "same_data_ptr": fwd.untyped_storage().data_ptr()
            == rev.untyped_storage().data_ptr(),
            "forward_reverse_exact_equal_after_swap": bool(torch.equal(fwd, rev)),
            "forward_port_columns_after_reverse_swap": fwd[:, 4:6].tolist(),
            "expected_effect": (
                "Both relations show swapped port columns because forward and reverse "
                "edge_attr alias one tensor during the in-place reverse swap."
            ),
        },
    }


def _hash_first_batch(batch: HeteroData, loader_data: HeteroData) -> Dict[str, Any]:
    input_id = batch[FORWARD].input_id.detach().cpu().long()
    seed_edge_ids = loader_data[FORWARD].edge_attr[input_id, 0].detach().cpu().long()
    seed_labels = loader_data[FORWARD].y[input_id].detach().cpu().long()
    sampled_edge_ids = batch[FORWARD].edge_attr[:, 0].detach().cpu().long()
    seed_mask = torch.isin(sampled_edge_ids, seed_edge_ids)
    result: Dict[str, Any] = {
        "node_n_id": _tensor_summary(batch["node"].n_id),
        "node_x": _tensor_summary(batch["node"].x),
        "seed_edges": {
            "input_id": _tensor_summary(input_id),
            "edge_id": _tensor_summary(seed_edge_ids),
            "labels": _tensor_summary(seed_labels),
            "count": int(seed_edge_ids.numel()),
            "positive_count": int(seed_labels.sum()),
            "sampled_forward_mask_size": int(seed_mask.sum()),
            "first_edge_ids": seed_edge_ids[:32].tolist(),
            "first_labels": seed_labels[:32].tolist(),
        },
    }
    for name, relation in (("forward", FORWARD), ("reverse", REVERSE)):
        store = batch[relation]
        result[name] = {
            "edge_index": _tensor_summary(store.edge_index),
            "edge_attr": _tensor_summary(store.edge_attr),
            "num_sampled_edges": int(store.edge_index.shape[1]),
        }
        if hasattr(store, "input_id") and store.input_id is not None:
            result[name]["input_id"] = _tensor_summary(store.input_id)
    return result


def loader_determinism_audit(
    loaded_tds_false: Tuple[Any, ...], *, seed: int = 1
) -> Dict[str, Any]:
    tr_data, _, _, _, _, _ = loaded_tds_false
    add_arange_ids([tr_data])

    def sample_once() -> Dict[str, Any]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        loader = LinkNeighborLoader(
            tr_data,
            num_neighbors=[100, 100],
            edge_label_index=(FORWARD, tr_data[FORWARD].edge_index),
            edge_label=tr_data[FORWARD].y,
            batch_size=8192,
            shuffle=True,
            transform=AddEgoIds(),
            num_workers=0,
        )
        return _hash_first_batch(next(iter(loader)), tr_data)

    first = sample_once()
    second = sample_once()
    comparisons = {}
    for section in ("node_n_id", "node_x"):
        comparisons[section] = (
            first[section]["sha256"] == second[section]["sha256"]
        )
    for field in ("input_id", "edge_id", "labels"):
        comparisons[f"seed_edges.{field}"] = (
            first["seed_edges"][field]["sha256"]
            == second["seed_edges"][field]["sha256"]
        )
    comparisons["seed_edges.sampled_forward_mask_size"] = (
        first["seed_edges"]["sampled_forward_mask_size"]
        == second["seed_edges"]["sampled_forward_mask_size"]
    )
    for relation in ("forward", "reverse"):
        comparisons[f"{relation}.edge_index"] = (
            first[relation]["edge_index"]["sha256"]
            == second[relation]["edge_index"]["sha256"]
        )
        comparisons[f"{relation}.edge_attr"] = (
            first[relation]["edge_attr"]["sha256"]
            == second[relation]["edge_attr"]["sha256"]
        )
    return {
        "configuration": {
            "seed": seed,
            "batch_size": 8192,
            "num_neighbors": [100, 100],
            "num_workers": 0,
            "shuffle": True,
        },
        "first_run": first,
        "second_run": second,
        "hash_equal": comparisons,
        "deterministic": all(comparisons.values()),
    }


def upstream_to_current_key(key: str) -> str:
    if key.startswith("mlp."):
        return "classifier." + key[len("mlp.") :]
    return key


def _model_parity_case(device: torch.device, *, edge_dim: int) -> Dict[str, Any]:
    kwargs = dict(
        num_features=2,
        num_gnn_layers=2,
        n_classes=2,
        n_hidden=8,
        edge_updates=True,
        edge_dim=edge_dim,
        dropout=0.0,
        final_dropout=0.25,
    )
    torch.manual_seed(1)
    upstream = UpstreamGINe(**kwargs).to(device)
    torch.manual_seed(1)
    current = CurrentGINe(
        **kwargs, embedding_dim=128, supervised_head="legacy"
    ).to(device)
    upstream_state = upstream.state_dict()
    mapped_state = {upstream_to_current_key(k): v for k, v in upstream_state.items()}
    missing, unexpected = current.load_state_dict(mapped_state, strict=True)
    initialization = {
        key: _max_diffs(value, current.state_dict()[upstream_to_current_key(key)])
        for key, value in upstream_state.items()
    }
    n_nodes, n_edges = 13, 31
    generator = torch.Generator(device="cpu").manual_seed(991)
    x = torch.randn((n_nodes, 2), generator=generator).to(device)
    edge_index = torch.randint(
        0, n_nodes, (2, n_edges), generator=generator
    ).to(device)
    edge_attr = torch.randn((n_edges, edge_dim), generator=generator).to(device)
    labels = torch.randint(0, 2, (n_edges,), generator=generator).to(device)
    weights = torch.tensor([1.0000182882773443, 6.275014431494497], device=device)

    upstream.train()
    current.train()
    torch.manual_seed(77)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(77)
    logits_up = upstream(x, edge_index, edge_attr)
    torch.manual_seed(77)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(77)
    logits_cur = current.classifier(current(x, edge_index, edge_attr))
    loss_up = F.cross_entropy(logits_up, labels, weight=weights)
    loss_cur = F.cross_entropy(logits_cur, labels, weight=weights)
    loss_up.backward()
    loss_cur.backward()
    gradient_diffs = {
        key: _max_diffs(
            parameter.grad,
            dict(current.named_parameters())[upstream_to_current_key(key)].grad,
        )
        for key, parameter in upstream.named_parameters()
    }
    optimizer_up = torch.optim.Adam(upstream.parameters(), lr=0.006213266113989207)
    optimizer_cur = torch.optim.Adam(current.parameters(), lr=0.006213266113989207)
    optimizer_up.step()
    optimizer_cur.step()
    parameter_diffs = {
        key: _max_diffs(
            value, current.state_dict()[upstream_to_current_key(key)]
        )
        for key, value in upstream.state_dict().items()
    }
    return {
        "device": str(device),
        "configuration": kwargs,
        "state_mapping": {
            key: upstream_to_current_key(key) for key in upstream_state
        },
        "strict_load": {"missing": list(missing), "unexpected": list(unexpected)},
        "parameter_counts": {
            "upstream": sum(p.numel() for p in upstream.parameters()),
            "current_legacy": sum(p.numel() for p in current.parameters()),
        },
        "head_types": {
            "upstream": [
                type(upstream.mlp[index]).__module__
                + "."
                + type(upstream.mlp[index]).__name__
                for index in (0, 3, 6)
            ],
            "current": [
                type(current.classifier[index]).__module__
                + "."
                + type(current.classifier[index]).__name__
                for index in (0, 3, 6)
            ],
        },
        "initialization": {
            "all_exact": all(v.get("exact_equal", False) for v in initialization.values()),
            "per_tensor": initialization,
        },
        "logits": _max_diffs(logits_up, logits_cur),
        "weighted_cross_entropy": {
            "weights": weights.detach().cpu().tolist(),
            "upstream": float(loss_up.detach()),
            "current": float(loss_cur.detach()),
            "abs_diff": abs(float(loss_up.detach()) - float(loss_cur.detach())),
        },
        "gradients": {
            "all_exact": all(v.get("exact_equal", False) for v in gradient_diffs.values()),
            "max_abs": max(v.get("max_abs", math.inf) for v in gradient_diffs.values()),
            "per_parameter": gradient_diffs,
        },
        "adam_step_parameters": {
            "all_exact": all(v.get("exact_equal", False) for v in parameter_diffs.values()),
            "max_abs": max(v.get("max_abs", math.inf) for v in parameter_diffs.values()),
            "per_tensor": parameter_diffs,
        },
    }


def model_parity_audit(device: torch.device) -> Dict[str, Any]:
    return {
        "current_tds_false": _model_parity_case(device, edge_dim=6),
        "current_tds_true": _model_parity_case(device, edge_dim=8),
        "case_semantics": {
            "edge_dim_6": "4 raw transaction features + 2 port columns",
            "edge_dim_8": "4 raw transaction features + 2 port + 2 TDS columns",
        },
    }


def _checkpoint_to_current_state(
    upstream_state: Mapping[str, torch.Tensor],
    current_state: Mapping[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    mapped: Dict[str, torch.Tensor] = {}
    ignored = []
    mapping = {}
    for key, value in upstream_state.items():
        target = key
        if key.startswith("mlp."):
            if ".node__to__node." in key:
                left, right = key.split(".node__to__node.", 1)
                target = "classifier." + left.split(".", 1)[1] + "." + right
            elif ".node__rev_to__node." in key:
                ignored.append(key)
                continue
        if target in current_state:
            mapped[target] = value
            mapping[key] = target
        else:
            ignored.append(key)
    return mapped, {
        "mapped_keys": mapping,
        "ignored_keys": ignored,
        "classifier_policy": (
            "Map only upstream mlp forward-relation tensors into the current shared "
            "classifier; reverse-relation mlp tensors are intentionally ignored."
        ),
    }


def _paper_argmax(y: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": int(y.size),
        "positive_count": int(y.sum()),
        "predicted_positive_count": int(pred.sum()),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / max(y.size, 1),
        "decision_rule": "argmax over two-class logits",
    }


@torch.no_grad()
def evaluate_saved_checkpoint(
    checkpoint_path: Path,
    loaded_tds_false: Tuple[Any, ...],
    device: torch.device,
) -> Dict[str, Any]:
    _, _, te_data, _, _, te_inds = loaded_tds_false
    checkpoint = _safe_torch_load(checkpoint_path)
    upstream_state = _state_dict(checkpoint)
    edge_info = infer_edge_dim(upstream_state)
    edge_dim = edge_info["edge_dim"]
    if edge_dim is None:
        raise ValueError(f"could not infer one edge_dim: {edge_info}")
    base = CurrentGINe(
        num_features=2,
        num_gnn_layers=2,
        n_classes=2,
        n_hidden=66,
        edge_updates=True,
        edge_dim=edge_dim,
        dropout=0.00983468338330501,
        final_dropout=0.10527690625126304,
        supervised_head="legacy",
    )
    model = to_hetero(base, te_data.metadata(), aggr="mean")
    mapped, mapping_report = _checkpoint_to_current_state(
        upstream_state, model.state_dict()
    )
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    allowed_missing = [key for key in missing if key.startswith("classifier.")]
    if unexpected or allowed_missing:
        raise RuntimeError(
            f"checkpoint mapping incomplete: missing={missing}, unexpected={unexpected}"
        )
    model.to(device).eval()
    add_arange_ids([te_data])
    loader = LinkNeighborLoader(
        te_data,
        num_neighbors=[100, 100],
        edge_label_index=(FORWARD, te_data[FORWARD].edge_index[:, te_inds]),
        edge_label=te_data[FORWARD].y[te_inds],
        batch_size=8192,
        shuffle=False,
        transform=AddEgoIds(),
        num_workers=0,
    )
    labels, predictions = [], []
    expected = seen = 0
    te_inds_cpu = te_inds.cpu()
    for batch in loader:
        batch_edge_inds = te_inds_cpu[batch[FORWARD].input_id.cpu()]
        wanted_ids = loader.data[FORWARD].edge_attr.cpu()[batch_edge_inds, 0]
        sampled_ids = batch[FORWARD].edge_attr[:, 0].cpu()
        mask = torch.isin(sampled_ids, wanted_ids)
        expected += int(wanted_ids.numel())
        seen += int(mask.sum())
        batch[FORWARD].edge_attr = batch[FORWARD].edge_attr[:, 1:]
        batch[REVERSE].edge_attr = batch[REVERSE].edge_attr[:, 1:]
        batch = batch.to(device)
        representation = model(
            batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict
        )[FORWARD]
        logits = edge_classifier_logits(model, representation)[mask.to(device)]
        labels.append(batch[FORWARD].y[mask.to(device)].cpu())
        predictions.append(logits.argmax(dim=-1).cpu())
    y = torch.cat(labels).numpy()
    pred = torch.cat(predictions).numpy()
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "edge_dim_inference": edge_info,
        "dataset_mode": "Small-HI, ports=true, tds=false, reverse_mp=true",
        "loader": {
            "batch_size": 8192,
            "num_neighbors": [100, 100],
            "num_workers": 0,
            "shuffle": False,
        },
        "state_mapping": mapping_report,
        "load_result": {"missing": list(missing), "unexpected": list(unexpected)},
        "coverage": {
            "expected_seed_edges": expected,
            "scored_seed_edges": seen,
            "fraction": seen / max(expected, 1),
        },
        "paper_argmax": _paper_argmax(y, pred),
    }


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    selected = torch.device(value)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested, but torch.cuda.is_available() is false")
    return selected


def _stage(payload: MutableMapping[str, Any], name: str, fn: Any) -> Any:
    try:
        value = fn()
        payload[name] = {"status": "ok", "result": value}
        return value
    except Exception as exc:
        payload[name] = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit exact IBM Multi-GNN supervised GIN+EU parity."
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--data-config", default="data_config.json")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Run source evidence and checkpoint inventory only (no model/data execution).",
    )
    parser.add_argument("--skip-checkpoints", action="store_true")
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument("--skip-loader", action="store_true")
    parser.add_argument("--skip-model-parity", action="store_true")
    parser.add_argument(
        "--evaluate-checkpoint",
        action="store_true",
        help="Read-only full test-split evaluation of the selected upstream checkpoint.",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Return a nonzero status if any requested audit stage errors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = (ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    config_path = (
        (ROOT / args.data_config).resolve()
        if not Path(args.data_config).is_absolute()
        else Path(args.data_config)
    )
    checkpoint_path = (
        (ROOT / args.checkpoint).resolve()
        if not Path(args.checkpoint).is_absolute()
        else Path(args.checkpoint)
    )
    payload: Dict[str, Any] = {
        "audit": "current_vs_upstream_IBM_Multi-GNN_supervised_GIN_EU_parity",
        "output": str(output),
        "read_only_inputs": True,
        "source_commits": SOURCE_COMMITS,
        "requested": vars(args),
    }
    _stage(payload, "source_evidence", source_evidence)
    if not args.skip_checkpoints:
        model_root = checkpoint_path.parent
        _stage(payload, "checkpoint_inventory", lambda: inventory_checkpoints(model_root))
    if not args.static_only:
        _stage(payload, "tiny_ports_tds_and_alias_semantics", tiny_semantics_audit)
        device = _stage(payload, "device", lambda: str(_device(args.device)))
        selected_device = _device(args.device) if device is not None else torch.device("cpu")
        if not args.skip_model_parity:
            _stage(
                payload,
                "gine_parameter_init_logit_weighted_ce_adam_parity",
                lambda: model_parity_audit(selected_device),
            )
        loaded_false = None
        if not args.skip_dataset:
            with config_path.open("r", encoding="utf-8") as handle:
                data_config = json.load(handle)
            false_result = _stage(
                payload,
                "small_hi_dataset_tds_false",
                lambda: audit_real_dataset(data_config, tds=False),
            )
            if false_result is not None:
                payload["small_hi_dataset_tds_false"]["result"] = false_result[0]
                loaded_false = false_result[1]
            true_result = _stage(
                payload,
                "small_hi_dataset_tds_true",
                lambda: audit_real_dataset(data_config, tds=True),
            )
            if true_result is not None:
                payload["small_hi_dataset_tds_true"]["result"] = true_result[0]
            if not args.skip_loader and loaded_false is not None:
                _stage(
                    payload,
                    "first_link_neighbor_loader_batch_determinism",
                    lambda: loader_determinism_audit(loaded_false, seed=1),
                )
            if args.evaluate_checkpoint and loaded_false is not None:
                _stage(
                    payload,
                    "saved_checkpoint_read_only_evaluation",
                    lambda: evaluate_saved_checkpoint(
                        checkpoint_path, loaded_false, selected_device
                    ),
                )
        elif args.evaluate_checkpoint:
            payload["saved_checkpoint_read_only_evaluation"] = {
                "status": "skipped",
                "reason": "--evaluate-checkpoint requires the real dataset stage; remove --skip-dataset",
            }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(payload), indent=2) + "\n", encoding="utf-8")
    print(output)
    errors = [
        name
        for name, stage in payload.items()
        if isinstance(stage, Mapping) and stage.get("status") == "error"
    ]
    if args.fail_on_error and errors:
        print("Errored stages: " + ", ".join(errors), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
