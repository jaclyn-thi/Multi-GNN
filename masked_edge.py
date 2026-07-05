"""
Label-free masked edge-attribute reconstruction pretraining (GraphMAE-style).

Masks selected transaction attribute fields on seed edges, runs the GNN on the
corrupted subgraph, and trains a lightweight decoder to reconstruct targets only
on masked positions. Train-split statistics define normalization and mask tokens.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData

from train_util import FORWARD_EDGE_TYPE, REVERSE_EDGE_TYPE

FIELD_ALIASES = {
    "amount": "amount",
    "amount_received": "amount",
    "currency": "currency",
    "received_currency": "currency",
    "payment_format": "payment_format",
    "payment": "payment_format",
    "timestamp": "timestamp",
}

# Column indices in edge_attr after stripping synthetic edge-id column.
FIELD_COL_INDEX = {
    "timestamp": 0,
    "amount": 1,
    "currency": 2,
    "payment_format": 3,
}


@dataclass
class MaskedEdgeBatchState:
    """Per-batch masking state for seed edges."""

    seed_mask_fwd: torch.Tensor
    field_masks: Dict[str, torch.Tensor]
    targets: Dict[str, torch.Tensor]
    stats: Dict[str, float] = field(default_factory=dict)


@dataclass
class MaskedEdgePretrainSpec:
    """Train-fit metadata for masking and reconstruction targets."""

    fields: Tuple[str, ...]
    token_strategy: str
    mask_rate: float
    loss_weights: Dict[str, float]
    amount_loss: str
    currency_classes: torch.Tensor
    payment_classes: torch.Tensor
    amount_log1p_mean: float
    amount_log1p_std: float
    timestamp_mean: float
    timestamp_std: float
    mask_tokens: Dict[str, float]
    learned_mask_tokens: Optional[nn.ParameterDict] = None

    @property
    def n_currency(self) -> int:
        return int(self.currency_classes.numel())

    @property
    def n_payment(self) -> int:
        return int(self.payment_classes.numel())


def parse_mask_fields(spec: str) -> Tuple[str, ...]:
    fields: List[str] = []
    for raw in spec.split(","):
        key = raw.strip().lower()
        if not key:
            continue
        if key not in FIELD_ALIASES:
            raise ValueError(
                f"Unsupported mask field {raw!r}; expected subset of "
                f"amount,currency,payment_format,timestamp"
            )
        canon = FIELD_ALIASES[key]
        if canon not in fields:
            fields.append(canon)
    if not fields:
        raise ValueError("At least one --mask_edge_attr_fields entry is required.")
    return tuple(fields)


def parse_loss_weights(spec: str, fields: Sequence[str]) -> Dict[str, float]:
    weights = {f: 1.0 for f in fields}
    if not spec.strip():
        return weights
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid --masked_edge_loss_weights entry: {part!r}")
        key, val = part.split("=", 1)
        canon = FIELD_ALIASES.get(key.strip().lower(), key.strip().lower())
        if canon not in weights:
            raise ValueError(f"Loss weight field {key!r} is not in --mask_edge_attr_fields")
        weights[canon] = float(val)
    return weights


def _forward_edge_attr(data: Data | HeteroData) -> torch.Tensor:
    if isinstance(data, HeteroData):
        return data[FORWARD_EDGE_TYPE].edge_attr
    return data.edge_attr


def build_masked_edge_spec(
    tr_data: Data | HeteroData,
    args,
    *,
    device: torch.device,
) -> MaskedEdgePretrainSpec:
    fields = parse_mask_fields(str(getattr(args, "mask_edge_attr_fields", "amount,currency,payment_format")))
    token_strategy = str(getattr(args, "mask_edge_attr_token_strategy", "zero"))
    if token_strategy not in {"zero", "mean", "learned"}:
        raise ValueError(f"Unsupported mask token strategy {token_strategy!r}")

    edge_attr = _forward_edge_attr(tr_data)[:, 1:].detach().cpu()
    amount = edge_attr[:, FIELD_COL_INDEX["amount"]].double()
    amount = torch.clamp(amount, min=0.0)
    log_amount = torch.log1p(amount)
    amount_mean = float(log_amount.mean().item())
    amount_std = max(float(log_amount.std(unbiased=False).item()), 1e-6)

    ts = edge_attr[:, FIELD_COL_INDEX["timestamp"]].double()
    ts_mean = float(ts.mean().item())
    ts_std = max(float(ts.std(unbiased=False).item()), 1e-6)

    currency_vals = edge_attr[:, FIELD_COL_INDEX["currency"]].long()
    payment_vals = edge_attr[:, FIELD_COL_INDEX["payment_format"]].long()
    currency_classes = torch.unique(currency_vals, sorted=True)
    payment_classes = torch.unique(payment_vals, sorted=True)

    mask_tokens = {
        "amount": 0.0 if token_strategy == "zero" else amount_mean,
        "currency": 0.0,
        "payment_format": 0.0,
        "timestamp": 0.0 if token_strategy == "zero" else ts_mean,
    }
    learned: Optional[nn.ParameterDict] = None
    if token_strategy == "learned":
        learned = nn.ParameterDict(
            {
                f: nn.Parameter(torch.zeros(1, dtype=torch.float32))
                for f in ("amount", "currency", "payment_format", "timestamp")
            }
        )

    loss_weights = parse_loss_weights(str(getattr(args, "masked_edge_loss_weights", "")), fields)
    return MaskedEdgePretrainSpec(
        fields=fields,
        token_strategy=token_strategy,
        mask_rate=float(getattr(args, "mask_edge_attr_rate", 0.15)),
        loss_weights=loss_weights,
        amount_loss=str(getattr(args, "masked_edge_amount_loss", "smooth_l1")),
        currency_classes=currency_classes,
        payment_classes=payment_classes,
        amount_log1p_mean=amount_mean,
        amount_log1p_std=amount_std,
        timestamp_mean=ts_mean,
        timestamp_std=ts_std,
        mask_tokens=mask_tokens,
        learned_mask_tokens=learned,
    )


class MaskedEdgeDecoder(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, spec: MaskedEdgePretrainSpec):
        super().__init__()
        self.spec = spec
        self.trunk = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.heads = nn.ModuleDict()
        if "amount" in spec.fields:
            self.heads["amount"] = nn.Linear(hidden_dim, 1)
        if "currency" in spec.fields:
            self.heads["currency"] = nn.Linear(hidden_dim, spec.n_currency)
        if "payment_format" in spec.fields:
            self.heads["payment_format"] = nn.Linear(hidden_dim, spec.n_payment)
        if "timestamp" in spec.fields:
            self.heads["timestamp"] = nn.Linear(hidden_dim, 1)

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.trunk(z)
        return {name: head(h) for name, head in self.heads.items()}


def setup_masked_edge_decoder(
    args,
    spec: MaskedEdgePretrainSpec,
    device: torch.device,
    *,
    embed_dim: int = 128,
) -> MaskedEdgeDecoder:
    hidden = int(getattr(args, "masked_edge_decoder_hidden_dim", 128))
    decoder = MaskedEdgeDecoder(embed_dim, hidden, spec).to(device)
    if spec.learned_mask_tokens is not None:
        spec.learned_mask_tokens.to(device)
    return decoder


def _class_index(values: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
    # Map raw categorical codes to contiguous class ids; unseen -> -1.
    out = torch.full(values.shape, -1, device=values.device, dtype=torch.long)
    for i, cls in enumerate(classes.tolist()):
        out[values.long() == int(cls)] = i
    return out


def _mask_token_value(spec: MaskedEdgePretrainSpec, field: str, device: torch.device) -> float:
    if spec.token_strategy == "learned" and spec.learned_mask_tokens is not None:
        return float(spec.learned_mask_tokens[field].item())
    return float(spec.mask_tokens[field])


def extract_field_targets(
    edge_attr_4: torch.Tensor,
    spec: MaskedEdgePretrainSpec,
) -> Dict[str, torch.Tensor]:
    """Targets for seed edges from uncorrupted 4-d edge attributes."""
    targets: Dict[str, torch.Tensor] = {}
    if "amount" in spec.fields:
        amt = torch.clamp(edge_attr_4[:, FIELD_COL_INDEX["amount"]], min=0.0)
        targets["amount"] = torch.log1p(amt)
    if "currency" in spec.fields:
        targets["currency"] = _class_index(
            edge_attr_4[:, FIELD_COL_INDEX["currency"]], spec.currency_classes.to(edge_attr_4.device)
        )
    if "payment_format" in spec.fields:
        targets["payment_format"] = _class_index(
            edge_attr_4[:, FIELD_COL_INDEX["payment_format"]],
            spec.payment_classes.to(edge_attr_4.device),
        )
    if "timestamp" in spec.fields:
        ts = edge_attr_4[:, FIELD_COL_INDEX["timestamp"]]
        targets["timestamp"] = (ts - spec.timestamp_mean) / spec.timestamp_std
    return targets


def sample_field_masks(
    num_seed: int,
    spec: MaskedEdgePretrainSpec,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> Dict[str, torch.Tensor]:
    masks: Dict[str, torch.Tensor] = {}
    for field in spec.fields:
        probs = torch.rand((num_seed,), device=device, generator=generator)
        masks[field] = probs < spec.mask_rate
    return masks


def apply_masked_inputs(
    edge_attr_4: torch.Tensor,
    row_mask: torch.Tensor,
    field_masks: Dict[str, torch.Tensor],
    spec: MaskedEdgePretrainSpec,
) -> torch.Tensor:
    """
    Hide selected fields on ``row_mask`` rows. ``field_masks`` are per seed edge.
    """
    out = edge_attr_4.clone()
    if not row_mask.any():
        return out
    seed_rows = row_mask.nonzero(as_tuple=False).view(-1)
    for field, m in field_masks.items():
        if not m.any():
            continue
        col = FIELD_COL_INDEX[field]
        token = _mask_token_value(spec, field, edge_attr_4.device)
        masked_seed_rows = seed_rows[m]
        out[masked_seed_rows, col] = token
    return out


def _apply_masks_preserve_id(
    edge_attr_with_id: torch.Tensor,
    seed_rows: torch.Tensor,
    field_masks: Dict[str, torch.Tensor],
    spec: MaskedEdgePretrainSpec,
) -> torch.Tensor:
    """Corrupt attribute columns while keeping synthetic id column 0."""
    out = edge_attr_with_id.clone()
    body = out[:, 1:]
    for field, m in field_masks.items():
        if not m.any():
            continue
        col = FIELD_COL_INDEX[field]
        token = _mask_token_value(spec, field, edge_attr_with_id.device)
        masked_seed_rows = seed_rows[m]
        body[masked_seed_rows, col] = token
    return out


def _apply_reverse_masks_by_txn(
    rev_attr_with_id: torch.Tensor,
    rev_seed_rows: torch.Tensor,
    rev_txn_ids: torch.Tensor,
    txn_to_mask_idx: Dict[int, int],
    field_masks: Dict[str, torch.Tensor],
    spec: MaskedEdgePretrainSpec,
) -> torch.Tensor:
    out = rev_attr_with_id.clone()
    for r in rev_seed_rows.tolist():
        txn = int(rev_txn_ids[r].item())
        idx = txn_to_mask_idx.get(txn)
        if idx is None:
            continue
        for field in spec.fields:
            if not bool(field_masks[field][idx].item()):
                continue
            col = FIELD_COL_INDEX[field] + 1
            token = _mask_token_value(spec, field, rev_attr_with_id.device)
            out[r, col] = token
    return out


def prepare_masked_edge_batch(
    batch: HeteroData | Data,
    *,
    spec: MaskedEdgePretrainSpec,
    seed_edge_ids: torch.Tensor,
    is_hetero: bool,
    generator: torch.Generator,
    loader_data: HeteroData | Data | None = None,
) -> Tuple[HeteroData | Data, MaskedEdgeBatchState]:
    if is_hetero:
        if loader_data is None or not isinstance(loader_data, HeteroData):
            raise TypeError("prepare_masked_edge_batch on HeteroData requires loader_data.")
        fwd = batch[FORWARD_EDGE_TYPE]
        rev = batch[REVERSE_EDGE_TYPE]
        fwd_attr = fwd.edge_attr
        seed_edge_ids = seed_edge_ids.to(fwd_attr.device)
        seed_mask_fwd = torch.isin(fwd_attr[:, 0].long(), seed_edge_ids)
        if not seed_mask_fwd.any():
            empty = {f: torch.zeros(0, dtype=torch.bool, device=fwd_attr.device) for f in spec.fields}
            return batch, MaskedEdgeBatchState(seed_mask_fwd, empty, {}, {})

        seed_rows = seed_mask_fwd.nonzero(as_tuple=False).view(-1)
        seed_attr = fwd_attr[seed_rows, 1:].clone()
        field_masks = sample_field_masks(
            int(seed_rows.numel()), spec, device=fwd_attr.device, generator=generator
        )
        targets = extract_field_targets(seed_attr, spec)

        fwd.edge_attr = _apply_masks_preserve_id(fwd_attr, seed_rows, field_masks, spec)

        n_fwd_global = int(loader_data[FORWARD_EDGE_TYPE].edge_attr.shape[0])
        rev_txn = rev.edge_attr[:, 0].long() - n_fwd_global
        rev_seed_mask = torch.isin(rev_txn, seed_edge_ids)
        rev_seed_rows = rev_seed_mask.nonzero(as_tuple=False).view(-1)
        seed_txn_ids = fwd_attr[seed_rows, 0].long()
        txn_to_mask_idx = {int(t): i for i, t in enumerate(seed_txn_ids.tolist())}
        rev.edge_attr = _apply_reverse_masks_by_txn(
            rev.edge_attr, rev_seed_rows, rev_txn, txn_to_mask_idx, field_masks, spec
        )

        stats = {
            f"mask_rate_{field}": float(m.float().mean().item()) if m.numel() else 0.0
            for field, m in field_masks.items()
        }
        return batch, MaskedEdgeBatchState(seed_mask_fwd, field_masks, targets, stats)

    seed_edge_ids = seed_edge_ids.to(batch.edge_attr.device)
    seed_mask = torch.isin(batch.edge_attr[:, 0].long(), seed_edge_ids)
    if not seed_mask.any():
        empty = {f: torch.zeros(0, dtype=torch.bool, device=batch.edge_attr.device) for f in spec.fields}
        return batch, MaskedEdgeBatchState(seed_mask, empty, {}, {})
    seed_rows = seed_mask.nonzero(as_tuple=False).view(-1)
    seed_attr = batch.edge_attr[seed_rows, 1:].clone()
    field_masks = sample_field_masks(
        int(seed_rows.numel()), spec, device=batch.edge_attr.device, generator=generator
    )
    targets = extract_field_targets(seed_attr, spec)
    batch.edge_attr = _apply_masks_preserve_id(batch.edge_attr, seed_rows, field_masks, spec)
    stats = {
        f"mask_rate_{field}": float(m.float().mean().item()) if m.numel() else 0.0
        for field, m in field_masks.items()
    }
    return batch, MaskedEdgeBatchState(seed_mask, field_masks, targets, stats)


def compute_masked_edge_loss(
    z_seed: torch.Tensor,
    state: MaskedEdgeBatchState,
    decoder: MaskedEdgeDecoder,
    spec: MaskedEdgePretrainSpec,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if z_seed.numel() == 0:
        zero = z_seed.new_zeros(())
        return zero, {"masked_edge/total": 0.0}

    preds = decoder(z_seed)
    total = z_seed.new_zeros(())
    logs: Dict[str, float] = {}
    weight_sum = 0.0

    for field in spec.fields:
        mask = state.field_masks.get(field)
        target = state.targets.get(field)
        if mask is None or target is None or not mask.any():
            logs[f"masked_edge/{field}"] = 0.0
            continue
        pred = preds[field][mask]
        tgt = target[mask]
        if field == "amount":
            if spec.amount_loss == "mse":
                loss = F.mse_loss(pred.view(-1), tgt.view(-1))
            else:
                loss = F.smooth_l1_loss(pred.view(-1), tgt.view(-1))
        elif field == "timestamp":
            loss = F.smooth_l1_loss(pred.view(-1), tgt.view(-1))
        else:
            valid = tgt >= 0
            if not valid.any():
                logs[f"masked_edge/{field}"] = 0.0
                continue
            loss = F.cross_entropy(pred[valid], tgt[valid])
        w = float(spec.loss_weights.get(field, 1.0))
        total = total + loss * w
        weight_sum += w
        logs[f"masked_edge/{field}"] = float(loss.detach().item())

    if weight_sum > 0:
        total = total / weight_sum
    logs["masked_edge/total"] = float(total.detach().item()) if total.numel() else 0.0
    return total, logs


def verify_masked_inputs_hidden(
    original_attr: torch.Tensor,
    corrupted_attr: torch.Tensor,
    seed_mask: torch.Tensor,
    field_masks: Dict[str, torch.Tensor],
    spec: MaskedEdgePretrainSpec,
) -> None:
    """Assert masked fields differ from targets on corrupted rows."""
    seed_rows = seed_mask.nonzero(as_tuple=False).view(-1)
    for field, m in field_masks.items():
        if not m.any():
            continue
        col = FIELD_COL_INDEX[field]
        rows = seed_rows[m]
        if field in {"amount", "timestamp"}:
            if torch.allclose(corrupted_attr[rows, col], original_attr[rows, col]):
                raise AssertionError(f"Masked field {field} was not hidden from encoder input.")
        else:
            if torch.equal(corrupted_attr[rows, col].long(), original_attr[rows, col].long()):
                raise AssertionError(f"Masked field {field} was not hidden from encoder input.")
