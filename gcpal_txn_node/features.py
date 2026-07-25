"""Node features and random / KNN views for txn-as-node graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class FeaturePreprocessor:
    """Fit on train rows only; transform any row index set."""

    amount_col: str
    scaler: StandardScaler
    currency_encoder: OneHotEncoder
    format_encoder: OneHotEncoder
    feature_names: List[str]

    def transform(self, df_rows) -> np.ndarray:
        ts = df_rows["Timestamp"].astype(float).to_numpy().reshape(-1, 1)
        amt = np.log1p(np.maximum(df_rows[self.amount_col].astype(float).to_numpy(), 0.0)).reshape(
            -1, 1
        )
        cont = self.scaler.transform(np.hstack([ts, amt])).astype(np.float32)
        cur = self.currency_encoder.transform(
            df_rows[["Received Currency"]].astype(str).fillna("__missing__")
        )
        fmt = self.format_encoder.transform(
            df_rows[["Payment Format"]].astype(str).fillna("__missing__")
        )
        if hasattr(cur, "toarray"):
            cur = cur.toarray()
        if hasattr(fmt, "toarray"):
            fmt = fmt.toarray()
        return np.hstack([cont, cur.astype(np.float32), fmt.astype(np.float32)]).astype(np.float32)


def fit_feature_preprocessor(df_train, amount_col: str = "Amount Received") -> FeaturePreprocessor:
    ts = df_train["Timestamp"].astype(float).to_numpy().reshape(-1, 1)
    amt = np.log1p(np.maximum(df_train[amount_col].astype(float).to_numpy(), 0.0)).reshape(-1, 1)
    scaler = StandardScaler().fit(np.hstack([ts, amt]))
    cur = df_train[["Received Currency"]].astype(str).fillna("__missing__")
    fmt = df_train[["Payment Format"]].astype(str).fillna("__missing__")
    # handle_unknown for val/test in temporal mode
    try:
        currency_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        format_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        currency_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        format_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    currency_encoder.fit(cur)
    format_encoder.fit(fmt)
    names = (
        ["Timestamp_std", f"log1p_{amount_col}_std"]
        + [f"cur_{c}" for c in currency_encoder.categories_[0].tolist()]
        + [f"fmt_{c}" for c in format_encoder.categories_[0].tolist()]
    )
    return FeaturePreprocessor(
        amount_col=amount_col,
        scaler=scaler,
        currency_encoder=currency_encoder,
        format_encoder=format_encoder,
        feature_names=names,
    )


def drop_edges(
    edge_index: torch.Tensor,
    drop_rate: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Independently drop adjacency edges; never removes nodes."""
    e = int(edge_index.shape[1])
    if e == 0 or drop_rate <= 0:
        return edge_index
    if drop_rate >= 1.0:
        return edge_index[:, :0]
    keep = torch.rand(e, generator=generator, device=edge_index.device) >= float(drop_rate)
    if not bool(keep.any()):
        # Keep at least one edge if possible to avoid empty graphs collapsing MP entirely.
        keep = torch.zeros(e, dtype=torch.bool, device=edge_index.device)
        keep[0] = True
    return edge_index[:, keep]


def mask_feature_rows(
    x: torch.Tensor,
    drop_rate: float,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mask complete node-feature rows (paper-style), not individual cells.

    Returns (x_masked, row_mask) where row_mask[i]=True means row i was zeroed.
    """
    n = int(x.shape[0])
    if n == 0 or drop_rate <= 0:
        return x, torch.zeros(n, dtype=torch.bool, device=x.device)
    row_mask = torch.rand(n, generator=generator, device=x.device) < float(drop_rate)
    out = x.clone()
    out[row_mask] = 0.0
    return out, row_mask


@dataclass
class GraphView:
    x: torch.Tensor
    edge_index: torch.Tensor
    node_ids: torch.Tensor  # global / original transaction ids aligned with rows
    name: str


def make_random_structural_view(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    node_ids: torch.Tensor,
    *,
    edge_drop: float,
    feature_drop: float,
    generator: Optional[torch.Generator] = None,
    name: str = "random",
) -> GraphView:
    ei = drop_edges(edge_index, edge_drop, generator=generator)
    xm, _ = mask_feature_rows(x, feature_drop, generator=generator)
    return GraphView(x=xm, edge_index=ei, node_ids=node_ids.clone(), name=name)


def make_knn_view(
    x: torch.Tensor,
    knn_edge_index: torch.Tensor,
    node_ids: torch.Tensor,
    *,
    feature_drop: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> GraphView:
    """KNN message-passing view: sparse KNN edges; optional feature-row mask."""
    xm, _ = mask_feature_rows(x, feature_drop, generator=generator)
    return GraphView(x=xm, edge_index=knn_edge_index, node_ids=node_ids.clone(), name="knn")
