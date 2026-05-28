"""
Graph augmentations for contrastive learning on edge-level embeddings.

Supports independent random edge dropping per view (different edge counts /
ordering). Identity contrastive pairs are matched via ``edge_id``, not row index.
"""

import torch
from torch_geometric.data import Data, HeteroData

from train_util import FORWARD_EDGE_TYPE, REVERSE_EDGE_TYPE


def _edge_keep_mask(num_edges: int, drop_rate: float, device: torch.device) -> torch.Tensor:
    """Bernoulli keep mask of shape (E,); at least one edge kept when E > 0."""
    if num_edges == 0:
        return torch.ones(0, dtype=torch.bool, device=device)

    if not (0.0 <= drop_rate <= 1.0):
        raise ValueError(f"drop_rate must be in [0, 1], got {drop_rate}")

    keep = torch.rand(num_edges, device=device) > drop_rate
    if not keep.any():
        keep[torch.randint(0, num_edges, (1,), device=device)] = True
    return keep


def random_edge_drop(edge_index, edge_attr, drop_rate):
    """
    Randomly drops edges from a directed multigraph.

    Args:
        edge_index: (2, E)
        edge_attr: (E, D)
        drop_rate: float in [0,1]

    Returns:
        new_edge_index, new_edge_attr
    """
    E = edge_index.size(1)
    if E == 0:
        return edge_index, edge_attr

    keep = _edge_keep_mask(E, drop_rate, edge_index.device)
    return edge_index[:, keep], edge_attr[keep]


def _slice_edge_aligned_fields(data: Data, keep: torch.Tensor) -> None:
    """Apply ``keep`` to edge-aligned tensors on a Data object (mutates ``data``)."""
    E = keep.numel()
    data.edge_index = data.edge_index[:, keep].clone()
    if getattr(data, "edge_attr", None) is not None:
        data.edge_attr = data.edge_attr[keep].clone()

    eid = getattr(data, "edge_id", None)
    if eid is not None:
        if eid.shape[0] != E:
            raise ValueError("edge_id length must match edge_index before masking.")
        data.edge_id = eid[keep].clone()

    y = getattr(data, "y", None)
    if y is not None and y.dim() > 0 and y.shape[0] == E:
        data.y = y[keep].clone()

    ts = getattr(data, "timestamps", None)
    if ts is not None and ts.dim() > 0 and ts.shape[0] == E:
        data.timestamps = ts[keep].clone()


def _slice_hetero_edge_store(store, keep: torch.Tensor) -> None:
    """Apply ``keep`` to one heterogeneous edge store (mutates ``store``)."""
    _slice_edge_aligned_fields(store, keep)


def _random_edge_drop_view(data: Data, drop_rate: float) -> Data:
    """
    Independent random edge subset; does not mutate the input ``data``.

    Requires ``edge_id`` when ``drop_rate > 0`` so contrastive pairing stays defined.
    """
    out = data.clone()
    E = out.edge_index.size(1)
    if drop_rate <= 0.0:
        return out

    if getattr(out, "edge_id", None) is None:
        raise ValueError(
            "edge_id is required on the batch when edge_drop_rate > 0; "
            "use train_util.attach_edge_id_from_batch before augmentations."
        )

    keep = _edge_keep_mask(E, drop_rate, out.edge_index.device)
    _slice_edge_aligned_fields(out, keep)
    return out


def _hetero_random_edge_drop_view(
    batch: HeteroData,
    drop_rate: float,
    forward_et=FORWARD_EDGE_TYPE,
    reverse_et=REVERSE_EDGE_TYPE,
) -> HeteroData:
    """
    Independent random edge drop on forward transactions; reverse edges with the same
    transaction ``edge_id`` are dropped together (synchronized augmentation).
    """
    out = batch.clone()
    fwd = out[forward_et]
    E = fwd.edge_index.size(1)
    if drop_rate <= 0.0:
        return out

    if getattr(fwd, "edge_id", None) is None:
        raise ValueError(
            "edge_id is required on forward edges when edge_drop_rate > 0; "
            "use train_util.attach_edge_id_from_batch before augmentations."
        )

    keep_fwd = _edge_keep_mask(E, drop_rate, fwd.edge_index.device)
    _slice_hetero_edge_store(fwd, keep_fwd)
    kept_txn = fwd.edge_id

    rev = out[reverse_et]
    if getattr(rev, "edge_id", None) is None:
        raise ValueError("edge_id is required on reverse edges for synchronized hetero augmentation.")
    rev_keep = torch.isin(rev.edge_id, kept_txn)
    _slice_hetero_edge_store(rev, rev_keep)
    return out


def mask_edge_attr(
    edge_attr,
    mask_rate=0.1,
    mask_value=0.0,
    mask_cols=None,
    exclude_last_column=False,
):
    """
    Randomly mask entries in edge_attr. Shape is unchanged; input is not
    mutated.

    Args:
        edge_attr: (E, D) tensor.
        mask_rate: fraction in [0, 1] of eligible entries to mask (Bernoulli).
        mask_value: value written where masked.
        mask_cols: If None, all columns are eligible (subject to
            exclude_last_column). If an iterable of column indices, only those
            columns receive Bernoulli masks.
        exclude_last_column: If True, never mask the last column (e.g. RGCN
            relation / edge type in the final feature dim).

    Returns:
        New tensor (E, D) with same shape and dtype as edge_attr.
    """
    if not (0.0 <= mask_rate <= 1.0):
        raise ValueError(f"mask_rate must be in [0, 1], got {mask_rate}")

    out = edge_attr.clone()
    if out.numel() == 0:
        return out

    E, D = out.shape
    device = edge_attr.device
    eligible = torch.zeros((E, D), dtype=torch.bool, device=device)

    if mask_cols is None:
        eligible.fill_(True)
    else:
        cols = torch.as_tensor(list(mask_cols), device=device, dtype=torch.long)
        if cols.numel() > 0:
            eligible[:, cols] = True

    if exclude_last_column and D > 0:
        eligible[:, -1] = False

    if not eligible.any():
        return out

    rand = torch.rand((E, D), device=device)
    apply_mask = (rand < mask_rate) & eligible
    out[apply_mask] = mask_value
    return out


def generate_views(
    data,
    edge_attr_mask_rate=0.1,
    edge_drop_rate=0.1,
    mask_value=0.0,
    mask_cols=None,
    exclude_last_column=False,
):
    """
    Two augmented views for edge-level contrastive learning.

    - If ``edge_drop_rate > 0``: independent random edge dropping per view
      (same ``keep_mask`` applied to ``edge_index``, ``edge_attr``, ``edge_id``,
      ``y``, and ``timestamps`` when their first dim matches the edge count).
    - If ``edge_attr_mask_rate > 0``: independent edge-attribute masking on the
      surviving edges of each view.

    Does not mutate ``data`` (including ``data.edge_attr`` / ``data.edge_id``).

    For ``HeteroData``, edge drops are synchronized across forward and reverse edge
    types by transaction ``edge_id``; attribute masking is applied independently per
    view on both edge types.
    """
    if isinstance(data, HeteroData):
        view1 = _hetero_random_edge_drop_view(data, edge_drop_rate)
        view2 = _hetero_random_edge_drop_view(data, edge_drop_rate)
        if edge_attr_mask_rate > 0:
            for et in (FORWARD_EDGE_TYPE, REVERSE_EDGE_TYPE):
                store1, store2 = view1[et], view2[et]
                if store1.edge_attr is not None:
                    store1.edge_attr = mask_edge_attr(
                        store1.edge_attr,
                        mask_rate=edge_attr_mask_rate,
                        mask_value=mask_value,
                        mask_cols=mask_cols,
                        exclude_last_column=exclude_last_column,
                    )
                    store2.edge_attr = mask_edge_attr(
                        store2.edge_attr,
                        mask_rate=edge_attr_mask_rate,
                        mask_value=mask_value,
                        mask_cols=mask_cols,
                        exclude_last_column=exclude_last_column,
                    )
        return view1, view2

    view1 = _random_edge_drop_view(data, edge_drop_rate)
    view2 = _random_edge_drop_view(data, edge_drop_rate)

    if edge_attr_mask_rate > 0 and view1.edge_attr is not None:
        view1.edge_attr = mask_edge_attr(
            view1.edge_attr,
            mask_rate=edge_attr_mask_rate,
            mask_value=mask_value,
            mask_cols=mask_cols,
            exclude_last_column=exclude_last_column,
        )
        view2.edge_attr = mask_edge_attr(
            view2.edge_attr,
            mask_rate=edge_attr_mask_rate,
            mask_value=mask_value,
            mask_cols=mask_cols,
            exclude_last_column=exclude_last_column,
        )
    return view1, view2


def assert_shape_preserving_views(data, view1, view2, edge_attr_before, mask_rate):
    """
    Lightweight checks for the edge_drop_rate=0 path (same topology as ``data``).
    """
    assert type(view1) is type(data)
    assert type(view2) is type(data)
    assert view1.edge_index.shape == data.edge_index.shape
    assert view2.edge_index.shape == data.edge_index.shape
    assert torch.equal(view1.edge_index, data.edge_index)
    assert torch.equal(view2.edge_index, data.edge_index)
    assert view1.edge_attr.shape == data.edge_attr.shape
    assert view2.edge_attr.shape == data.edge_attr.shape
    assert torch.equal(data.edge_attr, edge_attr_before)
    if mask_rate > 0 and data.edge_attr.numel() > 0:
        assert not torch.equal(view1.edge_attr, view2.edge_attr)


if __name__ == "__main__":
    x = torch.ones(6, 1)
    ei = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=torch.long)
    ea = torch.randn(6, 4)
    d = Data(x=x, edge_index=ei, edge_attr=ea.clone())
    d.edge_id = torch.arange(6, dtype=torch.long)
    d.y = torch.zeros(6, dtype=torch.long)
    d.timestamps = torch.randn(6)
    ea_snap = d.edge_attr.clone()
    v1, v2 = generate_views(
        d,
        edge_drop_rate=0.0,
        edge_attr_mask_rate=0.5,
        exclude_last_column=True,
    )
    assert_shape_preserving_views(d, v1, v2, ea_snap, mask_rate=0.5)
    print("graph_augmentations: assert_shape_preserving_views OK")

    v1d, v2d = generate_views(d, edge_drop_rate=0.4, edge_attr_mask_rate=0.0)
    assert v1d.edge_id.shape[0] == v1d.edge_index.size(1)
    assert v2d.edge_id.shape[0] == v2d.edge_index.size(1)
    assert torch.equal(d.edge_attr, ea_snap)
    print("graph_augmentations: independent edge drop OK")

    from torch_geometric.data import HeteroData as HD

    h = HD()
    h["node"].x = x
    h["node", "to", "node"].edge_index = ei
    h["node", "rev_to", "node"].edge_index = ei.flip(0)
    h["node", "to", "node"].edge_attr = ea.clone()
    h["node", "rev_to", "node"].edge_attr = ea.clone()
    h["node", "to", "node"].edge_id = torch.arange(6, dtype=torch.long)
    h["node", "rev_to", "node"].edge_id = torch.arange(6, dtype=torch.long)
    hv1, hv2 = generate_views(h, edge_drop_rate=0.5, edge_attr_mask_rate=0.0)
    assert hv1["node", "to", "node"].edge_id.shape[0] == hv1["node", "to", "node"].edge_index.size(1)
    assert torch.equal(
        torch.sort(hv1["node", "to", "node"].edge_id).values,
        torch.sort(hv1["node", "rev_to", "node"].edge_id).values,
    )
    print("graph_augmentations: hetero synchronized edge drop OK")
