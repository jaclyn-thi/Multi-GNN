import torch.nn as nn
from torch_geometric.nn import GINEConv, BatchNorm, Linear, GATConv, PNAConv, RGCNConv
import torch.nn.functional as F
import torch
import logging
from torch.utils.checkpoint import checkpoint

SUPERVISED_HEADS = ("embedding", "legacy")


def _validate_supervised_head(supervised_head: str) -> str:
    head = str(supervised_head or "embedding").lower()
    if head not in SUPERVISED_HEADS:
        raise ValueError(
            f"Unsupported supervised_head {supervised_head!r}; use one of {SUPERVISED_HEADS}."
        )
    return head


def _legacy_edge_classifier(n_hidden: int, n_classes: int, final_dropout: float) -> nn.Sequential:
    """Fork-point IBM Multi-GNN (Egressy et al.) supervised head: 3*n_hidden -> 50 -> 25 -> n_classes.

    Reproduces ``self.mlp`` from commit ``fc751e8:models.py`` (identical across GINe/GATe/PNA/RGCN):
    ``Sequential(Linear(3h,50), ReLU, Dropout(fd), Linear(50,25), ReLU, Dropout(fd), Linear(25, n))``.
    Applied on the raw edge representation, it emits two-class logits directly (no 128-d bottleneck).
    """
    return nn.Sequential(
        nn.Linear(n_hidden * 3, 50),
        nn.ReLU(),
        nn.Dropout(final_dropout),
        nn.Linear(50, 25),
        nn.ReLU(),
        nn.Dropout(final_dropout),
        nn.Linear(25, n_classes),
    )


class GINe(torch.nn.Module):
    def __init__(self, num_features, num_gnn_layers, n_classes=2,
                n_hidden=100, edge_updates=False, residual=True,
                edge_dim=None, dropout=0.0, final_dropout=0.5,
                embedding_dim=128, use_gradient_checkpointing=False,
                supervised_head="embedding", bypass_embedding_head=False):

        super().__init__()
        self.n_hidden = n_hidden
        self.num_gnn_layers = num_gnn_layers
        self.edge_updates = edge_updates
        self.final_dropout = final_dropout
        self.embedding_dim = embedding_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.supervised_head = _validate_supervised_head(supervised_head)

        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)

        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(self.num_gnn_layers):
            conv = GINEConv(nn.Sequential(
                nn.Linear(self.n_hidden, self.n_hidden),
                nn.ReLU(),
                nn.Linear(self.n_hidden, self.n_hidden)
            ), edge_dim=self.n_hidden)

            if self.edge_updates:
                self.emlps.append(nn.Sequential(
                    nn.Linear(3 * self.n_hidden, self.n_hidden),
                    nn.ReLU(),
                    nn.Linear(self.n_hidden, self.n_hidden),
                ))

            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(n_hidden))

        # DIRECT_H: bypass embedding_head so forward returns R198 (pre_embedding_3h).
        self.bypass_embedding_head = bool(bypass_embedding_head)
        if self.supervised_head == "legacy":
            # Fork-point (IBM Multi-GNN, Egressy et al.) head: logits straight from the
            # 3*n_hidden edge representation, no embedding bottleneck. forward() returns
            # that representation and self.classifier maps it to two-class logits.
            self.embedding_head = None
            self.classifier = _legacy_edge_classifier(n_hidden, n_classes, self.final_dropout)
        elif self.bypass_embedding_head:
            # Contrastive DIRECT_H: return R198; keep a classifier scaffold unused in SSL.
            self.embedding_head = None
            self.classifier = _legacy_edge_classifier(n_hidden, n_classes, self.final_dropout)
            self.embedding_dim = int(n_hidden * 3)
        else:
            # Current project head: embedding layer compresses GNN output into a clean
            # representation space; classifier maps the embedding to a prediction.
            self.embedding_head = nn.Linear(n_hidden * 3, embedding_dim)
            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim, 50),
                nn.ReLU(),
                nn.Dropout(self.final_dropout),
                nn.Linear(50, n_classes)
            )

    def _input_embedding_forward(self, x: torch.Tensor, edge_attr: torch.Tensor) -> tuple:
        return self.node_emb(x), self.edge_emb(edge_attr)

    def _gine_layer_forward(
        self,
        layer_idx_t: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> tuple:
        i = int(layer_idx_t.item())
        src, dst = edge_index
        x = (x + F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr)))) / 2
        if self.edge_updates:
            edge_attr = edge_attr + self.emlps[i](
                torch.cat([x[src], x[dst], edge_attr], dim=-1)
            ) / 2
        return x, edge_attr

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index

        if self.use_gradient_checkpointing and self.training:
            x, edge_attr = checkpoint(
                self._input_embedding_forward,
                x,
                edge_attr,
                use_reentrant=False,
            )
        else:
            x = self.node_emb(x)
            edge_attr = self.edge_emb(edge_attr)

        for i in range(self.num_gnn_layers):
            if self.use_gradient_checkpointing and self.training:
                idx = torch.tensor(i, device=x.device, dtype=torch.int64)
                x, edge_attr = checkpoint(
                    self._gine_layer_forward,
                    idx,
                    x,
                    edge_index,
                    edge_attr,
                    use_reentrant=False,
                )
            else:
                x = (x + F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr)))) / 2
                if self.edge_updates:
                    edge_attr = edge_attr + self.emlps[i](
                        torch.cat([x[src], x[dst], edge_attr], dim=-1)
                    ) / 2

        # Per-edge readout (+ embedding head in embedding mode): large activations;
        # checkpoint when enabled. DIRECT_H / legacy: return R198 (3*n_hidden).
        use_r198_readout = (
            self.supervised_head == "legacy" or self.bypass_embedding_head
        )
        tail_fn = (
            self._legacy_readout_forward
            if use_r198_readout
            else self._embedding_tail_forward
        )
        if self.use_gradient_checkpointing and self.training:
            z = checkpoint(
                tail_fn,
                x,
                edge_index,
                edge_attr,
                use_reentrant=False,
            )
        else:
            z = tail_fn(x, edge_index, edge_attr)

        # Edge representation only (FX-safe for torch_geometric.to_hetero). Supervised:
        # model.classifier(z). embedding mode returns embedding_dim; legacy returns 3*n_hidden.
        return z

    def _legacy_readout_forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Upstream fork-point edge representation: cat(relu(node pair), edge_attr) -> 3*n_hidden."""
        x = x[edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
        return torch.cat((x, edge_attr), dim=1)

    def _embedding_tail_forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        x = x[edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
        x = torch.cat((x, edge_attr), dim=1)
        return self.embedding_head(x)


class GATe(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_gnn_layers,
        n_classes=2,
        n_hidden=100,
        n_heads=4,
        edge_updates=False,
        edge_dim=None,
        dropout=0.0,
        final_dropout=0.5,
        embedding_dim=128,
        supervised_head="embedding",
    ):
        super().__init__()

        # --- GAT-specific ---
        tmp_out = n_hidden // n_heads
        n_hidden = tmp_out * n_heads  # ensure divisible

        self.n_hidden = n_hidden
        self.n_heads = n_heads
        self.num_gnn_layers = num_gnn_layers
        self.edge_updates = edge_updates
        self.dropout = dropout
        self.final_dropout = final_dropout
        self.embedding_dim = embedding_dim
        self.supervised_head = _validate_supervised_head(supervised_head)

        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)

        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(self.num_gnn_layers):
            conv = GATConv(
                self.n_hidden,
                tmp_out,
                self.n_heads,
                concat=True,
                dropout=self.dropout,
                add_self_loops=True,
                edge_dim=self.n_hidden,
            )

            if self.edge_updates:
                self.emlps.append(
                    nn.Sequential(
                        nn.Linear(3 * self.n_hidden, self.n_hidden),
                        nn.ReLU(),
                        nn.Linear(self.n_hidden, self.n_hidden),
                    )
                )

            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(n_hidden))

        # supervised_head="legacy" is restored-but-unvalidated for GATe (uniform fork-point
        # head; only GINe is numerically validated against the fork point).
        if self.supervised_head == "legacy":
            self.embedding_head = None
            self.classifier = _legacy_edge_classifier(n_hidden, n_classes, self.final_dropout)
        else:
            self.embedding_head = nn.Linear(n_hidden * 3, embedding_dim)
            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim, 50),
                nn.ReLU(),
                nn.Dropout(self.final_dropout),
                nn.Linear(50, n_classes),
            )

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        for i in range(self.num_gnn_layers):
            x = (x + F.relu(self.batch_norms[i](
                self.convs[i](x, edge_index, edge_attr)
            ))) / 2

            if self.edge_updates:
                edge_attr = edge_attr + self.emlps[i](
                    torch.cat([x[src], x[dst], edge_attr], dim=-1)
                ) / 2

        # --- edge-level features ---
        logging.debug(f"x.shape = {x.shape}, x[edge_index.T].shape = {x[edge_index.T].shape}")
        x = x[edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
        logging.debug(f"x.shape after reshape = {x.shape}")

        x = torch.cat((x, edge_attr), dim=1)
        logging.debug(f"x.shape after concat = {x.shape}")

        if self.supervised_head == "legacy":
            return x
        z = self.embedding_head(x)
        return z


class PNA(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_gnn_layers,
        n_classes=2,
        n_hidden=100,
        edge_updates=True,
        edge_dim=None,
        dropout=0.0,
        final_dropout=0.5,
        deg=None,
        embedding_dim=128,
        supervised_head="embedding",
    ):
        super().__init__()

        n_hidden = int((n_hidden // 5) * 5)

        self.n_hidden = n_hidden
        self.num_gnn_layers = num_gnn_layers
        self.edge_updates = edge_updates
        self.final_dropout = final_dropout
        self.embedding_dim = embedding_dim
        self.supervised_head = _validate_supervised_head(supervised_head)

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)

        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(self.num_gnn_layers):
            conv = PNAConv(
                in_channels=n_hidden,
                out_channels=n_hidden,
                aggregators=aggregators,
                scalers=scalers,
                deg=deg,
                edge_dim=n_hidden,
                towers=5,
                pre_layers=1,
                post_layers=1,
                divide_input=False,
            )

            if self.edge_updates:
                self.emlps.append(
                    nn.Sequential(
                        nn.Linear(3 * self.n_hidden, self.n_hidden),
                        nn.ReLU(),
                        nn.Linear(self.n_hidden, self.n_hidden),
                    )
                )

            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(n_hidden))

        # supervised_head="legacy" is restored-but-unvalidated for PNA (uniform fork-point
        # head; only GINe is numerically validated against the fork point).
        if self.supervised_head == "legacy":
            self.embedding_head = None
            self.classifier = _legacy_edge_classifier(n_hidden, n_classes, self.final_dropout)
        else:
            self.embedding_head = nn.Linear(n_hidden * 3, embedding_dim)
            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim, 50),
                nn.ReLU(),
                nn.Dropout(self.final_dropout),
                nn.Linear(50, n_classes),
            )

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        for i in range(self.num_gnn_layers):
            x = (x + F.relu(
                self.batch_norms[i](self.convs[i](x, edge_index, edge_attr))
            )) / 2

            if self.edge_updates:
                edge_attr = edge_attr + self.emlps[i](
                    torch.cat([x[src], x[dst], edge_attr], dim=-1)
                ) / 2

        # --- edge-level features ---
        logging.debug(f"x.shape = {x.shape}, x[edge_index.T].shape = {x[edge_index.T].shape}")

        x = x[edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
        logging.debug(f"x.shape after reshape = {x.shape}")

        x = torch.cat((x, edge_attr), dim=1)
        logging.debug(f"x.shape after concat = {x.shape}")

        if self.supervised_head == "legacy":
            return x
        z = self.embedding_head(x)
        return z


class RGCN(nn.Module):
    def __init__(
        self,
        num_features,
        edge_dim,
        num_relations,
        num_gnn_layers,
        n_classes=2,
        n_hidden=100,
        edge_update=False,
        residual=True,
        dropout=0.0,
        final_dropout=0.5,
        n_bases=-1,
        embedding_dim=128,
        supervised_head="embedding",
    ):
        super(RGCN, self).__init__()

        self.num_features = num_features
        self.num_gnn_layers = num_gnn_layers
        self.n_hidden = n_hidden
        self.residual = residual
        self.dropout = dropout
        self.final_dropout = final_dropout
        self.n_classes = n_classes
        self.edge_update = edge_update
        self.num_relations = num_relations
        self.n_bases = n_bases
        self.embedding_dim = embedding_dim
        self.supervised_head = _validate_supervised_head(supervised_head)

        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.emlps = nn.ModuleList()

        for _ in range(self.num_gnn_layers):
            self.convs.append(
                RGCNConv(
                    self.n_hidden,
                    self.n_hidden,
                    num_relations,
                    num_bases=self.n_bases,
                )
            )

            self.bns.append(nn.BatchNorm1d(self.n_hidden))

            if self.edge_update:
                self.emlps.append(
                    nn.Sequential(
                        nn.Linear(3 * self.n_hidden, self.n_hidden),
                        nn.ReLU(),
                        nn.Linear(self.n_hidden, self.n_hidden),
                    )
                )

        # supervised_head="legacy" is restored-but-unvalidated for RGCN (uniform fork-point
        # head; only GINe is numerically validated against the fork point).
        if self.supervised_head == "legacy":
            self.embedding_head = None
            self.classifier = _legacy_edge_classifier(n_hidden, n_classes, self.final_dropout)
        else:
            self.embedding_head = nn.Linear(n_hidden * 3, embedding_dim)
            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim, 50),
                nn.ReLU(),
                nn.Dropout(self.final_dropout),
                nn.Linear(50, n_classes),
            )

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            elif isinstance(m, RGCNConv):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        edge_type = edge_attr[:, -1].long()
        src, dst = edge_index

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        for i in range(self.num_gnn_layers):
            x = (x + F.relu(
                self.bns[i](self.convs[i](x, edge_index, edge_type))
            )) / 2

            if self.edge_update:
                edge_attr = (
                    edge_attr
                    + F.relu(
                        self.emlps[i](
                            torch.cat([x[src], x[dst], edge_attr], dim=-1)
                        )
                    )
                ) / 2

        # edge-level features
        x = x[edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
        x = torch.cat((x, edge_attr), dim=1)

        if self.supervised_head == "legacy":
            return x
        z = self.embedding_head(x)
        return z
