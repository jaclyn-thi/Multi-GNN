#!/usr/bin/env python3
"""CPU smoke test for temporal-flow aux setup + one backward step on real Small-HI cache."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from morphology.temporal_flow_aux import setup_temporal_flow_aux, temporal_flow_aux_loss


def main() -> int:
    cache = ROOT / "results/cache/temporal_flow_causal/Small-HI"
    if not (cache / "features.npy").is_file():
        print(f"SKIP: missing cache {cache}")
        return 0
    device = torch.device("cpu")
    for mode, kwargs in (
        ("regression", {"aux_temporal_flow_loss": "huber"}),
        ("bins", {"aux_temporal_flow_bins": 5}),
    ):
        args = argparse.Namespace(
            aux_temporal_flow=mode,
            aux_temporal_flow_weight=0.1,
            aux_temporal_flow_loss=kwargs.get("aux_temporal_flow_loss", "huber"),
            aux_temporal_flow_bins=int(kwargs.get("aux_temporal_flow_bins", 5)),
            aux_temporal_flow_hidden=64,
            aux_temporal_flow_cache=str(cache),
            unique_name=f"smoke_tf_aux_{mode}",
        )
        head, cfg, ctx = setup_temporal_flow_aux(
            args, device, data_name="Small-HI", embedding_dim=128
        )
        assert head is not None and cfg is not None and ctx is not None
        assert cfg.uses_labels is False
        z = torch.randn(32, 128, requires_grad=True)
        ids = torch.arange(32, dtype=torch.long)
        loss, _ = temporal_flow_aux_loss(z, ids, head, cfg, ctx)
        assert torch.isfinite(loss), loss
        loss.backward()
        print(f"OK mode={mode} loss={float(loss.detach()):.6f} meta={cfg.metadata_path}")
    print("SMOKE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
