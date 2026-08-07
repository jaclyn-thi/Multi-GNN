#!/usr/bin/env python3
"""Phase-4C frozen R198 validation extract / probe / per-arm finalize.

No test data. Direct frozen encoder R198 only. Projection head bypassed at extract.
Requires SLURM_JOB_ID for real-data extract (get_data).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import torch
from torch_geometric.nn import to_hetero

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_loading import get_data  # noqa: E402
from mixed_ssl_phase2.bn import apply_bn_, clone_bn_bundle, collect_bn_bundle  # noqa: E402
from phase4b_frozen_eval.probe import fit_r198_probe  # noqa: E402
from phase4c_four_domain import ARMS, resolved_recipe  # noqa: E402
from phase4c_four_domain_frozen_eval import (  # noqa: E402
    CONTRACT_ID,
    EDGE_DIM,
    INIT_SHA256,
    PROBE,
    R198_DIM,
    TARGETS,
    all_extract_cells,
    bn_bundle_domain,
    cell_name,
    checkpoint_relpath,
    emb_cell_dir,
    extract_steps,
    result_root,
)
from train_util import (  # noqa: E402
    AddEgoIds,
    FORWARD_EDGE_TYPE,
    add_arange_ids,
    expected_seed_edge_ids,
    extract_param,
    extract_seed_embeddings_hetero,
    get_loaders,
    log_seed_coverage,
    save_embedding_split_npz,
)
from training import get_model  # noqa: E402
from util import create_parser, logger_setup, set_seed  # noqa: E402


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require_slurm() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Refuse: Phase-4C frozen eval real-data path requires SLURM_JOB_ID")


def make_extract_args(target: str, cell: str, emb_root: Path) -> argparse.Namespace:
    argv = [
        "--data", target,
        "--model", "gin",
        "--objective", "contrastive",
        "--unique_name", f"phase4c_frozen_{cell}",
        "--seed", "2",
        "--batch_size", "8192",
        "--num_neighs", "100", "100",
        "--loader_num_workers", "0",
        "--reverse_mp", "--ego", "--ports", "--emlps", "--tds",
        "--correct_reverse_edge_features",
        "--feature_contract", CONTRACT_ID,
        "--train_fit_edge_znorm",
        "--skip_test_eval",
        "--direct_r198_infonce",
    ]
    ns = create_parser().parse_args(argv)
    ns.preserve_seed_edges = False
    ns.skip_test_eval = True
    ns.embedding_dim = R198_DIM
    ns.representation_source = "pre_embedding_3h"
    ns.extract_splits = "train,val"
    ns.embeddings_dir = str(emb_root.parent)
    ns.embeddings_subdir = emb_root.name
    return ns


def run_extract(arm: str, step: int, target: str) -> Dict[str, Any]:
    require_slurm()
    if arm not in ARMS or target not in TARGETS or step not in extract_steps(arm):
        raise RuntimeError(f"bad extract cell {arm=} {step=} {target=}")
    recipe = resolved_recipe(arm)
    accept = ROOT / recipe["result_root"] / "train_acceptance.json"
    if not accept.is_file() or not json.loads(accept.read_text()).get("ok"):
        raise RuntimeError(f"missing/failed train_acceptance for {arm}")
    cell = cell_name(arm, step, target)
    out_dir = ROOT / emb_cell_dir(arm, step, target)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "test.npz").is_file():
        raise RuntimeError(f"test.npz present: {out_dir}")
    ckpt_p = ROOT / checkpoint_relpath(arm, step)
    if not ckpt_p.is_file():
        raise RuntimeError(f"missing checkpoint {ckpt_p}")
    blob = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    if blob.get("feature_contract_id") != CONTRACT_ID:
        raise RuntimeError("contract mismatch in checkpoint")
    if blob.get("init_sha256") != INIT_SHA256:
        raise RuntimeError("init sha mismatch in checkpoint")
    if blob.get("test_evaluated") is not False:
        raise RuntimeError("checkpoint claims test evaluation")
    if int(blob.get("global_optimizer_step", -1)) != int(step):
        raise RuntimeError("checkpoint step mismatch")
    # Projection may be present in ckpt but is intentionally ignored for frozen R198 extract.
    bn_dom = bn_bundle_domain(target)
    bn_sel = clone_bn_bundle(blob["bn_bundles"][bn_dom])
    with open(ROOT / "data_config.json", encoding="utf-8") as f:
        data_config = json.load(f)
    args = make_extract_args(target, cell, out_dir)
    set_seed(2)
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)
    if int(te_inds.numel()) != 0:
        raise RuntimeError("te_inds nonempty — refuse test")
    if int(tr_data[FORWARD_EDGE_TYPE].edge_attr.shape[1]) != EDGE_DIM:
        raise RuntimeError("edge_dim != 6")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = AddEgoIds()
    add_arange_ids([tr_data, val_data, te_data])
    sample_args = SimpleNamespace(**vars(args))
    sample_args.loader_num_workers = 0
    sample_loader, _, _ = get_loaders(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, sample_args, train_shuffle=False
    )
    sample_batch = next(iter(sample_loader))
    del sample_loader
    config = SimpleNamespace(
        model="gin",
        n_hidden=extract_param("n_hidden", args),
        n_gnn_layers=extract_param("n_gnn_layers", args),
        n_heads=None,
        dropout=extract_param("dropout", args),
        final_dropout=extract_param("final_dropout", args),
    )
    args.direct_r198_infonce = True
    model = get_model(sample_batch, config, args)
    model = to_hetero(model, tr_data.metadata(), aggr="mean")
    model.bypass_embedding_head = True
    model.load_state_dict(blob["model_state_dict"], strict=True)
    apply_bn_(model, bn_sel)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tr_loader, val_loader, te_loader = get_loaders(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args, train_shuffle=False
    )
    del te_loader
    staging = out_dir / f".staging_{os.getpid()}"
    staging.mkdir(parents=True, exist_ok=True)
    extracted = {}
    with torch.inference_mode():
        for split_name, loader, inds, gdata in (
            ("train", tr_loader, tr_inds, tr_data),
            ("val", val_loader, val_inds, val_data),
        ):
            expected = expected_seed_edge_ids(loader.data, inds, hetero=True)
            edge_ids, z, y = extract_seed_embeddings_hetero(
                loader, inds, model, gdata, device, args,
                representation_source="pre_embedding_3h",
                pre_dim=R198_DIM,
                emb_dim=int(getattr(model, "embedding_dim", R198_DIM)),
                head_spec=None,
                max_batches=None,
            )
            log_seed_coverage(edge_ids, expected, split_name)
            if int(z.shape[1]) != R198_DIM:
                raise RuntimeError(f"expected R198, got {z.shape}")
            save_embedding_split_npz(staging / f"{split_name}.npz", edge_ids=edge_ids, z=z, y=y)
            extracted[split_name] = {
                "n": int(z.shape[0]),
                "n_pos": int((y == 1).sum()) if hasattr(y, "__len__") else 0,
                "dim": int(z.shape[1]),
            }
    for split_name in ("train", "val"):
        os.replace(staging / f"{split_name}.npz", out_dir / f"{split_name}.npz")
    if (out_dir / "test.npz").is_file():
        raise RuntimeError("test.npz written — refuse")
    meta = {
        "ok": True,
        "arm": arm,
        "step": int(step),
        "target": target,
        "cell": cell,
        "checkpoint": str(ckpt_p.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(ckpt_p),
        "representation": "direct_frozen_r198",
        "projection_used": False,
        "concat_tf_or_native": False,
        "test_access": False,
        "extracted": extracted,
        "end": "SUCCESS",
    }
    write_json(out_dir / "meta.json", meta)
    cells_dir = ROOT / result_root(arm) / "cells"
    write_json(cells_dir / f"{cell}_extract.json", meta)
    del model, tr_data, val_data, te_data
    gc.collect()
    return meta


def run_probe(arm: str, step: int, target: str) -> Dict[str, Any]:
    require_slurm()
    cell = cell_name(arm, step, target)
    emb_dir = ROOT / emb_cell_dir(arm, step, target)
    tr = np.load(emb_dir / "train.npz")
    va = np.load(emb_dir / "val.npz")
    if int(tr["z"].shape[1]) != R198_DIM or int(va["z"].shape[1]) != R198_DIM:
        raise RuntimeError("probe requires R198 only")
    out = fit_r198_probe(tr["z"], tr["y"], va["z"], va["y"], device=torch.device("cpu"))
    payload = {
        "ok": True,
        "arm": arm,
        "step": int(step),
        "target": target,
        "cell": cell,
        "probe": PROBE,
        "metrics": out,
        "representation": "direct_frozen_r198",
        "end": "SUCCESS",
    }
    write_json(ROOT / result_root(arm) / "cells" / f"{cell}_probe.json", payload)
    return payload


def run_finalize(arm: str) -> Dict[str, Any]:
    require_slurm()
    cells = []
    missing = []
    for step, target in all_extract_cells(arm):
        cell = cell_name(arm, step, target)
        p = ROOT / result_root(arm) / "cells" / f"{cell}_probe.json"
        if not p.is_file():
            missing.append(str(p))
            continue
        blob = json.loads(p.read_text(encoding="utf-8"))
        if not blob.get("ok"):
            missing.append(str(p) + ":ok=false")
            continue
        cells.append(blob)
    ok = not missing
    payload = {
        "ok": ok,
        "arm": arm,
        "n_cells": len(cells),
        "missing": missing,
        "cells": [
            {
                "step": c["step"],
                "target": c["target"],
                "val_auprc": (c.get("metrics") or {}).get("validation_auprc"),
                "val_f1": ((c.get("metrics") or {}).get("validation_metrics_at_0.5") or {}).get("f1"),
            }
            for c in cells
        ],
        "cross_arm_aggregation": "deferred",
        "end": "SUCCESS" if ok else "FAILED",
    }
    write_json(ROOT / result_root(arm) / "finalize.json", payload)
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    se = sub.add_parser("extract")
    se.add_argument("--arm", required=True, choices=tuple(ARMS))
    se.add_argument("--step", type=int, required=True)
    se.add_argument("--target", required=True, choices=tuple(TARGETS))
    sp = sub.add_parser("probe")
    sp.add_argument("--arm", required=True, choices=tuple(ARMS))
    sp.add_argument("--step", type=int, required=True)
    sp.add_argument("--target", required=True, choices=tuple(TARGETS))
    sf = sub.add_parser("finalize")
    sf.add_argument("--arm", required=True, choices=tuple(ARMS))
    sd = sub.add_parser("dry-cells")
    sd.add_argument("--arm", required=True, choices=tuple(ARMS))
    args = ap.parse_args()
    logger_setup()
    if args.cmd == "dry-cells":
        print(json.dumps({"arm": args.arm, "cells": all_extract_cells(args.arm)}, indent=2))
        return 0
    if args.cmd == "extract":
        print(json.dumps(run_extract(args.arm, args.step, args.target), indent=2, default=str))
        return 0
    if args.cmd == "probe":
        print(json.dumps(run_probe(args.arm, args.step, args.target), indent=2, default=str))
        return 0
    if args.cmd == "finalize":
        out = run_finalize(args.arm)
        print(json.dumps(out, indent=2, default=str))
        return 0 if out.get("ok") else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
