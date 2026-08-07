#!/usr/bin/env python3
"""Corrected full-subgraph re-eval for one (run, epoch) of the 40ep linear-LR sweep.

- Extracts via extract_direct_r198_full_cell into a NEW embeddings root (never
  overwrites buggy seed-only artifacts under embeddings/<run>_epochXX/).
- Verifies val edge IDs: zero train overlap and match vs 10ep full-extract reference.
- Probes with PaperStyleMLP matching the 10ep scheduled analysis
  (best-by-val-AUPRC for ranking metrics; also records last-epoch BCE).

Never trains. Never loads/writes test.npz.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gcpal_txn_node.eval_mlp import PaperStyleMLP, _predict_proba  # noqa: E402
from util import set_seed  # noqa: E402
from direct_r198_eval_protocol import (  # noqa: E402
    PROTOCOL_FULL_SUBGRAPH,
    TIER_OFFICIAL,
    official_protocol_block,
)

MLP_EPOCHS = 20
MLP_LR = 1e-3
MLP_BS = 8192
MLP_SEED = 2
TF_CACHE = ROOT / "results/cache/temporal_flow_causal/Small-HI"

# Reference from known-correct 10ep full extract (true val seed set).
REF_VAL_EMB = (
    ROOT
    / "embeddings/direct_h_infonce_10ep_seed2_sched_epoch03/pre_embedding_3h/val.npz"
)
REF_TRAIN_EMB = (
    ROOT
    / "embeddings/direct_h_infonce_10ep_seed2_sched_epoch03/pre_embedding_3h/train.npz"
)

EMB_ROOT_DEFAULT = "embeddings/direct_r198_40ep_linear_lr_full_extract"
OUT_DEFAULT = "results/diagnostics/direct_r198_40ep_linear_lr_full_extract_reeval"

RUNS = [
    {
        "run": "direct_r198_infonce_40ep_seed2_linear_lr6p2e-3",
        "arm": "DIRECT_H",
        "peak_lr": 0.006213266113989207,
    },
    {
        "run": "direct_r198_tfmoe_40ep_seed2_linear_lr6p2e-3",
        "arm": "DIRECT_H_TFMOE",
        "peak_lr": 0.006213266113989207,
    },
    {
        "run": "direct_r198_infonce_40ep_seed2_linear_lr2e-3",
        "arm": "DIRECT_H",
        "peak_lr": 0.002,
    },
    {
        "run": "direct_r198_tfmoe_40ep_seed2_linear_lr2e-3",
        "arm": "DIRECT_H_TFMOE",
        "peak_lr": 0.002,
    },
]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cell_complete(emb: Path) -> bool:
    return (
        (emb / "train.npz").is_file()
        and (emb / "val.npz").is_file()
        and (emb / "meta.json").is_file()
        and not (emb / "test.npz").is_file()
    )


def _load_x_tf() -> Tuple[np.ndarray, np.ndarray]:
    spec = importlib.util.spec_from_file_location(
        "probe_feature_ablation", ROOT / "scripts" / "probe_feature_ablation.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["probe_feature_ablation"] = mod
    spec.loader.exec_module(mod)
    df, df_train, _, _, _, _ = mod.load_dataset_frames(
        "Small-HI", str(ROOT / "data_config.json")
    )
    x_raw, _, _, _ = mod.build_full_feature_matrix(
        df, df_train, ("edge_native",), categorical_encoding="one_hot"
    )
    tf = np.load(TF_CACHE / "features.npy").astype(np.float32)
    return x_raw.astype(np.float32), tf


def _bce_np(logits: np.ndarray, y: np.ndarray) -> float:
    t = torch.from_numpy(logits.astype(np.float32))
    yb = torch.from_numpy(y.astype(np.float32))
    return float(nn.functional.binary_cross_entropy_with_logits(t, yb).item())


def _metrics(y: np.ndarray, p: np.ndarray, thr: float) -> Dict[str, float]:
    pred = (p >= thr).astype(np.int64)
    return {
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "threshold": float(thr),
    }


def _tune_thr(y: np.ndarray, p: np.ndarray) -> float:
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.01, 0.99, 99):
        f1 = float(
            f1_score(y.astype(np.int64), (p >= thr).astype(np.int64), zero_division=0)
        )
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr


def verify_edge_ids(emb: Path) -> Dict[str, Any]:
    """Safety + tolerant agreement vs a prior full extract (not exact set equality).

    Hard requirements (catch the seed-only train-range ID bug):
      - zero train∩val edge-id overlap
      - all validation IDs above the reference train max (true val range)
      - no seed-only-style collapse of val IDs into the train prefix
      - R198 dims; no test.npz

    Soft agreement with 10ep full-extract reference (coverage jitter is expected):
      - Jaccard(train), Jaccard(val) >= 0.999
      - relative |n - n_ref| / n_ref <= 1% for train and val
    Exact set equality to one epoch03 file is recorded but not required.
    """
    tr = np.load(emb / "train.npz")
    va = np.load(emb / "val.npz")
    id_tr = tr["edge_id"].astype(np.int64).reshape(-1)
    id_va = va["edge_id"].astype(np.int64).reshape(-1)
    y_tr = tr["y"].astype(np.int64).reshape(-1)
    y_va = va["y"].astype(np.int64).reshape(-1)

    set_tr = set(id_tr.tolist())
    set_va = set(id_va.tolist())
    inter = set_tr & set_va

    ref_va = np.load(REF_VAL_EMB)["edge_id"].astype(np.int64).reshape(-1)
    ref_tr = np.load(REF_TRAIN_EMB)["edge_id"].astype(np.int64).reshape(-1)
    set_ref_va = set(ref_va.tolist())
    set_ref_tr = set(ref_tr.tolist())

    train_max_ref = int(ref_tr.max())
    val_min_ref = int(ref_va.min())
    val_max_ref = int(ref_va.max())

    def _jaccard(a: Set[int], b: Set[int]) -> float:
        if not a and not b:
            return 1.0
        union = a | b
        if not union:
            return 1.0
        return float(len(a & b) / len(union))

    j_train = _jaccard(set_tr, set_ref_tr)
    j_val = _jaccard(set_va, set_ref_va)
    n_train = int(id_tr.size)
    n_val = int(id_va.size)
    ref_n_train = int(ref_tr.size)
    ref_n_val = int(ref_va.size)
    rel_train = abs(n_train - ref_n_train) / max(ref_n_train, 1)
    rel_val = abs(n_val - ref_n_val) / max(ref_n_val, 1)

    # Fraction of reported "val" IDs that sit in the train prefix — ~1.0 under the
    # old seed-only bug; ~0 for a correct full extract.
    n_val_in_train_range = int((id_va <= train_max_ref).sum()) if id_va.size else 0
    frac_val_in_train_range = float(n_val_in_train_range / max(n_val, 1))
    # Classic buggy seed-only signature: ~138k rows, almost all IDs in train range.
    seed_only_bug_evidence = bool(
        frac_val_in_train_range > 0.01
        or (n_val > 0 and int(id_va.max()) <= train_max_ref)
        or (n_val < 0.5 * ref_n_val and frac_val_in_train_range > 0.0)
    )

    checks = {
        "n_train": n_train,
        "n_val": n_val,
        "n_train_pos": int((y_tr == 1).sum()),
        "n_val_pos": int((y_va == 1).sum()),
        "train_eid_min": int(id_tr.min()) if id_tr.size else None,
        "train_eid_max": int(id_tr.max()) if id_tr.size else None,
        "val_eid_min": int(id_va.min()) if id_va.size else None,
        "val_eid_max": int(id_va.max()) if id_va.size else None,
        "train_val_intersect": int(len(inter)),
        "val_in_train_frac": float(len(inter) / max(len(set_va), 1)),
        "ref_n_val": ref_n_val,
        "ref_n_train": ref_n_train,
        "ref_val_min": val_min_ref,
        "ref_val_max": val_max_ref,
        "ref_train_max": train_max_ref,
        "val_set_equals_ref": bool(set_va == set_ref_va),  # diagnostic only
        "val_subset_of_ref": bool(set_va <= set_ref_va),
        "train_set_equals_ref": bool(set_tr == set_ref_tr),
        "all_val_above_ref_train_max": bool(
            id_va.size > 0 and int(id_va.min()) > train_max_ref
        ),
        "frac_val_ids_in_train_range": frac_val_in_train_range,
        "n_val_ids_in_train_range": n_val_in_train_range,
        "seed_only_bug_evidence": seed_only_bug_evidence,
        "jaccard_train_vs_ref": j_train,
        "jaccard_val_vs_ref": j_val,
        "rel_diff_n_train_vs_ref": float(rel_train),
        "rel_diff_n_val_vs_ref": float(rel_val),
        "jaccard_min_required": 0.999,
        "rel_diff_n_max_allowed": 0.01,
        "train_symdiff_vs_ref": int(len(set_tr ^ set_ref_tr)),
        "val_symdiff_vs_ref": int(len(set_va ^ set_ref_va)),
        "z_dim_train": int(tr["Z"].shape[1]),
        "z_dim_val": int(va["Z"].shape[1]),
        "no_test_npz": not (emb / "test.npz").is_file(),
    }
    checks["train_near_ref"] = bool(
        j_train >= 0.999 and rel_train <= 0.01
    )
    checks["val_near_ref"] = bool(j_val >= 0.999 and rel_val <= 0.01)

    ok = (
        checks["train_val_intersect"] == 0
        and checks["all_val_above_ref_train_max"]
        and not checks["seed_only_bug_evidence"]
        and checks["train_near_ref"]
        and checks["val_near_ref"]
        and checks["z_dim_train"] == 198
        and checks["z_dim_val"] == 198
        and checks["no_test_npz"]
    )
    checks["ok"] = bool(ok)
    if not ok:
        checks["fail_reasons"] = []
        if checks["train_val_intersect"] != 0:
            checks["fail_reasons"].append("train_val_overlap")
        if not checks["all_val_above_ref_train_max"]:
            checks["fail_reasons"].append("val_ids_not_above_train_max")
        if checks["seed_only_bug_evidence"]:
            checks["fail_reasons"].append("seed_only_train_range_mapping")
        if not checks["train_near_ref"]:
            checks["fail_reasons"].append("train_disagree_vs_full_ref")
        if not checks["val_near_ref"]:
            checks["fail_reasons"].append("val_disagree_vs_full_ref")
        if checks["z_dim_train"] != 198 or checks["z_dim_val"] != 198:
            checks["fail_reasons"].append("bad_r198_dim")
        if not checks["no_test_npz"]:
            checks["fail_reasons"].append("test_npz_present")
    return checks


def _fit_probe(
    mat_tr: np.ndarray,
    y_tr: np.ndarray,
    mat_va: np.ndarray,
    y_va: np.ndarray,
    device,
) -> Dict[str, Any]:
    """Match 10ep analysis: ranking metrics from best-val-AUPRC epoch; also last-epoch BCE."""
    scaler = StandardScaler()
    tr = scaler.fit_transform(mat_tr).astype(np.float32)
    va = scaler.transform(mat_va).astype(np.float32)
    torch.manual_seed(MLP_SEED)
    np.random.seed(MLP_SEED)
    model = PaperStyleMLP(tr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=MLP_LR)
    x_t = torch.from_numpy(tr)
    y_t = torch.from_numpy(y_tr.astype(np.float32))
    n = tr.shape[0]
    history: List[Dict[str, float]] = []
    best_auprc, best_state, best_ep = -1.0, None, -1
    last_tr_log = None
    last_va_log = None

    for ep in range(MLP_EPOCHS):
        model.train()
        perm = np.random.RandomState(MLP_SEED * 1009 + ep).permutation(n)
        for start in range(0, n, MLP_BS):
            idx = perm[start : start + MLP_BS]
            opt.zero_grad(set_to_none=True)
            logits = model(x_t[idx].to(device))
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, y_t[idx].to(device)
            )
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            tr_logits = []
            for start in range(0, n, MLP_BS):
                tr_logits.append(
                    model(x_t[start : start + MLP_BS].to(device)).detach().cpu().numpy()
                )
            tr_log = np.concatenate(tr_logits, axis=0)
            va_log = []
            xva = torch.from_numpy(va)
            for start in range(0, va.shape[0], MLP_BS):
                va_log.append(
                    model(xva[start : start + MLP_BS].to(device)).detach().cpu().numpy()
                )
            va_log_a = np.concatenate(va_log, axis=0)
        last_tr_log, last_va_log = tr_log, va_log_a
        pva = 1.0 / (1.0 + np.exp(-np.clip(va_log_a, -50, 50)))
        auprc = float(average_precision_score(y_va, pva))
        history.append(
            {
                "epoch": ep + 1,
                "train_bce": _bce_np(tr_log, y_tr),
                "val_bce": _bce_np(va_log_a, y_va),
                "val_auprc": auprc,
            }
        )
        if auprc > best_auprc + 1e-12:
            best_auprc = auprc
            best_ep = ep + 1
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    assert best_state is not None and last_tr_log is not None and last_va_log is not None
    # Ranking metrics from best-val-AUPRC weights (10ep analysis convention).
    model.load_state_dict(best_state)
    model.to(device)
    pva = _predict_proba(model, va, batch_size=MLP_BS, device=device)
    thr = _tune_thr(y_va, pva)
    return {
        "learner": "PaperStyleMLP",
        "loss": "binary_cross_entropy_with_logits",
        "logits": "one_logit",
        "class_weights": None,
        "pos_weight": None,
        "reduction": "mean",
        "mlp_epochs": MLP_EPOCHS,
        "mlp_lr": MLP_LR,
        "mlp_batch_size": MLP_BS,
        "mlp_seed": MLP_SEED,
        "selection_within_probe": "best_val_auprc",
        "best_probe_epoch": int(best_ep),
        "validation_auprc": float(average_precision_score(y_va, pva)),
        "validation_auroc": float(roc_auc_score(y_va, pva))
        if len(np.unique(y_va)) > 1
        else float("nan"),
        "validation_metrics_at_0.5": _metrics(y_va, pva, 0.5),
        "validation_metrics_at_val_optimal_f1": {
            **_metrics(y_va, pva, thr),
            "optimistic_diagnostic": True,
        },
        "final_probe_train_bce": float(history[-1]["train_bce"]),
        "final_probe_val_bce": float(history[-1]["val_bce"]),
        "epoch_history": history,
        "input_dim": int(tr.shape[1]),
        "n_train": int(n),
        "n_val": int(y_va.shape[0]),
        "test_evaluated": False,
    }


def _stack(z, edge_id, x_raw, tf, mode: str):
    eid = edge_id.astype(np.int64)
    if mode == "primary":
        return np.concatenate([z, x_raw[eid], tf[eid]], axis=1)
    if mode == "diagnostic":
        return z
    raise ValueError(mode)


def _repr_stats(z: np.ndarray) -> Dict[str, float]:
    zn = z.astype(np.float64)
    norms = np.linalg.norm(zn, axis=1)
    zc = zn - zn.mean(axis=0, keepdims=True)
    rs = np.random.RandomState(2)
    if zc.shape[0] > 50000:
        zc = zc[rs.choice(zc.shape[0], 50000, replace=False)]
    try:
        s = np.linalg.svd(zc, full_matrices=False, compute_uv=False)
        p = (s ** 2)
        p = p / max(p.sum(), 1e-12)
        eff = float(np.exp(-(p * np.log(np.maximum(p, 1e-300))).sum()))
    except Exception:
        eff = float("nan")
    return {
        "mean_l2_norm": float(norms.mean()),
        "std_l2_norm": float(norms.std()),
        "effective_rank": eff,
    }


def extract_full(run: str, epoch: int, emb_root: str) -> Path:
    emb = ROOT / emb_root / f"{run}_epoch{epoch:02d}" / "pre_embedding_3h"
    if _cell_complete(emb):
        logging.info("Reuse existing full extract %s", emb)
        return emb
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    cmd = [
        sys.executable,
        str(ROOT / "scripts/extract_direct_r198_full_cell.py"),
        "--run",
        run,
        "--epoch",
        str(epoch),
        "--splits",
        "train,val",
        "--embeddings_dir",
        emb_root,
    ]
    logging.info("Subprocess full extract: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)
    if not _cell_complete(emb):
        raise RuntimeError(f"Full extract incomplete: {emb}")
    if (emb / "test.npz").is_file():
        raise RuntimeError(f"test.npz present: {emb}")
    # Stamp extractor provenance
    meta_path = emb / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    meta["seed_only_r198"] = False
    meta["extractor"] = "extract_direct_r198_full_cell"
    meta["reeval_note"] = (
        "Corrected full-subgraph extract for 40ep linear-LR sweep; "
        "does not overwrite seed-only artifacts."
    )
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return emb


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--arm", required=True, choices=["DIRECT_H", "DIRECT_H_TFMOE"])
    ap.add_argument("--peak_lr", type=float, required=True)
    ap.add_argument("--embeddings_dir", type=str, default=EMB_ROOT_DEFAULT)
    ap.add_argument("--out_dir", type=str, default=OUT_DEFAULT)
    ap.add_argument("--skip_extract", action="store_true")
    ap.add_argument("--skip_probe", action="store_true")
    ap.add_argument(
        "--probe_feature_protocol",
        type=str,
        default="both",
        choices=["both", "R198_only", "R198_X_TF"],
        help=(
            "both=legacy (primary=R198+X+TF, diagnostic=R198-only); "
            "R198_only=probe Z only (dim=198) as primary; "
            "R198_X_TF=probe stacked features only."
        ),
    )
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    cell_dir = out_dir / "cells" / args.run
    cell_dir.mkdir(parents=True, exist_ok=True)

    # Refuse writing R198-only ablation into official collaborator trees.
    if args.probe_feature_protocol == "R198_only":
        collab_pkg = (
            ROOT
            / "results/diagnostics/direct_r198_40ep_linear_lr_full_extract_reeval/collaborator_package"
        )
        if out_dir.resolve() == collab_pkg.resolve() or collab_pkg.resolve() in out_dir.resolve().parents:
            raise SystemExit("Refuse R198_only writes into collaborator_package")
        if out_dir.resolve() == (ROOT / OUT_DEFAULT).resolve():
            raise SystemExit(
                "Refuse R198_only writes into official full_extract_reeval root; "
                "use a dedicated --out_dir"
            )

    # Guard: never write under the default seed-only embedding path
    seed_only_path = ROOT / "embeddings" / f"{args.run}_epoch{args.epoch:02d}"
    full_path = ROOT / args.embeddings_dir / f"{args.run}_epoch{args.epoch:02d}"
    if full_path.resolve() == seed_only_path.resolve():
        raise SystemExit(
            "Refusing to write into seed-only embedding path; choose a distinct --embeddings_dir"
        )

    if not args.skip_extract:
        emb = extract_full(args.run, args.epoch, args.embeddings_dir)
    else:
        emb = ROOT / args.embeddings_dir / f"{args.run}_epoch{args.epoch:02d}" / "pre_embedding_3h"
        if not _cell_complete(emb):
            raise SystemExit(f"Missing extract: {emb}")

    verify = verify_edge_ids(emb)
    verify_path = cell_dir / f"epoch_{args.epoch:02d}_verify.json"
    verify_path.write_text(json.dumps(verify, indent=2) + "\n")
    logging.info("Verify ok=%s n_val=%s intersect=%s", verify["ok"], verify["n_val"], verify["train_val_intersect"])
    if not verify["ok"]:
        report = {
            "status": "verify_failed",
            "run": args.run,
            "epoch": args.epoch,
            "arm": args.arm,
            "peak_lr": args.peak_lr,
            "embedding_dir": str(emb),
            "verify": verify,
            "seed_only_metrics_invalid": True,
        }
        (cell_dir / f"epoch_{args.epoch:02d}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(json.dumps(report, indent=2))
        return 2

    if args.skip_probe:
        print(json.dumps({"status": "verify_ok_skip_probe", "verify": verify}, indent=2))
        return 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tr = np.load(emb / "train.npz")
    va = np.load(emb / "val.npz")
    z_tr, y_tr, id_tr = tr["Z"], tr["y"].reshape(-1), tr["edge_id"].reshape(-1)
    z_va, y_va, id_va = va["Z"], va["y"].reshape(-1), va["edge_id"].reshape(-1)
    assert int(z_tr.shape[1]) == 198 and int(z_va.shape[1]) == 198

    protocol = str(args.probe_feature_protocol)
    prim = None
    diag = None
    if protocol in ("both", "R198_X_TF"):
        logging.info("Loading X+TF for stacked probe")
        x_raw, tf = _load_x_tf()
        set_seed(MLP_SEED)
        prim = _fit_probe(
            _stack(z_tr, id_tr, x_raw, tf, "primary"),
            y_tr,
            _stack(z_va, id_va, x_raw, tf, "primary"),
            y_va,
            device,
        )
        assert int(prim["input_dim"]) == 227
    if protocol in ("both", "R198_only"):
        set_seed(MLP_SEED)
        diag = _fit_probe(z_tr, y_tr, z_va, y_va, device)
        assert int(diag["input_dim"]) == 198

    if protocol == "R198_only":
        prim = diag
        reported_protocol = "R198_only"
        concat_x = False
        concat_tf = False
        tier = "r198_only_weight_ablation"
        merge_ok = False
    elif protocol == "R198_X_TF":
        reported_protocol = "R198_X_TF"
        concat_x = True
        concat_tf = True
        tier = TIER_OFFICIAL
        merge_ok = True
    else:
        reported_protocol = "R198_X_TF_primary_plus_R198_only_diagnostic"
        concat_x = True
        concat_tf = True
        tier = TIER_OFFICIAL
        merge_ok = True

    ckpt = ROOT / "saved-models" / f"checkpoint_{args.run}_epoch{args.epoch:02d}.tar"
    protocol_block = official_protocol_block(embeddings_dir=args.embeddings_dir)
    if protocol == "R198_only":
        protocol_block = dict(protocol_block)
        protocol_block["evaluation_tier"] = tier
        protocol_block["collaborator_merge_allowed"] = False
        protocol_block["probe"] = {
            **protocol_block.get("probe", {}),
            "features": "R198 only",
            "probe_feature_protocol": "R198_only",
            "input_dim": 198,
        }

    report = {
        "status": "ok",
        "arm": args.arm,
        "run": args.run,
        "peak_lr": args.peak_lr,
        "schedule": "direct_h_warmup_linear",
        "epoch": args.epoch,
        "checkpoint": str(ckpt),
        "checkpoint_sha256": _sha256_file(ckpt) if ckpt.is_file() else None,
        "embedding_dir": str(emb),
        "extractor": "full_subgraph_run_embedding_extraction",
        "protocol": PROTOCOL_FULL_SUBGRAPH,
        "probe_feature_protocol": reported_protocol if protocol != "both" else "R198_X_TF",
        "concatenated_raw_edge_X": concat_x if protocol != "both" else True,
        "concatenated_temporal_flow": concat_tf if protocol != "both" else True,
        "evaluation_tier": tier,
        "collaborator_merge_allowed": merge_ok,
        "protocol_block": protocol_block,
        "seed_only_r198": False,
        "seed_only_prior_metrics_invalid": True,
        "verify": verify,
        "repr_val": _repr_stats(z_va.astype(np.float32)),
        "primary": prim,
        "diagnostic": diag if protocol == "both" else (diag if protocol == "R198_only" else None),
        "test_evaluated": False,
    }
    if protocol == "R198_only":
        report["probe_input_dim"] = 198
        report["thesis_primary"] = False
    out_json = cell_dir / f"epoch_{args.epoch:02d}.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n")
    logging.info(
        "OK ep=%s protocol=%s AUPRC=%.4f F1@0.5=%.4f F1@val-thr=%.4f final_val_BCE=%.4f input_dim=%s",
        args.epoch,
        protocol,
        prim["validation_auprc"],
        prim["validation_metrics_at_0.5"]["f1"],
        prim["validation_metrics_at_val_optimal_f1"]["f1"],
        prim["final_probe_val_bce"],
        prim["input_dim"],
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "run": args.run,
                "epoch": args.epoch,
                "probe_feature_protocol": protocol,
                "out": str(out_json),
                "val_auprc": prim["validation_auprc"],
                "f1_at_0.5": prim["validation_metrics_at_0.5"]["f1"],
                "f1_at_val_thr": prim["validation_metrics_at_val_optimal_f1"]["f1"],
                "final_probe_train_bce": prim["final_probe_train_bce"],
                "final_probe_val_bce": prim["final_probe_val_bce"],
                "input_dim": prim["input_dim"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
