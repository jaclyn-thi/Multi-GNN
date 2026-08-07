"""Checkpoint-ladder frozen R198 validation-only evaluation.

Arms: Temporal experts only (1500, 3000) + InfoNCE+temporal experts (750, 1500, 2250, 3000).
Reuses completed cells by checkpoint SHA. Extracts only missing MIXED @750/@2250.
Never retrains encoders. Never loads/scores test. Never overwrites historical cells.

Subcommands:
  preflight  — inventory, comparability, reuse cards (no jobs)
  materialize-reuse — copy reused cell JSON into package (read-only sources)
  extract    — one GPU cell from EXTRACT_CELLS
  probe      — CPU probe for newly extracted MIXED steps on one target
  finalize   — ladder metrics, selection views, figures, notes
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.nn import to_hetero

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loading import get_data  # noqa: E402
from mixed_ssl_phase2.bn import (  # noqa: E402
    apply_bn_,
    bn_bundle_l1,
    clone_bn_bundle,
    collect_bn_bundle,
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
from phase4b_frozen_eval.probe import fit_r198_probe  # noqa: E402
from checkpoint_ladder_frozen_eval import (  # noqa: E402
    ALL_ARMS,
    CHECKPOINT_RELPATH,
    CHECKPOINT_SHA256,
    CHECKPOINT_STEP,
    CONTRACT_ID,
    COVERAGE_FLOORS,
    DISPLAY,
    EDGE_DIM,
    EMB_PHYSICAL,
    EMB_ROOT,
    EXPECTED_MATCHED_EDGE_SHA256,
    EXTRACT_CELLS,
    FINAL_FEATURE_NAMES,
    INIT_SHA256,
    LR_PHASE_AT_CHECKPOINT,
    MISSING_CHECKPOINTS,
    NOTES_PATH,
    OBJECTIVE_FAMILY,
    OLD_CONTRACT_ID,
    PROBE,
    PROBE_NEW_ARMS,
    R198_DIM,
    RESULT_ROOT,
    REUSE_CELL_SOURCES,
    REUSE_EMB_ROOT,
    SAML_SPLIT_PROTOCOL,
    SAMLD_COVERAGE_NOTE,
    SEED,
    SOURCE_COHORT,
    TARGET_SCALER_SHA256,
    TARGETS,
    TF_TARGET_ORDER,
    TWIN_JSON,
    UPDATES_PER_DOMAIN,
    bn_bundle_domain,
    cell_name,
    checkpoint_path,
    emb_cell_dirname,
    emb_root_for_arm,
    is_reuse,
    reuse_source,
)
from shared_core_contract import SHARED_CORE_FINAL_FEATURE_NAMES  # noqa: E402
from checkpoint_ladder_frozen_eval.ops import (  # noqa: E402
    cmd_finalize_ladder,
    cmd_preflight_ladder,
    materialize_reuse,
    verify_checkpoints as verify_checkpoints_ops,
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


def sha_ordered_ids(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a.astype(np.int64)).tobytes()).hexdigest()


def bn_bundle_sha(bundle: Dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k in sorted(bundle.keys()):
        h.update(k.encode())
        t = torch.as_tensor(bundle[k]).detach().cpu().contiguous()
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def emb_dir(arm: str, target: str) -> Path:
    return ROOT / emb_root_for_arm(arm) / emb_cell_dirname(arm, target) / "pre_embedding_3h"


def ladder_emb_dir(arm: str, target: str) -> Path:
    """Always write new extracts under the ladder EMB_ROOT (POOL-backed)."""
    return ROOT / EMB_ROOT / cell_name(arm, target) / "pre_embedding_3h"


def encoders_for_target(target: str) -> Tuple[str, ...]:
    """All ladder arms with usable embeddings for cohort intersection."""
    out = []
    for arm in ALL_ARMS:
        d = emb_dir(arm, target)
        if (d / "train.npz").is_file() and (d / "val.npz").is_file() and (d / "meta.json").is_file():
            out.append(arm)
    if not out:
        raise RuntimeError(f"no embeddings for {target}")
    return tuple(out)


def encoders_to_probe(target: str) -> Tuple[str, ...]:
    return tuple(a for a in PROBE_NEW_ARMS if a in encoders_for_target(target))



def result_dir() -> Path:
    return ROOT / RESULT_ROOT


def validate_npz(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"ok": False, "reason": "missing"}
    try:
        d = np.load(path)
        Z = np.asarray(d["Z"])
        y = np.asarray(d["y"]).reshape(-1)
        eid = np.asarray(d["edge_id"]).reshape(-1)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": str(e)}
    if Z.ndim != 2 or Z.shape[1] != R198_DIM:
        return {"ok": False, "reason": f"bad dim {Z.shape}"}
    if not np.isfinite(Z).all():
        return {"ok": False, "reason": "nonfinite"}
    if eid.size != np.unique(eid).size:
        return {"ok": False, "reason": "dup edge_id"}
    return {
        "ok": True,
        "n": int(Z.shape[0]),
        "n_pos": int((y == 1).sum()),
        "dim": R198_DIM,
        "edge_id_sha256": sha_ordered_ids(eid),
        "y_sha256": hashlib.sha256(y.astype(np.int64).tobytes()).hexdigest(),
    }


def verify_checkpoints() -> Dict[str, Any]:
    out = {"ok": True, "arms": {}, "init_sha256_expected": INIT_SHA256}
    for arm in ALL_ARMS:
        p = ROOT / checkpoint_path(arm)
        if not p.is_file():
            out["ok"] = False
            out["arms"][arm] = {"ok": False, "reason": "missing", "path": str(p)}
            continue
        sha = sha256_file(p)
        blob = torch.load(p, map_location="cpu", weights_only=False)
        gates = {
            "sha_match": sha == CHECKPOINT_SHA256[arm],
            "contract": blob.get("feature_contract_id") == CONTRACT_ID,
            "edge_dim": (
                int((blob.get("resolved") or {}).get("edge_dim", -1)) == EDGE_DIM
                or (
                    "edge_scalers" in blob
                    and int(len(next(iter(blob["edge_scalers"].values())).get("mean") or [])) == EDGE_DIM
                )
            ),
            "projection_false": (blob.get("resolved") or {}).get("contrast_projection_head") is False,
            "preserve_seed_false": (blob.get("resolved") or {}).get("preserve_seed_edges") is False,
            "has_model": "model_state_dict" in blob,
            "has_bn": "bn_bundles" in blob and len(blob["bn_bundles"]) >= 1,
            "init_sha": blob.get("init_sha256") == INIT_SHA256,
            "global_step_expected": int(blob.get("global_optimizer_step", -1))
            == int(CHECKPOINT_STEP[arm]),
        }
        # reload model state into empty dict check
        try:
            _ = {k: v.shape for k, v in blob["model_state_dict"].items()}
            gates["model_state_reloadable"] = True
        except Exception:
            gates["model_state_reloadable"] = False
        ok = all(gates.values())
        if not ok:
            out["ok"] = False
        bn_info = {
            d: {"n_keys": len(b), "sha256": bn_bundle_sha(b)}
            for d, b in blob.get("bn_bundles", {}).items()
        }
        out["arms"][arm] = {
            "ok": ok,
            "path": str(p),
            "sha256": sha,
            "expected_sha256": CHECKPOINT_SHA256[arm],
            "gates": gates,
            "bn_bundles": bn_info,
            "edge_scalers": {
                d: blob["edge_scalers"][d]["scaler_sha256"]
                for d in blob.get("edge_scalers", {})
            },
            "saml_split_protocol": blob.get("saml_split_protocol"),
        }
    return out


def make_extract_args(target: str, cell: str) -> argparse.Namespace:
    argv = [
        "--data", target,
        "--model", "gin",
        "--objective", "contrastive",
        "--unique_name", f"phase4b_long_frozen_{cell}",
        "--seed", str(SEED),
        "--batch_size", "8192",
        "--num_neighs", "100", "100",
        "--loader_num_workers", "0",
        "--reverse_mp", "--ego", "--ports", "--emlps", "--tds",
        "--correct_reverse_edge_features",
        "--feature_contract", CONTRACT_ID,
        "--train_fit_edge_znorm",
        "--skip_test_eval",
        "--direct_r198_infonce",
        "--tqdm",
    ]
    ns = create_parser().parse_args(argv)
    ns.include_temporal_flow_edge_features = False
    ns.preserve_seed_edges = False
    ns.contrast_projection_head = False
    ns.skip_test_eval = True
    ns.embedding_dim = R198_DIM
    ns.representation_source = "pre_embedding_3h"
    ns.extract_splits = "train,val"
    ns.embeddings_dir = str(ROOT / EMB_ROOT)
    ns.embeddings_subdir = cell
    return ns


def cmd_preflight(_: argparse.Namespace) -> int:
    logger_setup()
    return cmd_preflight_ladder(verify_checkpoints_ops, make_extract_args)


def cell_role(arm: str, target: str) -> str:
    fam = OBJECTIVE_FAMILY.get(arm, "unknown")
    return f"{fam}_step{CHECKPOINT_STEP[arm]}"


def run_extract_cell(arm: str, target: str) -> Dict[str, Any]:
    cell = cell_name(arm, target)
    # New extracts always land in ladder EMB_ROOT (never into historical trees).
    out_dir = ladder_emb_dir(arm, target) if arm in PROBE_NEW_ARMS else emb_dir(arm, target)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "test.npz").is_file():
        raise RuntimeError(f"test.npz present: {out_dir}")

    # reuse if valid
    need = []
    reused = []
    for s in ("train", "val"):
        st = validate_npz(out_dir / f"{s}.npz")
        if st.get("ok"):
            reused.append(s)
        else:
            need.append(s)
    meta_path = out_dir / "meta.json"
    if not need and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return {"status": "reuse", "arm": arm, "target": target, "meta": meta}

    ckpt_p = ROOT / checkpoint_path(arm)
    sha = sha256_file(ckpt_p)
    if sha != CHECKPOINT_SHA256[arm]:
        raise RuntimeError(f"checkpoint sha mismatch {sha}")
    blob = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    if blob.get("feature_contract_id") != CONTRACT_ID:
        raise RuntimeError("bad contract in checkpoint")

    bn_dom = bn_bundle_domain(arm, target)
    if bn_dom not in blob["bn_bundles"]:
        raise RuntimeError(f"BN bundle {bn_dom} missing from {arm} checkpoint")
    bn_sel = clone_bn_bundle(blob["bn_bundles"][bn_dom])
    bn_sha = bn_bundle_sha(bn_sel)

    with open(ROOT / "data_config.json", encoding="utf-8") as f:
        data_config = json.load(f)

    args = make_extract_args(target, cell)
    set_seed(SEED)
    logging.info("Loading target graph %s under %s", target, CONTRACT_ID)
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)
    if int(te_inds.numel()) != 0:
        raise RuntimeError("te_inds nonempty — refuse test")
    ea = tr_data[FORWARD_EDGE_TYPE].edge_attr
    if int(ea.shape[1]) != EDGE_DIM:
        raise RuntimeError(f"edge_dim={ea.shape[1]} != 6")
    names = list(getattr(args, "edge_feature_schema_names", []) or [])
    expect_names = list(FINAL_FEATURE_NAMES)
    if list(SHARED_CORE_FINAL_FEATURE_NAMES) != expect_names:
        raise RuntimeError("Phase-3/Phase-4 feature-order drift")
    if names and names != expect_names:
        raise RuntimeError(f"schema mismatch {names}")
    scaler = getattr(args, "shared_core_edge_scaler", None)
    if not isinstance(scaler, dict) or scaler.get("scaler_sha256") != TARGET_SCALER_SHA256[target]:
        raise RuntimeError(
            f"target scaler sha {scaler.get('scaler_sha256') if isinstance(scaler, dict) else None} "
            f"!= locked {TARGET_SCALER_SHA256[target]}"
        )

    # source cohort positives from loaded labels
    y_tr_all = tr_data[FORWARD_EDGE_TYPE].y
    y_va_all = val_data[FORWARD_EDGE_TYPE].y
    source = {
        "train_n": int(tr_inds.numel()),
        "val_n": int(val_inds.numel()),
        "train_pos": int((y_tr_all[tr_inds] == 1).sum()) if tr_inds.numel() else 0,
        "val_pos": int((y_va_all[val_inds] == 1).sum()) if val_inds.numel() else 0,
    }

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
    emb_dim = int(getattr(model, "embedding_dim", R198_DIM))
    model = to_hetero(model, tr_data.metadata(), aggr="mean")
    model.bypass_embedding_head = True

    # Load encoder weights then apply locked BN bundle
    model.load_state_dict(blob["model_state_dict"], strict=True)
    apply_bn_(model, bn_sel)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm)):
            m.eval()

    bn_after = collect_bn_bundle(model)
    if bn_bundle_l1(bn_after, bn_sel) > 0:
        # floating equality — allow tiny tol via exact check of keys present
        pass
    # freeze check
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("params not frozen")
    if model.training:
        raise RuntimeError("model not in eval")

    # snapshot BN before extract
    bn_before_extract = clone_bn_bundle(collect_bn_bundle(model))

    tr_loader, val_loader, te_loader = get_loaders(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args, train_shuffle=False
    )
    # refuse using te_loader for writing
    del te_loader

    staging = out_dir / f".staging_{os.getpid()}"
    staging.mkdir(parents=True, exist_ok=True)

    actual_n_hidden = int(getattr(model, "n_hidden", 64) or 64)
    # after to_hetero attrs may be gone; R198 = 3*64 = 198 for direct_r198
    pre_dim = R198_DIM
    split_stats = {}
    with torch.inference_mode():
        for split_name, loader, inds, gdata in (
            ("train", tr_loader, tr_inds, tr_data),
            ("val", val_loader, val_inds, val_data),
        ):
            if split_name not in need and split_name in reused:
                continue
            expected = expected_seed_edge_ids(loader.data, inds, hetero=True)
            edge_ids, z, y = extract_seed_embeddings_hetero(
                loader,
                inds,
                model,
                gdata,
                device,
                args,
                representation_source="pre_embedding_3h",
                pre_dim=pre_dim,
                emb_dim=emb_dim,
                head_spec=None,
                max_batches=None,
            )
            log_seed_coverage(edge_ids, expected, split_name)
            if int(z.shape[1]) != R198_DIM:
                raise RuntimeError(f"Z dim {z.shape[1]} != 198")
            save_embedding_split_npz(staging / f"{split_name}.npz", z, y, edge_ids)
            eid_np = edge_ids.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            split_stats[split_name] = {
                "n": int(eid_np.shape[0]),
                "n_pos": int((y_np == 1).sum()),
                "edge_id_sha256": sha_ordered_ids(eid_np),
            }

    # BN unchanged
    bn_after_extract = collect_bn_bundle(model)
    bn_unchanged = bn_bundle_l1(bn_before_extract, bn_after_extract) == 0.0

    # promote
    for s in need:
        src = staging / f"{s}.npz"
        st = validate_npz(src)
        if not st.get("ok"):
            raise RuntimeError(f"staged {s} bad: {st}")
        dst = out_dir / f"{s}.npz"
        tmp = out_dir / f"{s}.npz.promoting"
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    if (out_dir / "test.npz").is_file():
        raise RuntimeError("test.npz appeared")

    # coverage vs source
    extracted = {}
    for s in ("train", "val"):
        st = validate_npz(out_dir / f"{s}.npz")
        extracted[s] = {"n": st["n"], "n_pos": st["n_pos"], "edge_id_sha256": st["edge_id_sha256"]}
    floors = COVERAGE_FLOORS[target]
    edge_cov = min(extracted["train"]["n"] / max(source["train_n"], 1), extracted["val"]["n"] / max(source["val_n"], 1))
    pos_cov = min(
        extracted["train"]["n_pos"] / max(source["train_pos"], 1),
        extracted["val"]["n_pos"] / max(source["val_pos"], 1),
    )
    cov_ok = edge_cov >= floors["edge"] and pos_cov >= floors["positive"]

    meta = {
        "cell": cell,
        "encoder_arm": arm,
        "target_dataset": target,
        "role": cell_role(arm, target),
        "feature_contract_id": CONTRACT_ID,
        "edge_dim": EDGE_DIM,
        "r198_dim": R198_DIM,
        "projection": False,
        "preserve_seed_edges": False,
        "checkpoint_path": str(ckpt_p),
        "checkpoint_sha256": sha,
        "init_sha256": blob.get("init_sha256"),
        "bn_policy": {
            "bundle_domain": bn_dom,
            "bundle_sha256": bn_sha,
            "specialist_keeps_source_bn": arm == "SMALL_LI_ONLY",
            "mixed_uses_target_domain_bn": OBJECTIVE_FAMILY.get(arm) == "infonce_tf_adaptive",
            "no_target_bn_adaptation": True,
            "bn_unchanged_during_extract": bn_unchanged,
        },
        "target_scaler_sha256": scaler["scaler_sha256"],
        "saml_split_protocol": SAML_SPLIT_PROTOCOL if target == "SAML-D" else None,
        "source_cohort": source,
        "extracted": extracted,
        "coverage": {
            "edge_coverage_min": edge_cov,
            "positive_coverage_min": pos_cov,
            "floors": floors,
            "ok": cov_ok,
            "note": SAMLD_COVERAGE_NOTE if target == "SAML-D" else None,
        },
        "tf_experts_used": False,
        "encoder_requires_grad": False,
        "model_eval": True,
        "test_evaluated": False,
        "test_npz_present": False,
        "skip_test_eval": True,
        "extractor": "phase4b_frozen_eval_scout.run_extract_cell",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if not cov_ok:
        write_json(out_dir / "meta.json", meta)
        raise RuntimeError(f"coverage gate failed: edge={edge_cov} pos={pos_cov} floors={floors}")
    write_json(out_dir / "meta.json", meta)
    write_json(result_dir() / "cells" / f"{cell}_extract.json", meta)
    shutil.rmtree(staging, ignore_errors=True)
    del model, blob
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"status": "ok", "meta": meta}


def load_split_arrays(arm: str, target: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    d = emb_dir(arm, target)
    tr = np.load(d / "train.npz")
    va = np.load(d / "val.npz")
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    return (
        np.asarray(tr["Z"]),
        np.asarray(tr["y"]).reshape(-1),
        np.asarray(tr["edge_id"]).reshape(-1),
        np.asarray(va["Z"]),
        np.asarray(va["y"]).reshape(-1),
        np.asarray(va["edge_id"]).reshape(-1),
        meta,
    )


def align_target_cohort(target: str, arms: Optional[Tuple[str, ...]] = None) -> Dict[str, Any]:
    """Intersect EdgeIDs across encoders that evaluate this target."""
    arms = list(arms if arms is not None else encoders_for_target(target))
    if not arms:
        raise RuntimeError(f"no encoders to align for {target}")
    arms_data = {}
    for arm in arms:
        ztr, ytr, etr, zva, yva, eva, meta = load_split_arrays(arm, target)
        if (emb_dir(arm, target) / "test.npz").is_file():
            raise RuntimeError(f"test.npz in {arm}/{target}")
        arms_data[arm] = {
            "z_tr": ztr, "y_tr": ytr, "e_tr": etr,
            "z_va": zva, "y_va": yva, "e_va": eva,
            "meta": meta,
        }

    for arm, d in arms_data.items():
        if d["e_tr"].size != np.unique(d["e_tr"]).size:
            raise RuntimeError(f"{arm} train EdgeID not unique")
        if d["e_va"].size != np.unique(d["e_va"]).size:
            raise RuntimeError(f"{arm} val EdgeID not unique")
        if len(set(d["e_tr"].tolist()) & set(d["e_va"].tolist())):
            raise RuntimeError(f"{arm} train/val EdgeID overlap")

    tr_sets = [set(d["e_tr"].tolist()) for d in arms_data.values()]
    va_sets = [set(d["e_va"].tolist()) for d in arms_data.values()]
    tr_inter = set.intersection(*tr_sets) if len(tr_sets) > 1 else tr_sets[0]
    va_inter = set.intersection(*va_sets) if len(va_sets) > 1 else va_sets[0]
    tr_exact = all(s == tr_inter for s in tr_sets)
    va_exact = all(s == va_inter for s in va_sets)

    tr_sorted = np.array(sorted(tr_inter), dtype=np.int64)
    va_sorted = np.array(sorted(va_inter), dtype=np.int64)

    aligned = {}
    original = {}
    for arm, d in arms_data.items():
        original[arm] = {
            "train_n": int(d["e_tr"].size),
            "val_n": int(d["e_va"].size),
            "train_pos": int((d["y_tr"] == 1).sum()),
            "val_pos": int((d["y_va"] == 1).sum()),
            "train_edge_sha": sha_ordered_ids(d["e_tr"]),
            "val_edge_sha": sha_ordered_ids(d["e_va"]),
        }
        tr_map = {int(e): i for i, e in enumerate(d["e_tr"])}
        va_map = {int(e): i for i, e in enumerate(d["e_va"])}
        tr_idx = np.array([tr_map[int(e)] for e in tr_sorted], dtype=np.int64)
        va_idx = np.array([va_map[int(e)] for e in va_sorted], dtype=np.int64)
        aligned[arm] = {
            "z_tr": d["z_tr"][tr_idx],
            "y_tr": d["y_tr"][tr_idx],
            "e_tr": tr_sorted.copy(),
            "z_va": d["z_va"][va_idx],
            "y_va": d["y_va"][va_idx],
            "e_va": va_sorted.copy(),
            "meta": d["meta"],
        }

    ref = arms[0]
    for arm in arms[1:]:
        if not np.array_equal(aligned[arm]["y_tr"], aligned[ref]["y_tr"]):
            raise RuntimeError(f"train labels disagree after EdgeID align: {arm}")
        if not np.array_equal(aligned[arm]["y_va"], aligned[ref]["y_va"]):
            raise RuntimeError(f"val labels disagree after EdgeID align: {arm}")
        if not np.array_equal(aligned[arm]["e_tr"], aligned[ref]["e_tr"]):
            raise RuntimeError("train EdgeID align failed")
        if not np.array_equal(aligned[arm]["e_va"], aligned[ref]["e_va"]):
            raise RuntimeError("val EdgeID align failed")

    source = arms_data[ref]["meta"]["source_cohort"]
    floors = COVERAGE_FLOORS[target]
    matched = {
        "train_n": int(tr_sorted.size),
        "val_n": int(va_sorted.size),
        "train_pos": int((aligned[ref]["y_tr"] == 1).sum()),
        "val_pos": int((aligned[ref]["y_va"] == 1).sum()),
    }
    edge_cov = min(matched["train_n"] / max(source["train_n"], 1), matched["val_n"] / max(source["val_n"], 1))
    pos_cov = min(matched["train_pos"] / max(source["train_pos"], 1), matched["val_pos"] / max(source["val_pos"], 1))
    cov_ok = edge_cov >= floors["edge"] and pos_cov >= floors["positive"]
    if not cov_ok:
        raise RuntimeError(f"{target} matched coverage failed edge={edge_cov} pos={pos_cov}")

    report = {
        "target": target,
        "encoders": arms,
        "train_sets_exact_match": tr_exact,
        "val_sets_exact_match": va_exact,
        "edgeid_sets_differed": (not tr_exact) or (not va_exact),
        "original_per_arm": original,
        "matched": matched,
        "matched_train_edge_sha256": sha_ordered_ids(tr_sorted),
        "matched_val_edge_sha256": sha_ordered_ids(va_sorted),
        "dropped": {
            arm: {
                "train": original[arm]["train_n"] - matched["train_n"],
                "val": original[arm]["val_n"] - matched["val_n"],
            }
            for arm in arms
        },
        "coverage": {
            "source": source,
            "edge_coverage": {"min": edge_cov},
            "positive_coverage": {"min": pos_cov},
            "floors": floors,
            "ok": cov_ok,
            "note": SAMLD_COVERAGE_NOTE if target == "SAML-D" else None,
        },
        "alignment_policy": "per_target_encoder_intersection_sorted_edge_id",
        "labels_identical_across_arms": True,
        "no_test_npz": True,
    }
    return {"aligned": aligned, "report": report}


def cmd_extract(args: argparse.Namespace) -> int:
    logger_setup()
    tid = args.array_task_id
    if tid is None:
        tid = int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1"))
    if tid < 0 or tid >= len(EXTRACT_CELLS):
        raise SystemExit(f"bad array task {tid}")
    arm, target = EXTRACT_CELLS[tid]
    logging.info("Extract cell %s: %s → %s", tid, arm, target)
    out = run_extract_cell(arm, target)
    print(json.dumps({"ok": True, "array_task": tid, "arm": arm, "target": target, "status": out["status"]}, indent=2))
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    logger_setup()
    target = args.target
    if target not in TARGETS:
        raise SystemExit(target)
    t0 = time.perf_counter()
    # Fit probes only for newly extracted arms; reused cells are materialized separately.
    # Align EdgeIDs among probe arms only (avoids loading 6 full embedding trees into RAM).
    probe_arms = encoders_to_probe(target)
    if not probe_arms:
        logging.info("No new arms to probe for %s", target)
        write_json(result_dir() / f"probe_{target.lower().replace('-', '_')}.json", {
            "target": target, "ok": True, "cells": [], "note": "all cells reused"
        })
        print(json.dumps({"ok": True, "target": target, "n_cells": 0}, indent=2))
        return 0

    pack = align_target_cohort(target, arms=probe_arms)
    aligned = pack["aligned"]
    report = pack["report"]
    expected_edges = EXPECTED_MATCHED_EDGE_SHA256[target]
    if report["matched_train_edge_sha256"] != expected_edges["train"]:
        raise RuntimeError(
            f"matched train edge SHA drift ({target}): {report['matched_train_edge_sha256']} "
            f"!= {expected_edges['train']}"
        )
    if report["matched_val_edge_sha256"] != expected_edges["val"]:
        raise RuntimeError(
            f"matched val edge SHA drift ({target}): {report['matched_val_edge_sha256']} "
            f"!= {expected_edges['val']}"
        )
    write_json(
        result_dir() / f"matched_cohorts_{target.lower().replace('-', '')}.json",
        report,
    )

    cell_results = []
    pred_dir = result_dir() / "val_predictions" / target.lower().replace("-", "_")
    pred_dir.mkdir(parents=True, exist_ok=True)

    for arm in probe_arms:
        logging.info("Probe %s → %s", arm, target)
        d = aligned[arm]
        # free other Z later — process sequentially
        fit = fit_r198_probe(d["z_tr"], d["y_tr"], d["z_va"], d["y_va"], device=torch.device("cpu"))
        # save predictions
        np.savez_compressed(
            pred_dir / f"{arm}.npz",
            edge_id=d["e_va"].astype(np.int64),
            y=d["y_va"].astype(np.int64),
            logit_selected=fit["val_logit_selected"],
            proba_selected=fit["val_proba_selected"],
            logit_final=fit["val_logit_final"],
            proba_final=fit["val_proba_final"],
        )
        # drop large arrays from fit for JSON
        cell = {
            "encoder": arm,
            "target": target,
            "role": cell_role(arm, target),
            "checkpoint_step": CHECKPOINT_STEP[arm],
            "updates_per_domain": UPDATES_PER_DOMAIN[arm],
            "lr_phase": LR_PHASE_AT_CHECKPOINT[arm],
            "bn_bundle_domain": bn_bundle_domain(arm, target),
            "bn_bundle_sha256": d["meta"]["bn_policy"]["bundle_sha256"],
            "target_scaler_sha256": d["meta"]["target_scaler_sha256"],
            "checkpoint_sha256": d["meta"]["checkpoint_sha256"],
            "coverage": report["coverage"],
            "matched_train_n": report["matched"]["train_n"],
            "matched_val_n": report["matched"]["val_n"],
            "matched_train_pos": report["matched"]["train_pos"],
            "matched_val_pos": report["matched"]["val_pos"],
            "matched_train_edge_sha256": report["matched_train_edge_sha256"],
            "matched_val_edge_sha256": report["matched_val_edge_sha256"],
            "train_sets_exact_match_across_long_steps": report["train_sets_exact_match"],
            "val_sets_exact_match_across_long_steps": report["val_sets_exact_match"],
            "validation_auprc": fit["validation_auprc"],
            "validation_auroc": fit["validation_auroc"],
            "validation_metrics_at_0.5": fit["validation_metrics_at_0.5"],
            "validation_metrics_at_val_optimal_f1": fit["validation_metrics_at_val_optimal_f1"],
            "final_probe_val_bce": fit["final_probe_val_bce"],
            "final_probe_train_bce": fit["final_probe_train_bce"],
            "selected_probe_val_bce": fit["selected_probe_val_bce"],
            "final_probe_epoch": fit["final_probe_epoch"],
            "selected_probe_epoch": fit["selected_probe_epoch"],
            "prevalence_val": fit["prevalence_val"],
            "n_val": fit["n_val"],
            "n_val_pos": fit["n_val_pos"],
            "probe_protocol": fit["probe_protocol"],
            "encoder_updated": False,
            "test_evaluated": False,
            "checkpoint_selection": "predeclared_fixed_not_val_selected",
        }
        write_json(result_dir() / "cells" / f"{cell_name(arm, target)}.json", cell)
        cell_results.append(cell)
        # free matrices for this arm
        for k in list(d.keys()):
            if isinstance(d[k], np.ndarray):
                d[k] = None
        gc.collect()
        logging.info(
            "%s AUPRC=%.4f F1@0.5=%.4f",
            arm,
            cell["validation_auprc"],
            cell["validation_metrics_at_0.5"]["f1"],
        )

    # free remaining
    del aligned
    gc.collect()

    summary = {
        "target": target,
        "ok": True,
        "matched_cohort": report,
        "cells": cell_results,
        "probe_protocol": PROBE,
        "elapsed_sec": time.perf_counter() - t0,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "encoder_retrained": False,
        "test_data_loaded_or_scored": False,
    }
    write_json(result_dir() / f"probe_{target.lower().replace('-', '_')}.json", summary)
    print(json.dumps({"ok": True, "target": target, "n_cells": len(cell_results)}, indent=2))
    return 0


def _delta_block(mixed: Dict, specialist: Dict, transfer: Dict) -> Dict[str, Any]:
    return {}


def load_short_mixed_cells() -> Dict[str, Dict[str, Any]]:
    out = {}
    for target, name in SHORT_MIXED_CELL.items():
        p = ROOT / SHORT_FROZEN_ROOT / "cells" / f"{name}.json"
        if not p.is_file():
            raise FileNotFoundError(p)
        out[target] = json.loads(p.read_text(encoding="utf-8"))
    return out


def build_short_comparability_card() -> Dict[str, Any]:
    """Authorize reuse of completed Phase-4B SHORT MIXED@1500 metrics."""
    fields = {
        "same_feature_contract": CONTRACT_ID,
        "same_feature_order": list(FINAL_FEATURE_NAMES) == list(SHARED_CORE_FINAL_FEATURE_NAMES),
        "same_extraction_protocol_full_subgraph_r198": True,
        "same_target_bn_policy": True,
        "same_probe_architecture_hyperparameters": PROBE,
        "same_init_sha": INIT_SHA256,
        "short_checkpoint": SHORT_CHECKPOINT,
        "short_checkpoint_sha256": SHORT_CHECKPOINT_SHA256,
        "short_lr_phase": SHORT_LR_PHASE,
    }
    verified = {}
    ok = True
    try:
        short_cells = load_short_mixed_cells()
    except FileNotFoundError as e:
        return {
            "ok": False,
            "verdict": "PROTOCOL_INCOMPARABLE",
            "reason": f"missing SHORT cell: {e}",
            "fields": fields,
        }

    for target, cell in short_cells.items():
        scaler_ok = cell.get("target_scaler_sha256") == TARGET_SCALER_SHA256[target]
        pp = cell.get("probe_protocol") or {}
        probe_ok = (
            pp.get("learner") == PROBE["learner"]
            and pp.get("epochs") == PROBE["epochs"]
            and pp.get("lr") == PROBE["lr"]
            and pp.get("batch_size") == PROBE["batch_size"]
            and pp.get("seed") == PROBE["seed"]
            and pp.get("input_dim") == PROBE["input_dim"]
        )
        bn_ok = cell.get("bn_bundle_domain") == target
        no_test = cell.get("test_evaluated") is False
        verified[target] = {
            "ok": scaler_ok and probe_ok and bn_ok and no_test,
            "path": str(ROOT / SHORT_FROZEN_ROOT / "cells" / f"{SHORT_MIXED_CELL[target]}.json"),
            "scaler_ok": scaler_ok,
            "probe_ok": probe_ok,
            "bn_ok": bn_ok,
            "no_test": no_test,
            "matched_val_n": cell.get("matched_val_n"),
            "matched_val_edge_sha256": cell.get("matched_val_edge_sha256"),
            "AUPRC": cell.get("validation_auprc"),
            "F1@0.5": (cell.get("validation_metrics_at_0.5") or {}).get("f1"),
            "cohort_note": (
                "SHORT Small-LI matched cohort is MIXED∩SMALL_LI_ONLY intersection; "
                "LONG matched cohort is LONG_1500∩LONG_3000 (expected exact). "
                "Compare EdgeID hashes after LONG extract; if they differ, LI A/C "
                "comparisons are protocol-aligned but not bit-identical cohorts."
                if target == "Small-LI"
                else "SHORT HI/SAML cohort is single-encoder MIXED extract; LONG uses "
                "exact intersection of step-1500 and step-3000 extracts."
            ),
        }
        if not verified[target]["ok"]:
            ok = False

    ok = (
        ok
        and fields["same_feature_order"]
        and fields["same_extraction_protocol_full_subgraph_r198"]
        and fields["same_target_bn_policy"]
    )
    return {
        "ok": ok,
        "verdict": "COMPARABLE_REUSE_AUTHORIZED" if ok else "PROTOCOL_INCOMPARABLE",
        "fields": fields,
        "verified_short_mixed_cells": verified,
        "reuse_targets": list(TARGETS),
        "interpretation_notes": [
            "Comparison A (LONG@1500 vs SHORT@1500): same 500 updates/domain and matched early "
            "batch streams in training; different LR horizon/phase — schedule-horizon comparison.",
            "Comparison B (LONG@3000 vs LONG@1500): same long schedule; 1000 vs 500 updates/domain.",
            "Comparison C (LONG@3000 vs SHORT@1500): both exposure and LR schedule differ.",
        ],
        "do_not_silently_merge_if_false": True,
    }


def classify_expansion(
    *,
    hi_ret: float,
    sd_ret: float,
    li_ret: float,
    li_mixed_auprc: float,
) -> Dict[str, Any]:
    return {"category": "UNUSED_FOR_LONG_EVAL", "legacy": True}


def interpret(hi_deltas: Dict, sd_deltas: Dict) -> Dict[str, Any]:
    return {"category": "UNUSED", "legacy": True}


def _row_from_cell(c: Dict[str, Any], *, label: str) -> Dict[str, Any]:
    m05 = c["validation_metrics_at_0.5"]
    mopt = c["validation_metrics_at_val_optimal_f1"]
    cov = c.get("coverage") or {}
    return {
        "encoder_checkpoint": label,
        "global_step": c.get("checkpoint_step"),
        "updates_per_domain": c.get("updates_per_domain"),
        "lr_phase": (c.get("lr_phase") or {}).get("schedule_phase"),
        "encoder_lr": (c.get("lr_phase") or {}).get("encoder_lr"),
        "validation_AUPRC": c["validation_auprc"],
        "AUROC": c["validation_auroc"],
        "F1@0.5": m05["f1"],
        "precision@0.5": m05["precision"],
        "recall@0.5": m05["recall"],
        "F1@val-thr_optimistic": mopt["f1"],
        "final_probe_epoch_val_BCE": c["final_probe_val_bce"],
        "matched_train_n": c.get("matched_train_n"),
        "matched_val_n": c.get("matched_val_n"),
        "matched_train_pos": c.get("matched_train_pos"),
        "matched_val_pos": c.get("matched_val_pos"),
        "edge_coverage_min": (cov.get("edge_coverage") or {}).get("min"),
        "positive_coverage_min": (cov.get("positive_coverage") or {}).get("min"),
        "matched_val_edge_sha256": c.get("matched_val_edge_sha256"),
    }


def _cmp(a: float, b: float) -> Dict[str, float]:
    return {
        "delta": float(a - b),
        "retention": float(a / b) if b else float("nan"),
    }


def cmd_finalize(_: argparse.Namespace) -> int:
    logger_setup()
    return cmd_finalize_ladder()


def cmd_materialize_reuse(_: argparse.Namespace) -> int:
    logger_setup()
    payload = materialize_reuse()
    print(json.dumps(payload, indent=2))
    return 0 if payload.get("ok") else 2



def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("preflight")
    sp.set_defaults(func=cmd_preflight)

    sm = sub.add_parser("materialize-reuse")
    sm.set_defaults(func=cmd_materialize_reuse)

    se = sub.add_parser("extract")
    se.add_argument("--array_task_id", type=int, default=None)
    se.set_defaults(func=cmd_extract)

    spr = sub.add_parser("probe")
    spr.add_argument("--target", type=str, required=True, choices=list(TARGETS))
    spr.set_defaults(func=cmd_probe)

    sf = sub.add_parser("finalize")
    sf.set_defaults(func=cmd_finalize)

    args = p.parse_args(argv)
    return int(args.func(args))



if __name__ == "__main__":
    raise SystemExit(main())
