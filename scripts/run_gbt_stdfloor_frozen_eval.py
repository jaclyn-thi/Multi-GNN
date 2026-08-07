#!/usr/bin/env python3
"""GBT stdfloor full3000 frozen R198 validation-only evaluation.

Subcommands:
  preflight  — verify training integrity + checkpoints + baseline comparability
  extract    — one GPU cell: GBT step × target full-subgraph R198 train/val
  probe      — one CPU job per target: PaperStyleMLP on GBT@1500 + GBT@3000
  finalize   — tables vs objective-ablation + ADAPTIVE LONG@3000 baselines

Never retrains encoders. Never loads/scores test. Never uses recovery checkpoints.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import shutil
import subprocess
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
from graph_barlow_twins_r198 import (  # noqa: E402
    FULL3000_CKPT_ROOT,
    RECOVERY_STDFLOOR_CKPT_ROOT,
)
from mixed_ssl_phase4b import CANONICAL_DOMAINS  # noqa: E402
from mixed_ssl_phase2.bn import (  # noqa: E402
    apply_bn_,
    bn_bundle_l1,
    clone_bn_bundle,
    collect_bn_bundle,
)
from phase4b_frozen_eval.probe import fit_r198_probe  # noqa: E402
from gbt_stdfloor_frozen_eval import (  # noqa: E402
    BASELINE_ARMS,
    BASELINE_LONG_ROOT,
    BASELINE_OBJABL_ROOT,
    CKPT_ROOT,
    CELLS,
    CHECKPOINT_SHA256,
    CHECKPOINT_SHA256_PATH,
    CHECKPOINT_STEP,
    CONTRACT_ID,
    COVERAGE_FLOORS,
    EDGE_DIM,
    EMB_ROOT,
    FINAL_FEATURE_NAMES,
    GBT_STD_FLOOR,
    INIT_SHA256,
    MIN_DISK_HEADROOM_GIB,
    NOTES_PATH,
    OBJECTIVE_ID,
    OLD_CONTRACT_ID,
    PROBE,
    R198_DIM,
    RESULT_ROOT,
    SAML_SPLIT_PROTOCOL,
    SAMLD_COVERAGE_NOTE,
    SEED,
    SOURCE_COHORT,
    TARGET_SCALER_SHA256,
    TARGETS,
    TRAIN_ARMS,
    TRAIN_RESULT_ROOT,
    TWIN_JSON,
    UPDATES_PER_DOMAIN,
    baseline_cell_path,
    bn_bundle_domain,
    cell_name,
    cell_role,
    checkpoint_path,
    encoders_for_target,
)
from shared_core_contract import SHARED_CORE_FINAL_FEATURE_NAMES  # noqa: E402
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
    return ROOT / EMB_ROOT / cell_name(arm, target) / "pre_embedding_3h"


def result_dir() -> Path:
    return ROOT / RESULT_ROOT


def load_checkpoint_sha_registry() -> Dict[str, str]:
    reg = dict(CHECKPOINT_SHA256)
    p = ROOT / CHECKPOINT_SHA256_PATH
    if p.is_file():
        blob = json.loads(p.read_text(encoding="utf-8"))
        for arm in TRAIN_ARMS:
            if blob.get(arm):
                reg[arm] = str(blob[arm])
    return reg


def persist_checkpoint_sha(arm: str, sha: str) -> None:
    reg = load_checkpoint_sha_registry()
    if not reg.get(arm):
        reg[arm] = sha
        write_json(ROOT / CHECKPOINT_SHA256_PATH, reg)


def assert_allowed_checkpoint_path(path: Path) -> None:
    resolved = str(path.resolve())
    lowered = resolved.lower()
    if "recovery" in lowered:
        raise RuntimeError(f"refusing recovery checkpoint tree: {path}")
    forbidden_roots = (
        str((ROOT / RECOVERY_STDFLOOR_CKPT_ROOT).resolve()),
        str((ROOT / FULL3000_CKPT_ROOT).resolve()),
    )
    for forbidden in forbidden_roots:
        if resolved.startswith(forbidden):
            raise RuntimeError(f"refusing forbidden checkpoint root: {path}")


def validate_npz(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"ok": False, "reason": "missing"}
    try:
        d = np.load(path)
        z = np.asarray(d["Z"])
        y = np.asarray(d["y"]).reshape(-1)
        eid = np.asarray(d["edge_id"]).reshape(-1)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": str(e)}
    if z.ndim != 2 or z.shape[1] != R198_DIM:
        return {"ok": False, "reason": f"bad dim {z.shape}"}
    if not np.isfinite(z).all():
        return {"ok": False, "reason": "nonfinite"}
    if eid.size != np.unique(eid).size:
        return {"ok": False, "reason": "dup edge_id"}
    return {
        "ok": True,
        "n": int(z.shape[0]),
        "n_pos": int((y == 1).sum()),
        "dim": R198_DIM,
        "edge_id_sha256": sha_ordered_ids(eid),
        "y_sha256": hashlib.sha256(y.astype(np.int64).tobytes()).hexdigest(),
    }


def model_state_finite(blob: Dict[str, Any]) -> bool:
    msd = blob.get("model_state_dict") or {}
    for v in msd.values():
        if isinstance(v, torch.Tensor) and not torch.isfinite(v).all():
            return False
    return bool(msd)


def verify_gbt_checkpoint(arm: str) -> Dict[str, Any]:
    rel = checkpoint_path(arm)
    p = ROOT / rel
    assert_allowed_checkpoint_path(p)
    if not p.is_file():
        return {"ok": False, "arm": arm, "reason": "missing_checkpoint", "path": str(p)}

    sha = sha256_file(p)
    persist_checkpoint_sha(arm, sha)
    sha_reg = load_checkpoint_sha_registry()
    blob = torch.load(p, map_location="cpu", weights_only=False)
    recipe = blob.get("recipe") or {}
    extra = blob.get("extra") or {}
    step_counts = extra.get("step_counts") or blob.get("step_counts") or {}
    expected_step = CHECKPOINT_STEP[arm]
    expected_upd = UPDATES_PER_DOMAIN[arm]

    bn_bundles = blob.get("bn_bundles") or {}
    bn_info = {
        d: {"n_keys": len(b), "sha256": bn_bundle_sha(b)}
        for d, b in bn_bundles.items()
    }

    gates = {
        "sha_match_if_set": (not sha_reg.get(arm)) or (sha == sha_reg[arm]),
        "objective_id": blob.get("objective_id") == OBJECTIVE_ID,
        "recipe_objective_id": recipe.get("objective_id") == OBJECTIVE_ID,
        "gbt_std_floor": float(recipe.get("gbt_std_floor") or 0) == GBT_STD_FLOOR,
        "contract_id": recipe.get("contract_id") == CONTRACT_ID,
        "edge_dim": int(recipe.get("edge_dim", -1)) == EDGE_DIM,
        "representation_dim": int(recipe.get("representation_dim", -1)) == R198_DIM,
        "model_state_finite": model_state_finite(blob),
        "has_model": "model_state_dict" in blob,
        "has_bn_three_domains": all(d in bn_bundles for d in CANONICAL_DOMAINS),
        "global_step_expected": int(blob.get("global_step", blob.get("global_optimizer_step", -1)))
        == expected_step,
        "updates_per_domain_expected": all(
            int(step_counts.get(d, -1)) == expected_upd for d in CANONICAL_DOMAINS
        ),
        "preserve_seed_false": recipe.get("preserve_seed_edges") is False,
        "projection_false": recipe.get("contrast_projection_head") is False,
        "infonce_disabled": not bool(recipe.get("infonce_enabled")),
        "tfmoe_disabled": not bool(recipe.get("tfmoe_enabled")),
        "not_recovery_scout": not bool(recipe.get("is_recovery_scout")),
        "not_failed_official_full3000": recipe.get("ckpt_root") != FULL3000_CKPT_ROOT,
    }
    try:
        _ = {k: v.shape for k, v in blob["model_state_dict"].items()}
        gates["model_state_reloadable"] = True
    except Exception:
        gates["model_state_reloadable"] = False

    resume_meta = extra.get("resume_exact_verified")
    ok = all(gates.values())
    return {
        "ok": ok,
        "arm": arm,
        "path": str(p),
        "sha256": sha,
        "expected_sha256": sha_reg.get(arm) or sha,
        "gates": gates,
        "bn_bundles": bn_info,
        "step_counts": step_counts,
        "resume_exact_verified": resume_meta,
        "resume_exact_note": (
            "resume_exact_verified=false is acceptable for frozen eval"
            if resume_meta is False
            else None
        ),
    }


def verify_training_aggregate() -> Dict[str, Any]:
    agg_path = ROOT / TRAIN_RESULT_ROOT / "aggregate.json"
    if not agg_path.is_file():
        return {"ok": False, "reason": "missing aggregate.json", "path": str(agg_path)}
    agg = json.loads(agg_path.read_text(encoding="utf-8"))
    gates = agg.get("gates") or {}
    ok = bool(agg.get("ok"))
    if not ok and isinstance(gates, dict) and gates:
        ok = all(bool(v) for v in gates.values())
    if not ok and str(agg.get("classification", "")).upper() == "PASS":
        ok = True
    seed_path = ROOT / TRAIN_RESULT_ROOT / "seed_stream_vs_long.json"
    seed_blob = json.loads(seed_path.read_text(encoding="utf-8")) if seed_path.is_file() else {}
    seed_ok = {}
    if seed_blob:
        for d in CANONICAL_DOMAINS:
            seed_ok[d] = bool((seed_blob.get(d) or {}).get("ok"))
        if seed_ok and not all(seed_ok.values()):
            ok = False
    init_ok = True
    init_prov = agg.get("init_provenance") or {}
    init_sha = init_prov.get("init_sha256") or agg.get("init_sha256")
    if init_sha and init_sha != INIT_SHA256:
        init_ok = False
        ok = False
    return {
        "ok": ok,
        "path": str(agg_path),
        "classification": agg.get("classification"),
        "gates": gates,
        "init_sha256": init_sha,
        "init_sha256_expected": INIT_SHA256,
        "init_ok": init_ok,
        "seed_stream_vs_long": seed_blob if seed_blob else None,
        "seed_stream_ok": seed_ok if seed_ok else None,
    }


def verify_disk_headroom() -> Dict[str, Any]:
    emb_root = ROOT / EMB_ROOT
    check_path = emb_root if emb_root.exists() else emb_root.parent
    usage = shutil.disk_usage(check_path)
    free_gib = float(usage.free) / (1024**3)
    return {
        "ok": free_gib >= MIN_DISK_HEADROOM_GIB,
        "path_checked": str(check_path),
        "free_gib": free_gib,
        "required_gib": MIN_DISK_HEADROOM_GIB,
    }


def verify_cli_guards() -> Dict[str, Any]:
    from extract_direct_r198_full_cell import parse_extract_splits, refuse_seed_only_path  # noqa: E402

    seed_only_refused = False
    try:
        refuse_seed_only_path()
    except RuntimeError:
        seed_only_refused = True

    test_refused = False
    try:
        parse_extract_splits("train,val,test")
    except SystemExit:
        test_refused = True

    help_cmds = [
        [sys.executable, str(ROOT / "scripts/run_gbt_stdfloor_frozen_eval.py"), "--help"],
        [sys.executable, str(ROOT / "scripts/run_gbt_stdfloor_frozen_eval.py"), "preflight", "--help"],
        [sys.executable, str(ROOT / "scripts/run_gbt_stdfloor_frozen_eval.py"), "extract", "--help"],
        [sys.executable, str(ROOT / "scripts/run_gbt_stdfloor_frozen_eval.py"), "probe", "--help"],
        [sys.executable, str(ROOT / "scripts/run_gbt_stdfloor_frozen_eval.py"), "finalize", "--help"],
    ]
    help_ok = {}
    for cmd in help_cmds:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=str(ROOT))
        help_ok[" ".join(cmd[-2:])] = proc.returncode == 0

    ok = seed_only_refused and test_refused and all(help_ok.values())
    return {
        "ok": ok,
        "seed_only_refused": seed_only_refused,
        "test_splits_refused": test_refused,
        "subcommand_help_ok": help_ok,
    }


def load_baseline_cell(baseline_label: str, target: str) -> Dict[str, Any]:
    rel = baseline_cell_path(baseline_label, target)
    p = ROOT / rel
    if not p.is_file():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def build_baseline_comparability_card() -> Dict[str, Any]:
    fields = {
        "same_feature_contract": CONTRACT_ID,
        "same_feature_order": list(FINAL_FEATURE_NAMES) == list(SHARED_CORE_FINAL_FEATURE_NAMES),
        "same_extraction_protocol_full_subgraph_r198": True,
        "same_target_bn_policy": True,
        "same_probe_architecture_hyperparameters": PROBE,
        "same_init_sha": INIT_SHA256,
        "objective_id": OBJECTIVE_ID,
        "gbt_std_floor": GBT_STD_FLOOR,
        "no_reextract_baselines": True,
    }
    verified = {}
    ok = True
    for label in BASELINE_ARMS:
        verified[label] = {}
        for target in TARGETS:
            try:
                cell = load_baseline_cell(label, target)
            except FileNotFoundError as e:
                return {
                    "ok": False,
                    "verdict": "BASELINE_INCOMPARABLE",
                    "reason": str(e),
                    "fields": fields,
                }
            pp = cell.get("probe_protocol") or {}
            probe_ok = (
                pp.get("learner") == PROBE["learner"]
                and pp.get("epochs") == PROBE["epochs"]
                and pp.get("lr") == PROBE["lr"]
                and pp.get("batch_size") == PROBE["batch_size"]
                and pp.get("seed") == PROBE["seed"]
                and pp.get("input_dim") == PROBE["input_dim"]
            )
            no_test = cell.get("test_evaluated") is False
            entry_ok = probe_ok and no_test
            verified[label][target] = {
                "ok": entry_ok,
                "path": str(ROOT / baseline_cell_path(label, target)),
                "probe_ok": probe_ok,
                "no_test": no_test,
                "validation_auprc": cell.get("validation_auprc"),
            }
            if not entry_ok:
                ok = False
    ok = ok and fields["same_feature_order"]
    return {
        "ok": ok,
        "verdict": "COMPARABLE_REUSE_AUTHORIZED" if ok else "BASELINE_INCOMPARABLE",
        "fields": fields,
        "verified_baselines": verified,
        "baseline_roots": {
            "objective_ablation": BASELINE_OBJABL_ROOT,
            "adaptive_long": BASELINE_LONG_ROOT,
        },
    }


def make_extract_args(target: str, cell: str) -> argparse.Namespace:
    argv = [
        "--data",
        target,
        "--model",
        "gin",
        "--objective",
        "contrastive",
        "--unique_name",
        f"gbt_stdfloor_frozen_{cell}",
        "--seed",
        str(SEED),
        "--batch_size",
        "8192",
        "--num_neighs",
        "100",
        "100",
        "--loader_num_workers",
        "0",
        "--reverse_mp",
        "--ego",
        "--ports",
        "--emlps",
        "--tds",
        "--correct_reverse_edge_features",
        "--feature_contract",
        CONTRACT_ID,
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
    t0 = time.perf_counter()
    result_dir().mkdir(parents=True, exist_ok=True)
    (result_dir() / "cells").mkdir(parents=True, exist_ok=True)
    (ROOT / EMB_ROOT).mkdir(parents=True, exist_ok=True)

    train_agg = verify_training_aggregate()
    ck = {"ok": True, "arms": {}}
    for arm in TRAIN_ARMS:
        info = verify_gbt_checkpoint(arm)
        ck["arms"][arm] = info
        if not info.get("ok"):
            ck["ok"] = False

    bn_ok = True
    bn_checks = {}
    for arm, target in CELLS:
        info = ck["arms"].get(arm, {})
        bn_dom = bn_bundle_domain(arm, target)
        present = bn_dom in (info.get("bn_bundles") or {})
        bn_checks[cell_name(arm, target)] = {
            "bn_bundle_domain": bn_dom,
            "bn_present": present,
            "target_scaler_sha256_expected": TARGET_SCALER_SHA256[target],
        }
        if not present:
            bn_ok = False
            ck["ok"] = False

    comparability = build_baseline_comparability_card()
    write_json(result_dir() / "baseline_comparability_card.json", comparability)

    disk = verify_disk_headroom()
    cli = verify_cli_guards()

    cells = []
    sha_reg = load_checkpoint_sha_registry()
    for i, (arm, target) in enumerate(CELLS):
        cells.append(
            {
                "array_index": i,
                "arm": arm,
                "target": target,
                "objective_id": OBJECTIVE_ID,
                "gbt_std_floor": GBT_STD_FLOOR,
                "role": cell_role(arm, target),
                "bn_bundle_domain": bn_bundle_domain(arm, target),
                "target_scaler_sha256_expected": TARGET_SCALER_SHA256[target],
                "checkpoint": checkpoint_path(arm),
                "checkpoint_sha256": sha_reg.get(arm),
                "checkpoint_step": CHECKPOINT_STEP[arm],
                "updates_per_domain": UPDATES_PER_DOMAIN[arm],
                "embeddings_dir": str(emb_dir(arm, target)),
            }
        )

    feature_order_ok = list(FINAL_FEATURE_NAMES) == list(SHARED_CORE_FINAL_FEATURE_NAMES)
    report = {
        "ok": bool(
            train_agg.get("ok")
            and ck.get("ok")
            and feature_order_ok
            and comparability.get("ok")
            and bn_ok
            and disk.get("ok")
            and cli.get("ok")
        ),
        "phase": "gbt_stdfloor_full3000_frozen_eval",
        "objective_id": OBJECTIVE_ID,
        "gbt_std_floor": GBT_STD_FLOOR,
        "feature_contract_id": CONTRACT_ID,
        "old_contract_geometry_equivalent": OLD_CONTRACT_ID,
        "final_feature_names": list(FINAL_FEATURE_NAMES),
        "feature_order_matches_shared_core": feature_order_ok,
        "saml_split_protocol": SAML_SPLIT_PROTOCOL,
        "edge_dim": EDGE_DIM,
        "r198_dim": R198_DIM,
        "projection_at_extract": False,
        "preserve_seed_edges": False,
        "init_sha256": INIT_SHA256,
        "training_aggregate": train_agg,
        "baseline_comparability": comparability,
        "checkpoints": ck,
        "bn_checks": bn_checks,
        "disk_headroom": disk,
        "cli_guards": cli,
        "cells": cells,
        "cell_array_mapping": {
            "0-2": "GBT_STDFLOOR_1500 × (Small-HI, SAML-D, Small-LI)",
            "3-5": "GBT_STDFLOOR_3000 × targets",
        },
        "coverage_floors": COVERAGE_FLOORS,
        "samld_coverage_note": SAMLD_COVERAGE_NOTE,
        "probe_audit": dict(PROBE),
        "source_cohort_protocol_card": SOURCE_COHORT,
        "never_use_recovery_checkpoints": True,
        "never_use_failed_official_full3000": True,
        "no_encoder_retrain": True,
        "no_test_eval": True,
        "no_test_npz_allowed": True,
        "elapsed_sec": time.perf_counter() - t0,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(result_dir() / "preflight.json", report)
    print(
        json.dumps(
            {
                "ok": report["ok"],
                "n_cells": len(cells),
                "comparability": comparability.get("verdict"),
                "disk_free_gib": disk.get("free_gib"),
            },
            indent=2,
        )
    )
    if not report["ok"]:
        logging.error("Preflight FAILED — do not submit")
        return 2
    logging.info("Preflight OK")
    return 0


def run_extract_cell(arm: str, target: str) -> Dict[str, Any]:
    info = verify_gbt_checkpoint(arm)
    if not info.get("ok"):
        raise RuntimeError(f"checkpoint gate failed for {arm}: {info}")

    cell = cell_name(arm, target)
    out_dir = emb_dir(arm, target)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "test.npz").is_file():
        raise RuntimeError(f"test.npz present: {out_dir}")

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
    assert_allowed_checkpoint_path(ckpt_p)
    sha = sha256_file(ckpt_p)
    persist_checkpoint_sha(arm, sha)
    sha_reg = load_checkpoint_sha_registry()
    if sha_reg.get(arm) and sha != sha_reg[arm]:
        raise RuntimeError(f"checkpoint sha mismatch {sha} != {sha_reg[arm]}")
    blob = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    recipe = blob.get("recipe") or {}
    if recipe.get("contract_id") != CONTRACT_ID:
        raise RuntimeError(f"bad contract in checkpoint recipe: {recipe.get('contract_id')}")
    if blob.get("objective_id") != OBJECTIVE_ID:
        raise RuntimeError(f"bad objective_id: {blob.get('objective_id')}")
    if float(recipe.get("gbt_std_floor") or 0) != GBT_STD_FLOOR:
        raise RuntimeError("gbt_std_floor mismatch")

    bn_dom = bn_bundle_domain(arm, target)
    if bn_dom not in blob["bn_bundles"]:
        raise RuntimeError(f"BN bundle {bn_dom} missing from {arm} checkpoint")
    bn_sel = clone_bn_bundle(blob["bn_bundles"][bn_dom])
    bn_sha = bn_bundle_sha(bn_sel)

    with open(ROOT / "data_config.json", encoding="utf-8") as f:
        data_config = json.load(f)

    args = make_extract_args(target, cell)
    set_seed(SEED)
    logging.info("Loading target graph %s under %s (arm=%s)", target, CONTRACT_ID, arm)
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
    args.contrast_projection_head = False
    model = get_model(sample_batch, config, args)
    emb_dim = int(getattr(model, "embedding_dim", R198_DIM))
    model = to_hetero(model, tr_data.metadata(), aggr="mean")
    model.bypass_embedding_head = True

    model.load_state_dict(blob["model_state_dict"], strict=True)
    apply_bn_(model, bn_sel)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm)):
            m.eval()

    bn_before_extract = clone_bn_bundle(collect_bn_bundle(model))

    tr_loader, val_loader, te_loader = get_loaders(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args, train_shuffle=False
    )
    del te_loader

    staging = out_dir / f".staging_{os.getpid()}"
    staging.mkdir(parents=True, exist_ok=True)

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

    bn_after_extract = collect_bn_bundle(model)
    bn_unchanged = bn_bundle_l1(bn_before_extract, bn_after_extract) == 0.0

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

    extracted = {}
    for s in ("train", "val"):
        st = validate_npz(out_dir / f"{s}.npz")
        extracted[s] = {"n": st["n"], "n_pos": st["n_pos"], "edge_id_sha256": st["edge_id_sha256"]}
    floors = COVERAGE_FLOORS[target]
    edge_cov = min(
        extracted["train"]["n"] / max(source["train_n"], 1),
        extracted["val"]["n"] / max(source["val_n"], 1),
    )
    pos_cov = min(
        extracted["train"]["n_pos"] / max(source["train_pos"], 1),
        extracted["val"]["n_pos"] / max(source["val_pos"], 1),
    )
    cov_ok = edge_cov >= floors["edge"] and pos_cov >= floors["positive"]

    meta = {
        "cell": cell,
        "encoder_arm": arm,
        "target_dataset": target,
        "objective_id": OBJECTIVE_ID,
        "gbt_std_floor": GBT_STD_FLOOR,
        "role": cell_role(arm, target),
        "feature_contract_id": CONTRACT_ID,
        "edge_dim": EDGE_DIM,
        "r198_dim": R198_DIM,
        "checkpoint_path": str(ckpt_p),
        "checkpoint_sha256": sha,
        "checkpoint_step": CHECKPOINT_STEP[arm],
        "updates_per_domain": UPDATES_PER_DOMAIN[arm],
        "bn_policy": {
            "bundle_domain": bn_dom,
            "bundle_sha256": bn_sha,
            "mixed_uses_target_domain_bn": True,
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
        "projection_bypassed": True,
        "encoder_requires_grad": False,
        "model_eval": True,
        "test_evaluated": False,
        "test_npz_present": False,
        "skip_test_eval": True,
        "extractor": "run_gbt_stdfloor_frozen_eval.run_extract_cell",
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


def cmd_extract(args: argparse.Namespace) -> int:
    logger_setup()
    tid = args.array_task_id
    if tid is None:
        tid = int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1"))
    if tid < 0 or tid >= len(CELLS):
        raise SystemExit(f"bad array task {tid}")
    arm, target = CELLS[tid]
    logging.info("Extract cell %s: %s → %s", tid, arm, target)
    out = run_extract_cell(arm, target)
    print(json.dumps({"ok": True, "array_task": tid, "arm": arm, "target": target, "status": out["status"]}, indent=2))
    return 0


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


def align_target_cohort(target: str) -> Dict[str, Any]:
    arms = list(encoders_for_target(target))
    arms_data = {}
    for arm in arms:
        ztr, ytr, etr, zva, yva, eva, meta = load_split_arrays(arm, target)
        if (emb_dir(arm, target) / "test.npz").is_file():
            raise RuntimeError(f"test.npz in {arm}/{target}")
        arms_data[arm] = {
            "z_tr": ztr,
            "y_tr": ytr,
            "e_tr": etr,
            "z_va": zva,
            "y_va": yva,
            "e_va": eva,
            "meta": meta,
        }

    tr_sets = [set(d["e_tr"].tolist()) for d in arms_data.values()]
    va_sets = [set(d["e_va"].tolist()) for d in arms_data.values()]
    tr_inter = set.intersection(*tr_sets)
    va_inter = set.intersection(*va_sets)
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
            raise RuntimeError(f"train labels disagree: {arm}")
        if not np.array_equal(aligned[arm]["y_va"], aligned[ref]["y_va"]):
            raise RuntimeError(f"val labels disagree: {arm}")

    source = arms_data[ref]["meta"]["source_cohort"]
    floors = COVERAGE_FLOORS[target]
    matched = {
        "train_n": int(tr_sorted.size),
        "val_n": int(va_sorted.size),
        "train_pos": int((aligned[ref]["y_tr"] == 1).sum()),
        "val_pos": int((aligned[ref]["y_va"] == 1).sum()),
    }
    edge_cov = min(
        matched["train_n"] / max(source["train_n"], 1),
        matched["val_n"] / max(source["val_n"], 1),
    )
    pos_cov = min(
        matched["train_pos"] / max(source["train_pos"], 1),
        matched["val_pos"] / max(source["val_pos"], 1),
    )
    cov_ok = edge_cov >= floors["edge"] and pos_cov >= floors["positive"]
    if not cov_ok:
        raise RuntimeError(f"{target} matched coverage failed edge={edge_cov} pos={pos_cov}")

    report = {
        "target": target,
        "encoders": arms,
        "train_sets_exact_match": tr_exact,
        "val_sets_exact_match": va_exact,
        "original_per_arm": original,
        "matched": matched,
        "matched_train_edge_sha256": sha_ordered_ids(tr_sorted),
        "matched_val_edge_sha256": sha_ordered_ids(va_sorted),
        "coverage": {
            "source": source,
            "edge_coverage": {"min": edge_cov},
            "positive_coverage": {"min": pos_cov},
            "floors": floors,
            "ok": cov_ok,
        },
        "no_test_npz": True,
    }
    return {"aligned": aligned, "report": report}


def cmd_probe(args: argparse.Namespace) -> int:
    logger_setup()
    target = args.target
    if target not in TARGETS:
        raise SystemExit(target)
    t0 = time.perf_counter()
    pack = align_target_cohort(target)
    aligned = pack["aligned"]
    report = pack["report"]
    write_json(result_dir() / f"matched_cohorts_{target.lower().replace('-', '')}.json", report)

    cell_results = []
    pred_dir = result_dir() / "val_predictions" / target.lower().replace("-", "_")
    pred_dir.mkdir(parents=True, exist_ok=True)

    for arm in encoders_for_target(target):
        logging.info("Probe %s → %s", arm, target)
        d = aligned[arm]
        fit = fit_r198_probe(d["z_tr"], d["y_tr"], d["z_va"], d["y_va"], device=torch.device("cpu"))
        np.savez_compressed(
            pred_dir / f"{arm}.npz",
            edge_id=d["e_va"].astype(np.int64),
            y=d["y_va"].astype(np.int64),
            logit_selected=fit["val_logit_selected"],
            proba_selected=fit["val_proba_selected"],
            logit_final=fit["val_logit_final"],
            proba_final=fit["val_proba_final"],
        )
        m05 = fit["validation_metrics_at_0.5"]
        mopt = fit["validation_metrics_at_val_optimal_f1"]
        cell = {
            "encoder": arm,
            "target": target,
            "objective_id": OBJECTIVE_ID,
            "gbt_std_floor": GBT_STD_FLOOR,
            "role": cell_role(arm, target),
            "checkpoint_step": CHECKPOINT_STEP[arm],
            "updates_per_domain": UPDATES_PER_DOMAIN[arm],
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
            "train_sets_exact_match_across_arms": report["train_sets_exact_match"],
            "val_sets_exact_match_across_arms": report["val_sets_exact_match"],
            "validation_auprc": fit["validation_auprc"],
            "validation_auroc": fit["validation_auroc"],
            "validation_metrics_at_0.5": m05,
            "validation_metrics_at_val_optimal_f1": mopt,
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
        }
        write_json(result_dir() / "cells" / f"{cell_name(arm, target)}.json", cell)
        cell_results.append(cell)
        for k in list(d.keys()):
            if isinstance(d[k], np.ndarray):
                d[k] = None
        gc.collect()

    del aligned
    gc.collect()

    summary = {
        "target": target,
        "ok": True,
        "matched_cohort": report,
        "cells": cell_results,
        "probe_protocol": PROBE,
        "elapsed_sec": time.perf_counter() - t0,
        "encoder_retrained": False,
        "test_data_loaded_or_scored": False,
    }
    write_json(result_dir() / f"probe_{target.lower().replace('-', '_')}.json", summary)
    print(json.dumps({"ok": True, "target": target, "n_cells": len(cell_results)}, indent=2))
    return 0


def _row_from_cell(c: Dict[str, Any]) -> Dict[str, Any]:
    m05 = c["validation_metrics_at_0.5"]
    mopt = c["validation_metrics_at_val_optimal_f1"]
    return {
        "Arm": c["encoder"],
        "Objective": OBJECTIVE_ID,
        "Step": c.get("checkpoint_step"),
        "Target": c["target"],
        "AUPRC": c["validation_auprc"],
        "AUROC": c["validation_auroc"],
        "F1@0.5": m05["f1"],
        "P": m05["precision"],
        "R": m05["recall"],
        "F1@val-thr": mopt["f1"],
        "F1@val-thr_optimistic": bool(mopt.get("optimistic_diagnostic")),
        "Final val BCE": c["final_probe_val_bce"],
    }


def _row_from_baseline(cell: Dict[str, Any], label: str) -> Dict[str, Any]:
    m05 = cell["validation_metrics_at_0.5"]
    mopt = cell["validation_metrics_at_val_optimal_f1"]
    return {
        "Arm": label,
        "Objective": cell.get("objective") or label,
        "Step": cell.get("checkpoint_step", 3000),
        "Target": cell["target"],
        "AUPRC": cell["validation_auprc"],
        "AUROC": cell["validation_auroc"],
        "F1@0.5": m05["f1"],
        "P": m05["precision"],
        "R": m05["recall"],
        "F1@val-thr": mopt["f1"],
        "F1@val-thr_optimistic": bool(mopt.get("optimistic_diagnostic")),
        "Final val BCE": cell["final_probe_val_bce"],
        "reused_baseline": True,
    }


def cmd_finalize(_: argparse.Namespace) -> int:
    logger_setup()
    cells = []
    for arm, target in CELLS:
        p = result_dir() / "cells" / f"{cell_name(arm, target)}.json"
        if not p.is_file():
            raise SystemExit(f"missing cell result {p}")
        cells.append(json.loads(p.read_text(encoding="utf-8")))
    by = {(c["encoder"], c["target"]): c for c in cells}

    card_path = result_dir() / "baseline_comparability_card.json"
    comparability = (
        json.loads(card_path.read_text(encoding="utf-8"))
        if card_path.is_file()
        else build_baseline_comparability_card()
    )

    baselines = {}
    for label in BASELINE_ARMS:
        baselines[label] = {t: load_baseline_cell(label, t) for t in TARGETS}

    table_rows = []
    delta_rows = []
    for arm, target in CELLS:
        c = by[(arm, target)]
        table_rows.append(_row_from_cell(c))
        for label, bcells in baselines.items():
            ref = bcells[target]
            delta_rows.append(
                {
                    "GBT_arm": arm,
                    "Target": target,
                    "GBT_AUPRC": c["validation_auprc"],
                    "Baseline": label,
                    "Baseline_AUPRC": ref["validation_auprc"],
                    "delta_AUPRC": c["validation_auprc"] - ref["validation_auprc"],
                    "retention_AUPRC": c["validation_auprc"] / ref["validation_auprc"]
                    if ref["validation_auprc"]
                    else float("nan"),
                }
            )

    for label in BASELINE_ARMS:
        for target in TARGETS:
            table_rows.append(_row_from_baseline(baselines[label][target], label))

    gbt_step_delta = []
    for target in TARGETS:
        c1500 = by[("GBT_STDFLOOR_1500", target)]
        c3000 = by[("GBT_STDFLOOR_3000", target)]
        gbt_step_delta.append(
            {
                "Target": target,
                "AUPRC_1500": c1500["validation_auprc"],
                "AUPRC_3000": c3000["validation_auprc"],
                "delta_AUPRC_3000_minus_1500": c3000["validation_auprc"] - c1500["validation_auprc"],
            }
        )

    write_json(result_dir() / "cells" / "gbt_stdfloor_table.json", table_rows)
    write_json(result_dir() / "cells" / "deltas_vs_baselines.json", delta_rows)
    write_json(result_dir() / "cells" / "gbt_1500_vs_3000.json", gbt_step_delta)

    payload = {
        "ok": True,
        "phase": "gbt_stdfloor_full3000_frozen_eval",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective_id": OBJECTIVE_ID,
        "gbt_std_floor": GBT_STD_FLOOR,
        "scientific_question": (
            "Does GBT stdfloor full3000 at steps 1500/3000 match or beat adaptive "
            "LONG and objective-ablation baselines under frozen R198 validation protocol?"
        ),
        "baseline_comparability": comparability,
        "probe_protocol": PROBE,
        "table": table_rows,
        "deltas_vs_baselines": delta_rows,
        "gbt_1500_vs_3000": gbt_step_delta,
        "baselines_reused_not_reextracted": True,
        "test_data_loaded_or_scored": False,
        "encoder_retrained": False,
        "no_recovery_checkpoint_eval": True,
    }
    write_json(result_dir() / "aggregate.json", payload)
    write_json(ROOT / TWIN_JSON, payload)

    lines = [
        "# GBT stdfloor full3000 frozen R198 validation eval",
        "",
        f"> Twin: `{TWIN_JSON}`",
        f"> Objective: `{OBJECTIVE_ID}` (gbt_std_floor={GBT_STD_FLOOR})",
        "",
        f"**ok={payload['ok']}** — validation-only; no encoder retrain; no test.",
        "",
        f"Baseline comparability: `{comparability.get('verdict')}`",
        "",
        "## Protocol locks",
        "",
        f"- contract: `{CONTRACT_ID}`",
        f"- probe: PaperStyleMLP {PROBE['epochs']}ep lr={PROBE['lr']} bs={PROBE['batch_size']} seed={PROBE['seed']}",
        "- extract: full-subgraph R198 train/val; projection bypassed",
        f"- checkpoints: `{CKPT_ROOT}` steps 1500 + 3000 only",
        "",
        "## Main table",
        "",
        "| Arm | Step | Target | AUPRC | AUROC | F1@0.5 | P | R | F1@val-thr | Final val BCE |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in table_rows:
        if r.get("reused_baseline"):
            continue
        lines.append(
            f"| {r['Arm']} | {r['Step']} | {r['Target']} | "
            f"{float(r['AUPRC']):.4f} | {float(r['AUROC']):.4f} | {float(r['F1@0.5']):.4f} | "
            f"{float(r['P']):.4f} | {float(r['R']):.4f} | {float(r['F1@val-thr']):.4f} | "
            f"{float(r['Final val BCE']):.6f} |"
        )

    lines += ["", "## GBT@3000 vs GBT@1500", ""]
    for d in gbt_step_delta:
        lines.append(
            f"- **{d['Target']}**: ΔAUPRC={d['delta_AUPRC_3000_minus_1500']:+.4f} "
            f"({d['AUPRC_1500']:.4f} → {d['AUPRC_3000']:.4f})"
        )

    lines += ["", "## Deltas vs reused baselines (AUPRC)", ""]
    for d in delta_rows:
        lines.append(
            f"- **{d['GBT_arm']} / {d['Target']} vs {d['Baseline']}**: "
            f"Δ={d['delta_AUPRC']:+.4f} (retention {d['retention_AUPRC']:.3f})"
        )

    lines += [
        "",
        "Confirmation: no test data loaded/scored; baselines not re-extracted; "
        "recovery / failed official full3000 checkpoints never evaluated.",
        "",
    ]
    notes = ROOT / NOTES_PATH
    notes.parent.mkdir(parents=True, exist_ok=True)
    notes.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "notes": str(notes), "n_rows": len(table_rows)}, indent=2))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("preflight")
    sp.set_defaults(func=cmd_preflight)

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
