#!/usr/bin/env python3
"""GBT+TF fixed-half @1500 + EXPERT_ONLY@1500 frozen R198 validation-only eval.

Subcommands:
  preflight  — verify training integrity + dual-arm checkpoints + comparability
  extract    — one GPU cell: fixed-half .pt or EXPERT .tar × target train/val R198
  probe      — one CPU job per target: PaperStyleMLP on both new encoders
  finalize   — matched-step-1500 tables, deltas, figures, Q1–Q12, notes

Never retrains encoders. Never loads/scores test. Never uses recovery checkpoints.
Uses direct encoder R198 (projection bypassed). Final authorized GBT experiment.
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
from gbt_tf_fixed_half_stdfloor_frozen_eval import (  # noqa: E402
    BASELINE_ARMS,
    BASELINE_GBT_STDFLOOR_ROOT,
    BASELINE_GBT_TF_ADAPTIVE_ROOT,
    BASELINE_LONG_ROOT,
    BASELINE_MATCH_KIND,
    BASELINE_OBJABL_ROOT,
    CELLS,
    CHECKPOINT_SHA256,
    CHECKPOINT_SHA256_PATH,
    CHECKPOINT_STEP,
    CKPT_ROOT,
    CONTRACT_ID,
    COVERAGE_FLOORS,
    EDGE_DIM,
    EMB_ROOT,
    EXPERT_CKPT_ROOT,
    EXPERT_ONLY_ARM,
    EXPERT_TRAIN_RESULT_ROOT,
    FINAL_FEATURE_NAMES,
    FIXED_HALF_ARM,
    GBT_STD_FLOOR,
    INIT_SHA256,
    MIN_DISK_HEADROOM_GIB,
    NOTES_PATH,
    OBJECTIVE_ID,
    OLD_CONTRACT_ID,
    PRIMARY_MATCHED_1500_LABELS,
    PROBE,
    R198_DIM,
    RESULT_ROOT,
    SAML_SPLIT_PROTOCOL,
    SAMLD_COVERAGE_NOTE,
    SEED,
    SOURCE_COHORT,
    TARGET_SCALER_SHA256,
    TARGETS,
    TF_TARGET_NAMES,
    TRAIN_ARMS,
    TRAIN_RESULT_ROOT,
    TWIN_JSON,
    UPDATES_PER_DOMAIN,
    arm_kind,
    baseline_cell_path,
    bn_bundle_domain,
    cell_name,
    cell_role,
    checkpoint_path,
    encoders_for_target,
    expert_integrity_path,
)
from direct_r198 import LearnedAlphaBeta  # noqa: E402
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

SCRIPT_REL = "scripts/run_gbt_tf_fixed_half_stdfloor_frozen_eval.py"

ALPHA_INTERPRETATION_NOTE = (
    "beta_unfrozen_after_step_15: inherited training field alpha_unfrozen_at=15 "
    "means beta became eligible for updates after step 15; alpha remained fixed "
    "at 0.5 and never unfroze. Do not rewrite historical training artifacts."
)

REQUIRED_EXTRACT_ATTRS = (
    "data",
    "feature_contract",
    "skip_test_eval",
    "unique_name",
    "seed",
    "extract_splits",
    "representation_source",
    "embeddings_dir",
    "embeddings_subdir",
    "contrast_projection_head",
    "preserve_seed_edges",
    "embedding_dim",
    "model",
    "batch_size",
    "num_neighs",
    "reverse_mp",
    "ego",
    "ports",
    "emlps",
    "tds",
    "correct_reverse_edge_features",
    "train_fit_edge_znorm",
    "direct_r198_infonce",
)


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


def _alpha_policy_is_fixed(policy: Any) -> bool:
    s = str(policy or "").lower()
    return "fixed" in s


def verify_fixed_half_checkpoint() -> Dict[str, Any]:
    arm = FIXED_HALF_ARM
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
    locked_sha = CHECKPOINT_SHA256.get(arm)
    path_under_ckpt_root = str(p.resolve()).startswith(str((ROOT / CKPT_ROOT).resolve()))

    gates = {
        "path_under_ckpt_root": path_under_ckpt_root,
        "sha_locked_match": locked_sha is not None and sha == locked_sha,
        "sha_registry_match": (not sha_reg.get(arm)) or (sha == sha_reg[arm]),
        "objective_id": blob.get("objective_id") == OBJECTIVE_ID,
        "recipe_objective_id": recipe.get("objective_id") == OBJECTIVE_ID,
        "gbt_std_floor": float(recipe.get("gbt_std_floor") or 0) == GBT_STD_FLOOR,
        "contract_id": recipe.get("contract_id") == CONTRACT_ID,
        "edge_dim": int(recipe.get("edge_dim", -1)) == EDGE_DIM,
        "representation_dim": int(recipe.get("representation_dim", -1)) == R198_DIM,
        "weight_mode_fixed_half": recipe.get("weight_mode") == "fixed_half",
        "learn_alpha_false": recipe.get("learn_alpha") is False,
        "fixed_alpha_0_5": float(recipe.get("fixed_alpha", -1)) == 0.5,
        "fixed_w_gbt_0_5": float(recipe.get("fixed_w_gbt", -1)) == 0.5,
        "fixed_w_tf_mass_0_5": float(recipe.get("fixed_w_tf_mass", -1)) == 0.5,
        "alpha_policy_fixed": _alpha_policy_is_fixed(recipe.get("alpha_policy")),
        "schedule_horizon_3000": int(recipe.get("schedule_horizon", -1)) == 3000,
        "warmup_600": int(recipe.get("warmup_steps", -1)) == 600,
        "linear_decay_2400": int(recipe.get("linear_decay_steps", -1)) == 2400,
        "executed_stop_1500": int(recipe.get("executed_stop_step", -1)) == 1500,
        "model_state_finite": model_state_finite(blob),
        "has_model": "model_state_dict" in blob,
        "has_alpha_beta_state": "alpha_beta_state_dict" in blob,
        "has_bn_three_domains": all(d in bn_bundles for d in CANONICAL_DOMAINS),
        "global_step_expected": int(blob.get("global_step", blob.get("global_optimizer_step", -1)))
        == expected_step,
        "updates_per_domain_expected": all(
            int(step_counts.get(d, -1)) == expected_upd for d in CANONICAL_DOMAINS
        ),
        "preserve_seed_false": recipe.get("preserve_seed_edges") is False,
        "projection_false": recipe.get("contrast_projection_head") is False,
        "infonce_disabled": not bool(recipe.get("infonce_enabled")),
        "tfmoe_enabled": bool(recipe.get("tfmoe_enabled")),
        "alpha_beta_enabled": bool(recipe.get("alpha_beta_enabled")),
        "not_recovery_scout": not bool(recipe.get("is_recovery_scout")),
        "not_failed_official_full3000": recipe.get("ckpt_root") != FULL3000_CKPT_ROOT,
        "tf_target_names": list(recipe.get("tf_target_names") or []) == list(TF_TARGET_NAMES),
    }
    try:
        _ = {k: v.shape for k, v in blob["model_state_dict"].items()}
        gates["model_state_reloadable"] = True
    except Exception:
        gates["model_state_reloadable"] = False

    inherited_alpha_unfrozen = (
        extra.get("alpha_unfrozen_at")
        if extra.get("alpha_unfrozen_at") is not None
        else blob.get("alpha_unfrozen_at")
    )
    ok = all(gates.values())
    return {
        "ok": ok,
        "arm": arm,
        "arm_kind": "fixed_half",
        "path": str(p),
        "sha256": sha,
        "expected_sha256": sha_reg.get(arm) or sha,
        "gates": gates,
        "bn_bundles": bn_info,
        "step_counts": step_counts,
        "recipe_weight_mode": recipe.get("weight_mode"),
        "inherited_alpha_unfrozen_at": inherited_alpha_unfrozen,
        "beta_unfrozen_after_step_15": True,
        "alpha_interpretation_note": ALPHA_INTERPRETATION_NOTE,
        "do_not_rewrite_training_artifacts": True,
    }


def _expert_tf_targets_from_blob(blob: Dict[str, Any]) -> Optional[List[str]]:
    resolved = blob.get("resolved") or {}
    for key in ("tf_target_names", "tf_targets", "moe_targets"):
        v = resolved.get(key) or blob.get(key)
        if isinstance(v, (list, tuple)) and v:
            return [str(x) for x in v]
    pre = blob.get("preflight") or {}
    for key in ("moe_targets", "tf_target_names"):
        v = pre.get(key)
        if isinstance(v, (list, tuple)) and v:
            return [str(x) for x in v]
    domains = (pre.get("domain_registry") or {}).get("domains") or {}
    orders = []
    for d in CANONICAL_DOMAINS:
        order = (domains.get(d) or {}).get("expected_tf_target_order")
        if order:
            orders.append([str(x) for x in order])
    if orders and all(o == orders[0] for o in orders):
        return orders[0]
    return None


def verify_expert_integrity() -> Dict[str, Any]:
    ip = ROOT / expert_integrity_path()
    if not ip.is_file():
        return {"ok": False, "reason": "missing_integrity", "path": str(ip)}
    blob = json.loads(ip.read_text(encoding="utf-8"))
    gates = blob.get("gates") or {}
    matching = blob.get("matching_vs_long") or {}
    match_ok = {}
    first500_ok = {}
    for d in CANONICAL_DOMAINS:
        m = matching.get(d) or {}
        match_ok[d] = bool(m.get("ok"))
        n_compared = int(m.get("n_compared") or 0)
        # n_compared=1000 ok ⇒ first 500 seed hashes/domain match LONG
        first500_ok[d] = bool(m.get("ok")) and n_compared >= 500
    contrast_gate = bool(gates.get("contrast_grad_contribution_always_false"))
    ok = (
        bool(blob.get("ok"))
        and bool(gates.get("ok"))
        and all(match_ok.values())
        and all(first500_ok.values())
        and contrast_gate
    )
    return {
        "ok": ok,
        "path": str(ip),
        "gates": gates,
        "init_sha256": blob.get("init_sha256"),
        "matching_vs_long_ok": match_ok,
        "first_500_seed_hashes_ok": first500_ok,
        "contrast_grad_contribution_always_false": contrast_gate,
    }


def verify_expert_checkpoint() -> Dict[str, Any]:
    arm = EXPERT_ONLY_ARM
    rel = checkpoint_path(arm)
    p = ROOT / rel
    if not p.is_file():
        return {"ok": False, "arm": arm, "reason": "missing_checkpoint", "path": str(p)}

    integ = verify_expert_integrity()
    sha = sha256_file(p)
    persist_checkpoint_sha(arm, sha)
    sha_reg = load_checkpoint_sha_registry()
    blob = torch.load(p, map_location="cpu", weights_only=False)
    resolved = blob.get("resolved") or {}
    expected_step = CHECKPOINT_STEP[arm]
    expected_upd = UPDATES_PER_DOMAIN[arm]
    exposures = blob.get("per_domain_exposure_counts") or resolved.get("per_domain_exposure_counts") or {}
    bn_bundles = blob.get("bn_bundles") or {}
    bn_info = {
        d: {"n_keys": len(b), "sha256": bn_bundle_sha(b)}
        for d, b in bn_bundles.items()
    }
    edge_scalers = {
        d: (blob.get("edge_scalers") or {}).get(d, {}).get("scaler_sha256")
        for d in CANONICAL_DOMAINS
    }
    locked_sha = CHECKPOINT_SHA256.get(arm)
    path_under_expert_root = str(p.resolve()).startswith(
        str((ROOT / EXPERT_CKPT_ROOT).resolve())
    )

    ab = LearnedAlphaBeta(n_tf=3)
    if "alpha_beta_state_dict" in blob:
        ab.load_state_dict(blob["alpha_beta_state_dict"])
    with torch.no_grad():
        eff = ab.effective_weights("expert_only")
    w_contrast = float(eff["w_contrast"])
    sum_w_tf = float(eff["sum_w_tf"])

    tf_names = _expert_tf_targets_from_blob(blob)
    tf_ok = list(tf_names or []) == list(TF_TARGET_NAMES)

    gates = {
        "integrity_ok": bool(integ.get("ok")),
        "path_under_expert_root": path_under_expert_root,
        "sha_locked_match": locked_sha is not None and sha == locked_sha,
        "sha_registry_match": (not sha_reg.get(arm)) or (sha == sha_reg[arm]),
        "feature_contract_id": blob.get("feature_contract_id") == CONTRACT_ID,
        "init_sha256": blob.get("init_sha256") == INIT_SHA256,
        "weight_mode_expert_only": resolved.get("weight_mode") == "expert_only",
        "learn_alpha_false": resolved.get("learn_alpha") is False,
        "learn_beta_true": resolved.get("learn_beta") is True,
        "w_contrast_effectively_0": abs(w_contrast) < 1e-8,
        "sum_w_tf_effectively_1": abs(sum_w_tf - 1.0) < 1e-5,
        "global_optimizer_step_1500": int(blob.get("global_optimizer_step", -1)) == expected_step,
        "per_domain_exposure_500": all(
            int(exposures.get(d, -1)) == expected_upd for d in CANONICAL_DOMAINS
        ),
        "preserve_seed_false": resolved.get("preserve_seed_edges") is False,
        "projection_false": resolved.get("contrast_projection_head") is False,
        "warmup_600": int(resolved.get("warmup_steps", -1)) == 600,
        "linear_decay_2400": int(resolved.get("linear_decay_steps", -1)) == 2400,
        "max_optimizer_steps_3000": int(resolved.get("max_optimizer_steps", -1)) == 3000,
        "has_model": "model_state_dict" in blob,
        "has_bn_three_domains": all(d in bn_bundles for d in CANONICAL_DOMAINS),
        "edge_scalers_match": all(
            edge_scalers.get(d) == TARGET_SCALER_SHA256[d] for d in CANONICAL_DOMAINS
        ),
        "model_state_finite": model_state_finite(blob),
        "tf_target_names": tf_ok,
        "contrast_grad_contribution_always_false": bool(
            integ.get("contrast_grad_contribution_always_false")
        ),
        "first_500_seed_hashes_ok": all(
            (integ.get("first_500_seed_hashes_ok") or {}).values()
        ),
        "ignores_projection_state_at_extract": True,
    }
    try:
        _ = {k: v.shape for k, v in blob["model_state_dict"].items()}
        gates["model_state_reloadable"] = True
    except Exception:
        gates["model_state_reloadable"] = False

    ok = all(gates.values())
    return {
        "ok": ok,
        "arm": arm,
        "arm_kind": "expert_only",
        "path": str(p),
        "sha256": sha,
        "expected_sha256": sha_reg.get(arm) or sha,
        "gates": gates,
        "bn_bundles": bn_info,
        "edge_scalers": edge_scalers,
        "exposures": exposures,
        "integrity": integ,
        "tf_target_names": tf_names,
        "effective_weights_expert_only": {
            "w_contrast": w_contrast,
            "sum_w_tf": sum_w_tf,
            "w_tf": [float(eff[f"w_tf_{i}"]) for i in range(3)],
        },
    }


def verify_training_aggregate() -> Dict[str, Any]:
    """Authorize via plain aggregate PASS + seed_stream_vs_long (no adaptive C_shape)."""
    agg_path = ROOT / TRAIN_RESULT_ROOT / "aggregate.json"
    if not agg_path.is_file():
        return {"ok": False, "reason": "missing_aggregate", "path": str(agg_path)}
    agg = json.loads(agg_path.read_text(encoding="utf-8"))
    seed_path = ROOT / TRAIN_RESULT_ROOT / "seed_stream_vs_long.json"
    seed_blob = json.loads(seed_path.read_text(encoding="utf-8")) if seed_path.is_file() else {}
    if not seed_blob:
        seed_blob = agg.get("seed_stream_vs_long") or {}

    classification = agg.get("classification")
    pass_ok = bool(agg.get("ok")) and classification == "PASS"
    seed_ok = {}
    for d in CANONICAL_DOMAINS:
        seed_ok[d] = bool((seed_blob.get(d) or {}).get("ok"))
    seed_all_ok = bool(seed_ok) and all(seed_ok.values())

    init_prov = agg.get("shared_init_provenance") or agg.get("init_provenance") or {}
    init_sha = init_prov.get("init_sha256") or agg.get("init_sha256")
    init_ok = init_sha == INIT_SHA256

    ok = pass_ok and seed_all_ok and init_ok
    return {
        "ok": ok,
        "path": str(agg_path),
        "classification": classification,
        "aggregate_ok": bool(agg.get("ok")),
        "authorized_for_frozen_eval": ok,
        "acceptance_path": "plain_aggregate_PASS",
        "gates": agg.get("training_integrity_gates") or agg.get("gates") or {},
        "init_sha256": init_sha,
        "init_sha256_expected": INIT_SHA256,
        "init_ok": init_ok,
        "seed_stream_vs_long": seed_blob if seed_blob else None,
        "seed_stream_ok": seed_ok if seed_ok else None,
        "inherited_alpha_unfrozen_at": agg.get("alpha_unfrozen_at"),
        "beta_unfrozen_after_step_15": True,
        "alpha_interpretation_note": ALPHA_INTERPRETATION_NOTE,
        "no_adaptive_c_shape_revalidation": True,
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
        [sys.executable, str(ROOT / SCRIPT_REL), "--help"],
        [sys.executable, str(ROOT / SCRIPT_REL), "preflight", "--help"],
        [sys.executable, str(ROOT / SCRIPT_REL), "extract", "--help"],
        [sys.executable, str(ROOT / SCRIPT_REL), "probe", "--help"],
        [sys.executable, str(ROOT / SCRIPT_REL), "finalize", "--help"],
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
        "script": SCRIPT_REL,
    }


def verify_extract_args_construction() -> Dict[str, Any]:
    """Instantiate make_extract_args for every cell and assert required CLI fields."""
    cells = {}
    ok = True
    for arm, target in CELLS:
        cell = cell_name(arm, target)
        try:
            ns = make_extract_args(target, cell)
        except Exception as e:  # noqa: BLE001
            cells[cell] = {"ok": False, "error": str(e)}
            ok = False
            continue
        missing = [a for a in REQUIRED_EXTRACT_ATTRS if not hasattr(ns, a)]
        bad = {}
        if getattr(ns, "data", None) != target:
            bad["data"] = getattr(ns, "data", None)
        if getattr(ns, "feature_contract", None) != CONTRACT_ID:
            bad["feature_contract"] = getattr(ns, "feature_contract", None)
        if not bool(getattr(ns, "skip_test_eval", False)):
            bad["skip_test_eval"] = getattr(ns, "skip_test_eval", None)
        if getattr(ns, "extract_splits", None) != "train,val":
            bad["extract_splits"] = getattr(ns, "extract_splits", None)
        if getattr(ns, "representation_source", None) != "pre_embedding_3h":
            bad["representation_source"] = getattr(ns, "representation_source", None)
        if getattr(ns, "contrast_projection_head", True) is not False:
            bad["contrast_projection_head"] = getattr(ns, "contrast_projection_head", None)
        if getattr(ns, "preserve_seed_edges", True) is not False:
            bad["preserve_seed_edges"] = getattr(ns, "preserve_seed_edges", None)
        if int(getattr(ns, "embedding_dim", -1)) != R198_DIM:
            bad["embedding_dim"] = getattr(ns, "embedding_dim", None)
        if not str(getattr(ns, "unique_name", "")).startswith("gbt_tf_fh_frozen_"):
            bad["unique_name"] = getattr(ns, "unique_name", None)
        entry_ok = not missing and not bad
        cells[cell] = {
            "ok": entry_ok,
            "arm": arm,
            "target": target,
            "missing_attrs": missing,
            "bad_values": bad,
            "unique_name": getattr(ns, "unique_name", None),
            "embeddings_subdir": getattr(ns, "embeddings_subdir", None),
        }
        if not entry_ok:
            ok = False
    return {"ok": ok, "cells": cells, "required_attrs": list(REQUIRED_EXTRACT_ATTRS)}


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
        "objective_id_fixed_half": OBJECTIVE_ID,
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
                "match_kind": BASELINE_MATCH_KIND.get(label),
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
            "gbt_stdfloor": BASELINE_GBT_STDFLOOR_ROOT,
            "gbt_tf_adaptive": BASELINE_GBT_TF_ADAPTIVE_ROOT,
        },
        "match_kinds": dict(BASELINE_MATCH_KIND),
        "primary_matched_1500_labels": list(PRIMARY_MATCHED_1500_LABELS),
    }


def build_matched_arm_comparability(
    fh: Dict[str, Any],
    expert: Dict[str, Any],
    train_agg: Dict[str, Any],
) -> Dict[str, Any]:
    """Both new step-1500 checkpoints must share protocol locks."""
    fields = {
        "same_INIT_SHA256_phase3": INIT_SHA256,
        "financial_multidataset_shared_core_v1": CONTRACT_ID,
        "edge_dim": EDGE_DIM,
        "R198": R198_DIM,
        "round_robin_3_domains": list(CANONICAL_DOMAINS),
        "global_step": 1500,
        "updates_per_domain": 500,
        "same_3000_step_LR_prefix": {
            "warmup": 600,
            "linear_decay": 2400,
            "not_rescaled_to_1500": True,
        },
        "projection_bypassed_at_extract": True,
        "no_test": True,
        "aug_graph_bn_scaler_extraction_policy_locks": True,
    }
    checks = {
        "fixed_half_ok": bool(fh.get("ok")),
        "expert_ok": bool(expert.get("ok")),
        "fixed_half_sha_reload": bool((fh.get("gates") or {}).get("sha_locked_match"))
        and bool((fh.get("gates") or {}).get("model_state_reloadable")),
        "expert_sha_reload": bool((expert.get("gates") or {}).get("sha_locked_match"))
        and bool((expert.get("gates") or {}).get("model_state_reloadable")),
        "both_init_sha": bool(train_agg.get("init_ok"))
        and bool((expert.get("gates") or {}).get("init_sha256")),
        "both_contract": bool((fh.get("gates") or {}).get("contract_id"))
        and bool((expert.get("gates") or {}).get("feature_contract_id")),
        "both_edge_dim_r198": bool((fh.get("gates") or {}).get("edge_dim"))
        and bool((fh.get("gates") or {}).get("representation_dim")),
        "both_step_1500_500_per_domain": bool((fh.get("gates") or {}).get("global_step_expected"))
        and bool((fh.get("gates") or {}).get("updates_per_domain_expected"))
        and bool((expert.get("gates") or {}).get("global_optimizer_step_1500"))
        and bool((expert.get("gates") or {}).get("per_domain_exposure_500")),
        "same_lr_prefix": bool((fh.get("gates") or {}).get("schedule_horizon_3000"))
        and bool((fh.get("gates") or {}).get("warmup_600"))
        and bool((fh.get("gates") or {}).get("linear_decay_2400"))
        and bool((expert.get("gates") or {}).get("warmup_600"))
        and bool((expert.get("gates") or {}).get("linear_decay_2400"))
        and bool((expert.get("gates") or {}).get("max_optimizer_steps_3000")),
        "bn_three_domains_both": bool((fh.get("gates") or {}).get("has_bn_three_domains"))
        and bool((expert.get("gates") or {}).get("has_bn_three_domains")),
        "projection_false_both": bool((fh.get("gates") or {}).get("projection_false"))
        and bool((expert.get("gates") or {}).get("projection_false")),
        "fixed_half_seed_stream_ok": bool(train_agg.get("ok")),
        "expert_first_500_seeds_ok": bool(
            (expert.get("gates") or {}).get("first_500_seed_hashes_ok")
        ),
    }
    ok = all(checks.values())
    return {
        "ok": ok,
        "verdict": "MATCHED_ARM_COMPARABLE" if ok else "MATCHED_ARM_INCOMPARABLE",
        "fields": fields,
        "checks": checks,
        "alpha_interpretation_note": ALPHA_INTERPRETATION_NOTE,
        "fixed_half_sha256": fh.get("sha256"),
        "expert_sha256": expert.get("sha256"),
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
        f"gbt_tf_fh_frozen_{cell}",
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
    (result_dir() / "figures").mkdir(parents=True, exist_ok=True)
    (ROOT / EMB_ROOT).mkdir(parents=True, exist_ok=True)

    train_agg = verify_training_aggregate()
    fh = verify_fixed_half_checkpoint()
    expert = verify_expert_checkpoint()
    ck = {
        "ok": bool(fh.get("ok")) and bool(expert.get("ok")),
        "arms": {FIXED_HALF_ARM: fh, EXPERT_ONLY_ARM: expert},
    }

    matched = build_matched_arm_comparability(fh, expert, train_agg)
    write_json(result_dir() / "matched_arm_comparability.json", matched)

    bn_ok = True
    bn_checks = {}
    for arm, target in CELLS:
        info = ck["arms"].get(arm, {})
        bn_dom = bn_bundle_domain(arm, target)
        present = bn_dom in (info.get("bn_bundles") or {})
        scaler_ok = True
        if arm_kind(arm) == "expert_only":
            scaler_ok = (info.get("edge_scalers") or {}).get(target) == TARGET_SCALER_SHA256[target]
        bn_checks[cell_name(arm, target)] = {
            "bn_bundle_domain": bn_dom,
            "bn_present": present,
            "scaler_sha_match": scaler_ok,
            "target_scaler_sha256_expected": TARGET_SCALER_SHA256[target],
            "arm_kind": arm_kind(arm),
        }
        if not present or not scaler_ok:
            bn_ok = False
            ck["ok"] = False

    comparability = build_baseline_comparability_card()
    write_json(result_dir() / "baseline_comparability_card.json", comparability)

    disk = verify_disk_headroom()
    cli = verify_cli_guards()
    extract_args = verify_extract_args_construction()

    cells = []
    sha_reg = load_checkpoint_sha_registry()
    for i, (arm, target) in enumerate(CELLS):
        cells.append(
            {
                "array_index": i,
                "arm": arm,
                "arm_kind": arm_kind(arm),
                "target": target,
                "objective_id": OBJECTIVE_ID if arm_kind(arm) == "fixed_half" else "expert_only",
                "gbt_std_floor": GBT_STD_FLOOR if arm_kind(arm) == "fixed_half" else None,
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
            and matched.get("ok")
            and feature_order_ok
            and comparability.get("ok")
            and bn_ok
            and disk.get("ok")
            and cli.get("ok")
            and extract_args.get("ok")
        ),
        "phase": "gbt_tf_fixed_half_stdfloor_1e4_frozen_eval",
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
        "matched_arm_comparability": matched,
        "baseline_comparability": comparability,
        "checkpoints": ck,
        "bn_checks": bn_checks,
        "disk_headroom": disk,
        "cli_guards": cli,
        "extract_args_construction": extract_args,
        "alpha_interpretation_note": ALPHA_INTERPRETATION_NOTE,
        "cells": cells,
        "cell_array_mapping": {
            "0-2": "GBT_TF_FIXED_HALF_1500 × (Small-HI, SAML-D, Small-LI)",
            "3-5": "EXPERT_ONLY_1500 × (Small-HI, SAML-D, Small-LI)",
        },
        "coverage_floors": COVERAGE_FLOORS,
        "samld_coverage_note": SAMLD_COVERAGE_NOTE,
        "probe_audit": dict(PROBE),
        "source_cohort_protocol_card": SOURCE_COHORT,
        "tf_target_names": list(TF_TARGET_NAMES),
        "never_use_recovery_checkpoints": True,
        "no_encoder_retrain": True,
        "no_test_eval": True,
        "no_test_npz_allowed": True,
        "final_authorized_gbt_experiment": True,
        "elapsed_sec": time.perf_counter() - t0,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(result_dir() / "preflight.json", report)
    print(
        json.dumps(
            {
                "ok": report["ok"],
                "n_cells": len(cells),
                "matched_arm": matched.get("verdict"),
                "comparability": comparability.get("verdict"),
                "disk_free_gib": disk.get("free_gib"),
                "extract_args_ok": extract_args.get("ok"),
            },
            indent=2,
        )
    )
    if not report["ok"]:
        logging.error("Preflight FAILED — do not submit")
        return 2
    logging.info("Preflight OK")
    return 0


def _load_checkpoint_for_extract(arm: str) -> Tuple[Path, str, Dict[str, Any], Any]:
    """Return (path, sha, blob, bn_sel_fn context). Branch on arm_kind."""
    kind = arm_kind(arm)
    ckpt_p = ROOT / checkpoint_path(arm)
    if kind == "fixed_half":
        assert_allowed_checkpoint_path(ckpt_p)
        info = verify_fixed_half_checkpoint()
        if not info.get("ok"):
            raise RuntimeError(f"checkpoint gate failed for {arm}: {info}")
    else:
        info = verify_expert_checkpoint()
        if not info.get("ok"):
            raise RuntimeError(f"checkpoint gate failed for {arm}: {info}")

    sha = sha256_file(ckpt_p)
    persist_checkpoint_sha(arm, sha)
    sha_reg = load_checkpoint_sha_registry()
    if sha_reg.get(arm) and sha != sha_reg[arm]:
        raise RuntimeError(f"checkpoint sha mismatch {sha} != {sha_reg[arm]}")
    blob = torch.load(ckpt_p, map_location="cpu", weights_only=False)

    if kind == "fixed_half":
        recipe = blob.get("recipe") or {}
        if recipe.get("contract_id") != CONTRACT_ID:
            raise RuntimeError(f"bad contract in checkpoint recipe: {recipe.get('contract_id')}")
        if blob.get("objective_id") != OBJECTIVE_ID:
            raise RuntimeError(f"bad objective_id: {blob.get('objective_id')}")
        if float(recipe.get("gbt_std_floor") or 0) != GBT_STD_FLOOR:
            raise RuntimeError("gbt_std_floor mismatch")
        if recipe.get("weight_mode") != "fixed_half":
            raise RuntimeError("expected weight_mode=fixed_half")
        if recipe.get("learn_alpha") is not False:
            raise RuntimeError("expected learn_alpha=False")
        if not bool(recipe.get("tfmoe_enabled")):
            raise RuntimeError("expected tfmoe_enabled=True")
        if not bool(recipe.get("alpha_beta_enabled")):
            raise RuntimeError("expected alpha_beta_enabled=True")
        if bool(recipe.get("infonce_enabled")):
            raise RuntimeError("infonce must be disabled for fixed-half arm")
    else:
        if blob.get("feature_contract_id") != CONTRACT_ID:
            raise RuntimeError("bad contract in expert checkpoint")
        if blob.get("init_sha256") != INIT_SHA256:
            raise RuntimeError("bad init_sha256 in expert checkpoint")
        resolved = blob.get("resolved") or {}
        if resolved.get("weight_mode") != "expert_only":
            raise RuntimeError("expected weight_mode=expert_only")

    return ckpt_p, sha, blob, info


def run_extract_cell(arm: str, target: str) -> Dict[str, Any]:
    kind = arm_kind(arm)
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

    ckpt_p, sha, blob, _info = _load_checkpoint_for_extract(arm)

    bn_dom = bn_bundle_domain(arm, target)
    if bn_dom not in blob["bn_bundles"]:
        raise RuntimeError(f"BN bundle {bn_dom} missing from {arm} checkpoint")
    bn_sel = clone_bn_bundle(blob["bn_bundles"][bn_dom])
    bn_sha = bn_bundle_sha(bn_sel)

    with open(ROOT / "data_config.json", encoding="utf-8") as f:
        data_config = json.load(f)

    args = make_extract_args(target, cell)
    set_seed(SEED)
    logging.info("Loading target graph %s under %s (arm=%s kind=%s)", target, CONTRACT_ID, arm, kind)
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

    # Load encoder weights only (ignore projection_state_dict / moe / alpha_beta).
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
        "arm_kind": kind,
        "target_dataset": target,
        "objective_id": OBJECTIVE_ID if kind == "fixed_half" else "expert_only",
        "gbt_std_floor": GBT_STD_FLOOR if kind == "fixed_half" else None,
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
        "extractor": "run_gbt_tf_fixed_half_stdfloor_frozen_eval.run_extract_cell",
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
    print(
        json.dumps(
            {
                "ok": True,
                "array_task": tid,
                "arm": arm,
                "arm_kind": arm_kind(arm),
                "target": target,
                "status": out["status"],
            },
            indent=2,
        )
    )
    return 0


def load_split_arrays(
    arm: str, target: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
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
        kind = arm_kind(arm)
        cell = {
            "encoder": arm,
            "arm_kind": kind,
            "target": target,
            "objective_id": OBJECTIVE_ID if kind == "fixed_half" else "expert_only",
            "gbt_std_floor": GBT_STD_FLOOR if kind == "fixed_half" else None,
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
            "F1_at_val_threshold_optimistic_label": True,
            "F1_at_val_threshold_note": (
                "F1@val-thr is an optimistic diagnostic (threshold selected on the "
                "same validation set); not a test estimate."
            ),
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
        "F1_at_val_threshold_optimistic": True,
    }
    write_json(result_dir() / f"probe_{target.lower().replace('-', '_')}.json", summary)
    print(json.dumps({"ok": True, "target": target, "n_cells": len(cell_results)}, indent=2))
    return 0


def _row_from_cell(c: Dict[str, Any]) -> Dict[str, Any]:
    m05 = c["validation_metrics_at_0.5"]
    mopt = c["validation_metrics_at_val_optimal_f1"]
    return {
        "Arm": c["encoder"],
        "Objective": c.get("objective_id") or OBJECTIVE_ID,
        "Step": c.get("checkpoint_step"),
        "Target": c["target"],
        "AUPRC": c["validation_auprc"],
        "AUROC": c["validation_auroc"],
        "F1@0.5": m05["f1"],
        "P": m05["precision"],
        "R": m05["recall"],
        "F1@val-thr": mopt["f1"],
        "F1@val-thr_optimistic": True,
        "Final val BCE": c["final_probe_val_bce"],
        "match_kind": "matched_step_1500",
        "reused_baseline": False,
    }


def _row_from_baseline(cell: Dict[str, Any], label: str) -> Dict[str, Any]:
    m05 = cell["validation_metrics_at_0.5"]
    mopt = cell["validation_metrics_at_val_optimal_f1"]
    return {
        "Arm": label,
        "Objective": cell.get("objective") or label,
        "Step": cell.get("checkpoint_step", 3000 if "@3000" in label else 1500),
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
        "match_kind": BASELINE_MATCH_KIND.get(label),
    }


def _alpha_beta_from_ckpt(arm: str, weight_mode: str) -> Dict[str, Any]:
    p = ROOT / checkpoint_path(arm)
    blob = torch.load(p, map_location="cpu", weights_only=False)
    ab = LearnedAlphaBeta(n_tf=3)
    ab.load_state_dict(blob["alpha_beta_state_dict"])
    with torch.no_grad():
        alpha, beta = ab.forward()
        eff = ab.effective_weights(weight_mode)
    return {
        "arm": arm,
        "weight_mode": weight_mode,
        "checkpoint_step": CHECKPOINT_STEP[arm],
        "alpha": float(alpha),
        "beta": [float(x) for x in beta],
        "w_gbt": float(eff["w_contrast"]),
        "w_tf": [float(eff[f"w_tf_{i}"]) for i in range(3)],
        "sum_w_tf": float(eff["sum_w_tf"]),
        "sum_weights": float(eff["sum_weights"]),
        "tf_target_names": list(TF_TARGET_NAMES),
        "inherited_alpha_unfrozen_at": (blob.get("extra") or {}).get("alpha_unfrozen_at")
        or blob.get("alpha_unfrozen_at"),
        "beta_unfrozen_after_step_15": True,
        "alpha_interpretation_note": ALPHA_INTERPRETATION_NOTE,
    }


def _step_row_at(step: int) -> Optional[Dict[str, Any]]:
    path = ROOT / TRAIN_RESULT_ROOT / "logs" / "steps.jsonl"
    if not path.is_file():
        return None
    hit = None
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            s = int(
                row.get(
                    "global_optimizer_step",
                    row.get("step", row.get("global_step", -1)),
                )
            )
            if s == step:
                hit = row
    return hit


def _realized_shares_from_step(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not row:
        return {"ok": False}
    alpha = float(row.get("alpha", 0.5))
    betas = [float(row[f"beta_{i}"]) for i in range(3)]
    w_gbt = float(row.get("w_gbt", row.get("w_contrast", 0.5)))
    w_tf = [float(row.get(f"w_tf_{i}", 0.5 * betas[i])) for i in range(3)]
    return {
        "ok": True,
        "alpha": alpha,
        "beta": betas,
        "w_gbt": w_gbt,
        "w_tf": w_tf,
        "share_gbt": w_gbt,
        "share_tf": w_tf,
        "weighted_gbt": float(row.get("weighted_gbt", row.get("weighted_contrast"))),
        "weighted_tf": [float(row.get(f"weighted_tf_{i}")) for i in range(3)],
        "L_gbt_norm": float(row.get("L_gbt_norm", row.get("L_contrast_norm"))),
        "L_tf_norm": [float(row.get(f"L_tf_norm_{i}")) for i in range(3)],
        "step": int(
            row.get("global_optimizer_step", row.get("step", row.get("global_step", -1)))
        ),
        "weight_mode": row.get("weight_mode"),
    }


def _milestone_1500() -> Optional[Dict[str, Any]]:
    agg_path = ROOT / TRAIN_RESULT_ROOT / "aggregate.json"
    if not agg_path.is_file():
        return None
    agg = json.loads(agg_path.read_text(encoding="utf-8"))
    ms = agg.get("milestones") or {}
    return ms.get("1500") or ms.get(1500)


def _auprc(by: Dict[Tuple[str, str], Dict[str, Any]], arm: str, target: str) -> float:
    return float(by[(arm, target)]["validation_auprc"])


def _baseline_auprc(
    baselines: Dict[str, Dict[str, Dict[str, Any]]], label: str, target: str
) -> float:
    return float(baselines[label][target]["validation_auprc"])


def _metric_lookup(
    by: Dict[Tuple[str, str], Dict[str, Any]],
    baselines: Dict[str, Dict[str, Dict[str, Any]]],
    label: str,
    target: str,
) -> float:
    if label in (FIXED_HALF_ARM, EXPERT_ONLY_ARM):
        return _auprc(by, label, target)
    return _baseline_auprc(baselines, label, target)


def build_scientific_answers(
    by: Dict[Tuple[str, str], Dict[str, Any]],
    baselines: Dict[str, Dict[str, Dict[str, Any]]],
    ab_fh: Dict[str, Any],
    ab_expert: Dict[str, Any],
    shares: Dict[str, Any],
    milestone: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Answer Q1–Q12 carefully; no causal overclaim; seed-2 validation-only."""

    def fh_delta(label: str) -> Dict[str, float]:
        return {
            t: _auprc(by, FIXED_HALF_ARM, t) - _metric_lookup(by, baselines, label, t)
            for t in TARGETS
        }

    d_vs_adap = fh_delta("GBT_TF_ADAPTIVE@1500")
    d_vs_gbt = fh_delta("GBT_STDFLOOR@1500")
    d_vs_expert = fh_delta(EXPERT_ONLY_ARM)
    d_vs_long = fh_delta("ADAPTIVE_LONG@1500")

    q1 = {
        "question": "Does fixed-half GBT+TF beat adaptive GBT+TF at matched step 1500?",
        "match_kind": "matched_step_1500",
        "delta_AUPRC_vs_GBT_TF_ADAPTIVE@1500": d_vs_adap,
        "verdict": {
            "beats_all": all(v > 0 for v in d_vs_adap.values()),
            "beats_any": any(v > 0 for v in d_vs_adap.values()),
            "beats_majority": sum(1 for v in d_vs_adap.values() if v > 0) >= 2,
        },
        "note": "Seed-2 validation-only; do not treat as causal proof of objective design.",
    }
    q2 = {
        "question": "Does fixed-half beat standalone GBT at step 1500?",
        "match_kind": "matched_step_1500",
        "delta_AUPRC_vs_GBT_STDFLOOR@1500": d_vs_gbt,
        "verdict": {
            "beats_all": all(v > 0 for v in d_vs_gbt.values()),
            "beats_any": any(v > 0 for v in d_vs_gbt.values()),
        },
    }
    q3 = {
        "question": "Does fixed-half match or beat EXPERT_ONLY at step 1500?",
        "match_kind": "matched_step_1500",
        "delta_AUPRC_vs_EXPERT_ONLY_1500": d_vs_expert,
        "verdict": {
            "matches_or_exceeds_all": all(v >= 0 for v in d_vs_expert.values()),
            "exceeds_any": any(v > 0 for v in d_vs_expert.values()),
        },
    }
    q4 = {
        "question": "Does fixed-half match or beat adaptive InfoNCE+TF LONG at step 1500?",
        "match_kind": "matched_step_1500",
        "delta_AUPRC_vs_ADAPTIVE_LONG@1500": d_vs_long,
        "verdict": {
            "matches_or_exceeds_all": all(v >= 0 for v in d_vs_long.values()),
            "exceeds_any": any(v > 0 for v in d_vs_long.values()),
        },
    }

    gap_closed = {}
    for t in TARGETS:
        expert = _auprc(by, EXPERT_ONLY_ARM, t)
        gbt = _baseline_auprc(baselines, "GBT_STDFLOOR@1500", t)
        fh = _auprc(by, FIXED_HALF_ARM, t)
        gap = expert - gbt
        closed = (fh - gbt) / gap if abs(gap) > 1e-12 else float("nan")
        gap_closed[t] = {
            "AUPRC_GBT": gbt,
            "AUPRC_fixed_half": fh,
            "AUPRC_expert": expert,
            "expert_minus_gbt": gap,
            "fraction_of_expert_gap_closed": closed,
        }
    q5 = {
        "question": "How much of the expert-only gap is closed by protecting 50% TF mass?",
        "per_target": gap_closed,
        "note": (
            "Fraction (AUPRC_fh − AUPRC_GBT) / (AUPRC_expert − AUPRC_GBT); "
            "not a causal attribution to the 50% TF mass alone."
        ),
    }

    improved_vs_adap = {t: d_vs_adap[t] > 0 for t in TARGETS}
    q6 = {
        "question": "Does fixed weighting improve all three targets or only selected domains?",
        "improved_vs_adaptive_per_target": improved_vs_adap,
        "n_targets_improved_vs_adaptive": sum(1 for v in improved_vs_adap.values() if v),
        "verdict": (
            "all_three"
            if all(improved_vs_adap.values())
            else ("selected_domains" if any(improved_vs_adap.values()) else "none")
        ),
        "note": "Do not compare absolute AUPRC across datasets as equal difficulty.",
    }

    q7 = {
        "question": "What beta allocation is learned within the fixed TF half?",
        "checkpoint_fixed_half": {
            "beta": ab_fh["beta"],
            "w_tf": ab_fh["w_tf"],
            "tf_target_names": list(TF_TARGET_NAMES),
        },
        "checkpoint_expert_only": {
            "beta": ab_expert["beta"],
            "w_tf": ab_expert["w_tf"],
        },
    }

    q8 = {
        "question": "What are the nominal and realized weighted contributions at step 1500?",
        "nominal_from_checkpoint_effective_weights_fixed_half": {
            "w_gbt": ab_fh["w_gbt"],
            "w_tf": ab_fh["w_tf"],
            "sum_w_tf": ab_fh["sum_w_tf"],
        },
        "realized_from_steps_jsonl_index_1499": shares,
        "aggregate_milestone_1500": milestone,
        "note": "Weights and loss shares are not gradient compatibility measurements.",
    }

    supports_easier = bool(q1["verdict"]["beats_any"])
    q9 = {
        "question": (
            "Do the results support the hypothesis that learned alpha favored the "
            "easier GBT loss rather than the downstream-optimal mixture?"
        ),
        "fixed_half_beats_adaptive_any": supports_easier,
        "fixed_half_beats_adaptive_all": bool(q1["verdict"]["beats_all"]),
        "verdict": (
            "supports_but_does_not_prove"
            if supports_easier
            else "does_not_support_from_these_metrics"
        ),
        "note": (
            "Fixed-half beating adaptive supports—but does not by itself prove—"
            "the easier-objective hypothesis."
        ),
    }

    closes_or_beats_expert = bool(q3["verdict"]["matches_or_exceeds_all"]) or bool(
        q3["verdict"]["exceeds_any"]
    )
    beats_adap = bool(q1["verdict"]["beats_majority"] or q1["verdict"]["beats_all"])
    retain = None
    if closes_or_beats_expert and beats_adap:
        retain = "lean_retain_gbt_in_mixture_under_fixed_half_protection"
    elif beats_adap and not closes_or_beats_expert:
        retain = "inconclusive_for_retention_improves_vs_adaptive_but_expert_gap_remains"
    elif closes_or_beats_expert:
        retain = "lean_retain_only_if_expert_gap_closed_without_adaptive_win_disclosed"
    else:
        retain = "lean_against_retaining_gbt_as_primary_mixture_component"
    q10 = {
        "question": "Does fixed-half provide sufficient evidence to retain GBT in the final model?",
        "verdict": retain,
        "inputs": {
            "beats_adaptive_majority": beats_adap,
            "closes_or_beats_expert_any_or_all": closes_or_beats_expert,
            "beats_standalone_gbt_any": bool(q2["verdict"]["beats_any"]),
        },
        "note": (
            "Careful retention language only; seed-2 validation on pretraining domains; "
            "do not infer unseen-domain transfer."
        ),
    }

    means = {}
    for label in PRIMARY_MATCHED_1500_LABELS:
        vals = [_metric_lookup(by, baselines, label, t) for t in TARGETS]
        means[label] = {
            "mean_AUPRC": float(np.mean(vals)),
            "per_target": {t: _metric_lookup(by, baselines, label, t) for t in TARGETS},
        }
    strongest = max(means.items(), key=lambda kv: kv[1]["mean_AUPRC"])[0]
    q11 = {
        "question": "Which single common step-1500 encoder is strongest across the three validation targets?",
        "ranking_by_mean_AUPRC": means,
        "strongest_by_mean_AUPRC": strongest,
        "note": (
            "Mean AUPRC across heterogeneous targets is a convenience summary only; "
            "do not treat datasets as equal difficulty."
        ),
    }

    q12 = {
        "question": "Is there any scientifically compelling reason to continue GBT experiments?",
        "verdict": False,
        "rationale": (
            "This is the final authorized GBT experiment. Residual uncertainty "
            "(seed-2, validation-only, no gradient diagnostic) does not authorize "
            "further GBT training, weight sweeps, longer runs, or new seeds under "
            "the current stop condition."
        ),
        "do_not_launch": [
            "gradient-conflict diagnostics",
            "additional fixed weights",
            "longer runs",
            "additional seeds",
            "new morphology heads",
            "new augmentations",
            "projection variants",
            "test evaluation",
            "any other GBT experiment",
        ],
    }

    return {
        "Q1_vs_adaptive_gbt_tf": q1,
        "Q2_vs_standalone_gbt": q2,
        "Q3_vs_expert_only_1500": q3,
        "Q4_vs_adaptive_long_1500": q4,
        "Q5_expert_gap_closed": q5,
        "Q6_all_vs_selected_targets": q6,
        "Q7_beta_allocation": q7,
        "Q8_nominal_and_realized_weights": q8,
        "Q9_easier_objective_hypothesis": q9,
        "Q10_retain_gbt_in_final_model": q10,
        "Q11_strongest_matched_1500_encoder": q11,
        "Q12_continue_gbt_experiments": q12,
        "caveats": [
            "Seed 2 only; validation-only; no test.",
            "Weights/loss shares ≠ gradient compatibility.",
            "Do not infer unseen-domain transfer from pretraining-domain eval.",
            "Do not compare absolute AUPRC across datasets as equal difficulty.",
            ALPHA_INTERPRETATION_NOTE,
        ],
    }


def _write_figures(
    primary_rows: List[Dict[str, Any]],
    delta_rows: List[Dict[str, Any]],
) -> Dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = result_dir() / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    # Primary matched-step AUPRC / F1@0.5 grouped by target.
    for metric, fname in (("AUPRC", "auprc_matched_1500.png"), ("F1@0.5", "f1_at_0_5_matched_1500.png")):
        fig, ax = plt.subplots(figsize=(10, 4.5))
        arms = []
        for r in primary_rows:
            if r["Arm"] not in arms:
                arms.append(r["Arm"])
        x = np.arange(len(TARGETS))
        width = 0.15
        for i, arm in enumerate(arms):
            vals = []
            for t in TARGETS:
                hit = next(r for r in primary_rows if r["Arm"] == arm and r["Target"] == t)
                vals.append(float(hit[metric]))
            ax.bar(x + (i - len(arms) / 2) * width + width / 2, vals, width, label=str(arm))
        ax.set_xticks(x)
        ax.set_xticklabels(list(TARGETS))
        ax.set_ylabel(metric)
        ax.set_title(f"Matched-step-1500 {metric} (validation-only, seed 2)")
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        out = fig_dir / fname
        fig.savefig(out, dpi=140)
        plt.close(fig)
        paths[metric] = str(out)

    # Matched-step deltas for fixed-half vs key baselines.
    comparisons = [
        ("GBT_TF_ADAPTIVE@1500", "delta_vs_adaptive.png"),
        ("GBT_STDFLOOR@1500", "delta_vs_gbt.png"),
        (EXPERT_ONLY_ARM, "delta_vs_expert.png"),
        ("ADAPTIVE_LONG@1500", "delta_vs_long.png"),
    ]
    for baseline, fname in comparisons:
        fig, ax = plt.subplots(figsize=(7, 4))
        vals = []
        for t in TARGETS:
            hit = next(
                (
                    d
                    for d in delta_rows
                    if d["GBT_TF_arm"] == FIXED_HALF_ARM
                    and d["Target"] == t
                    and d["Baseline"] == baseline
                ),
                None,
            )
            vals.append(float(hit["delta_AUPRC"]) if hit else float("nan"))
        ax.bar(list(TARGETS), vals, color="#2a6f97")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel("ΔAUPRC (fixed-half − baseline)")
        ax.set_title(f"Matched-step ΔAUPRC vs {baseline}")
        fig.tight_layout()
        out = fig_dir / fname
        fig.savefig(out, dpi=140)
        plt.close(fig)
        paths[f"delta_{baseline}"] = str(out)

    return paths


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
    matched_path = result_dir() / "matched_arm_comparability.json"
    matched_card = (
        json.loads(matched_path.read_text(encoding="utf-8")) if matched_path.is_file() else None
    )

    baselines = {}
    for label in BASELINE_ARMS:
        baselines[label] = {t: load_baseline_cell(label, t) for t in TARGETS}

    # Primary matched-step-1500 table.
    primary_rows: List[Dict[str, Any]] = []
    for label in PRIMARY_MATCHED_1500_LABELS:
        for target in TARGETS:
            if label in (FIXED_HALF_ARM, EXPERT_ONLY_ARM):
                primary_rows.append(_row_from_cell(by[(label, target)]))
            else:
                primary_rows.append(_row_from_baseline(baselines[label][target], label))

    # Secondary unmatched @3000 table.
    secondary_rows: List[Dict[str, Any]] = []
    for label, kind in BASELINE_MATCH_KIND.items():
        if kind != "unmatched_checkpoint_3000":
            continue
        for target in TARGETS:
            row = _row_from_baseline(baselines[label][target], label)
            row["table"] = "secondary_unmatched_3000"
            secondary_rows.append(row)

    delta_rows = []
    for arm in TRAIN_ARMS:
        for target in TARGETS:
            c = by[(arm, target)]
            # vs baselines
            for label, bcells in baselines.items():
                ref = bcells[target]
                delta_rows.append(
                    {
                        "GBT_TF_arm": arm,
                        "Target": target,
                        "AUPRC": c["validation_auprc"],
                        "Baseline": label,
                        "Baseline_AUPRC": ref["validation_auprc"],
                        "delta_AUPRC": c["validation_auprc"] - ref["validation_auprc"],
                        "retention_AUPRC": c["validation_auprc"] / ref["validation_auprc"]
                        if ref["validation_auprc"]
                        else float("nan"),
                        "match_kind": BASELINE_MATCH_KIND.get(label),
                    }
                )
            # fixed-half vs newly extracted expert
            if arm == FIXED_HALF_ARM:
                ref = by[(EXPERT_ONLY_ARM, target)]
                delta_rows.append(
                    {
                        "GBT_TF_arm": arm,
                        "Target": target,
                        "AUPRC": c["validation_auprc"],
                        "Baseline": EXPERT_ONLY_ARM,
                        "Baseline_AUPRC": ref["validation_auprc"],
                        "delta_AUPRC": c["validation_auprc"] - ref["validation_auprc"],
                        "retention_AUPRC": c["validation_auprc"] / ref["validation_auprc"]
                        if ref["validation_auprc"]
                        else float("nan"),
                        "match_kind": "matched_step_1500",
                    }
                )

    ab_fh = _alpha_beta_from_ckpt(FIXED_HALF_ARM, "fixed_half")
    ab_expert = _alpha_beta_from_ckpt(EXPERT_ONLY_ARM, "expert_only")
    # steps.jsonl uses 0-based completed index; checkpoint 1500 ↔ logged step 1499
    shares = _realized_shares_from_step(_step_row_at(1499))
    milestone = _milestone_1500()
    weights_payload = {
        "fixed_half_checkpoint": ab_fh,
        "expert_only_checkpoint": ab_expert,
        "step_log_shares_1499": shares,
        "aggregate_milestone_1500": milestone,
        "alpha_interpretation_note": ALPHA_INTERPRETATION_NOTE,
    }
    write_json(result_dir() / "beta_weighted_contributions.json", weights_payload)

    answers = build_scientific_answers(by, baselines, ab_fh, ab_expert, shares, milestone)
    write_json(result_dir() / "scientific_answers.json", answers)

    write_json(result_dir() / "cells" / "primary_matched_1500_table.json", primary_rows)
    write_json(result_dir() / "cells" / "secondary_unmatched_3000_table.json", secondary_rows)
    write_json(result_dir() / "cells" / "deltas_vs_baselines.json", delta_rows)

    fig_paths = _write_figures(primary_rows, delta_rows)

    final_rec = answers["Q10_retain_gbt_in_final_model"]
    payload = {
        "ok": True,
        "phase": "gbt_tf_fixed_half_stdfloor_1e4_frozen_eval",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective_id": OBJECTIVE_ID,
        "gbt_std_floor": GBT_STD_FLOOR,
        "scientific_questions": answers,
        "beta_weighted_contributions": weights_payload,
        "matched_arm_comparability": matched_card,
        "baseline_comparability": comparability,
        "probe_protocol": PROBE,
        "primary_matched_1500_table": primary_rows,
        "secondary_unmatched_3000_table": secondary_rows,
        "deltas_vs_baselines": delta_rows,
        "figures": fig_paths,
        "final_recommendation_on_gbt": final_rec,
        "alpha_interpretation_note": ALPHA_INTERPRETATION_NOTE,
        "baselines_reused_not_reextracted": True,
        "test_data_loaded_or_scored": False,
        "encoder_retrained": False,
        "no_recovery_checkpoint_eval": True,
        "final_authorized_gbt_experiment": True,
        "no_further_gbt_experiments_authorized": True,
    }
    write_json(result_dir() / "aggregate.json", payload)
    write_json(ROOT / TWIN_JSON, payload)

    lines = [
        "# GBT+TF fixed-half @1500 + EXPERT_ONLY@1500 frozen R198 validation eval",
        "",
        f"> Twin: `{TWIN_JSON}`",
        f"> Objective: `{OBJECTIVE_ID}` (gbt_std_floor={GBT_STD_FLOOR})",
        "",
        f"**ok={payload['ok']}** — validation-only; direct encoder R198; no test; no retrain.",
        "",
        f"Matched-arm comparability: `{(matched_card or {}).get('verdict')}`",
        f"Baseline comparability: `{comparability.get('verdict')}`",
        "",
        f"**Alpha interpretation:** {ALPHA_INTERPRETATION_NOTE}",
        "",
        "## Protocol locks",
        "",
        f"- contract: `{CONTRACT_ID}`",
        f"- probe: PaperStyleMLP {PROBE['epochs']}ep lr={PROBE['lr']} bs={PROBE['batch_size']} seed={PROBE['seed']}",
        "- extract: full-subgraph R198 train/val; projection bypassed; no raw/TF concat",
        f"- new arms: `{FIXED_HALF_ARM}` (.pt) + `{EXPERT_ONLY_ARM}` (.tar) @1500",
        "- F1@val-thr is optimistic (same-val threshold); not a test estimate",
        "",
        "## Primary matched-step-1500 table",
        "",
        "| Arm | Step | Target | AUPRC | AUROC | F1@0.5 | P | R | F1@val-thr* | Final val BCE |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in primary_rows:
        lines.append(
            f"| {r['Arm']} | {r['Step']} | {r['Target']} | "
            f"{float(r['AUPRC']):.4f} | {float(r['AUROC']):.4f} | {float(r['F1@0.5']):.4f} | "
            f"{float(r['P']):.4f} | {float(r['R']):.4f} | {float(r['F1@val-thr']):.4f} | "
            f"{float(r['Final val BCE']):.6f} |"
        )
    lines += ["", "*F1@val-thr optimistic diagnostic.", ""]

    lines += [
        "",
        "## Secondary unmatched-checkpoint-3000 table (context only)",
        "",
        "| Arm | Step | Target | AUPRC | F1@0.5 | match_kind |",
        "|---|---:|---|---:|---:|---|",
    ]
    for r in secondary_rows:
        lines.append(
            f"| {r['Arm']} | {r['Step']} | {r['Target']} | "
            f"{float(r['AUPRC']):.4f} | {float(r['F1@0.5']):.4f} | {r.get('match_kind')} |"
        )

    lines += ["", "## Fixed-half β / weighted contributions @1500", ""]
    lines.append(
        f"- Nominal (effective_weights fixed_half): α={ab_fh['alpha']:.4f} "
        f"β={ab_fh['beta']} w_gbt={ab_fh['w_gbt']:.4f} w_tf={ab_fh['w_tf']}"
    )
    lines.append(
        f"- EXPERT_ONLY effective_weights: β={ab_expert['beta']} "
        f"w_contrast={ab_expert['w_gbt']:.4f} w_tf={ab_expert['w_tf']}"
    )
    if shares.get("ok"):
        lines.append(
            f"- Realized steps.jsonl@1499: w_gbt={shares['w_gbt']:.4f} "
            f"w_tf={shares['w_tf']} weighted_gbt={shares.get('weighted_gbt')} "
            f"weighted_tf={shares.get('weighted_tf')}"
        )
    if milestone:
        lines.append(f"- Aggregate milestone@1500: {json.dumps(milestone, default=str)}")

    lines += ["", "## Scientific answers (Q1–Q12)", ""]
    for key, block in answers.items():
        if key == "caveats":
            continue
        lines.append(f"### {key}")
        lines.append(f"- Q: {block.get('question')}")
        if "verdict" in block:
            lines.append(f"- Verdict: `{json.dumps(block['verdict'], default=str)}`")
        if block.get("note"):
            lines.append(f"- Note: {block['note']}")
        if block.get("rationale"):
            lines.append(f"- Rationale: {block['rationale']}")
        lines.append("")

    lines += ["", "## Final recommendation on GBT", ""]
    lines.append(f"- Verdict: `{final_rec.get('verdict')}`")
    lines.append(f"- Note: {final_rec.get('note')}")
    lines.append("- Q12: no further GBT experiments authorized under this stop condition.")

    lines += [
        "",
        "Confirmation: no test data loaded/scored; baselines not re-extracted; "
        "recovery checkpoints never evaluated; historical training artifacts unchanged.",
        "",
    ]
    notes = ROOT / NOTES_PATH
    notes.parent.mkdir(parents=True, exist_ok=True)
    notes.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": True,
                "notes": str(notes),
                "n_primary_rows": len(primary_rows),
                "n_secondary_rows": len(secondary_rows),
                "figures": fig_paths,
                "final_recommendation": final_rec.get("verdict"),
            },
            indent=2,
        )
    )
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
