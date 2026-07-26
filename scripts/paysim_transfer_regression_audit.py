#!/usr/bin/env python3
"""PaySim transfer regression audit (no encoder training).

Stages:
  provenance — write Stage-0 forensic JSON
  logistic   — H-only logistic on an embeddings dir (legacy protocol)
  mlp        — PaperStyleMLP stacks on embeddings + PaySim X (current protocol)
  assemble   — merge cell results into final audit MD/JSON

Never writes under historical ``embeddings/paysim/{hi_contrastive*,random_init_gin}/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcpal_txn_node.eval_mlp import PaperStyleMLP, _predict_proba  # noqa: E402
from linear_probe import load_embedding_npz  # noqa: E402
from ranking_metrics import alert_budget_metrics  # noqa: E402
from train_util import extract_param  # noqa: E402
from util import create_parser, logger_setup, set_seed  # noqa: E402

AUDIT_ROOT = ROOT / "results" / "diagnostics" / "paysim_regression_audit"
EMBED_ROOT = ROOT / "embeddings" / "paysim_regression_audit"
LEGACY_FORBIDDEN = ROOT / "embeddings" / "paysim"
FINAL_MD = ROOT / "notes" / "paysim_transfer_regression_audit.md"
FINAL_JSON = ROOT / "results" / "diagnostics" / "paysim_transfer_regression_audit.json"

MLP_EPOCHS = 15
MLP_LR = 1e-3
MLP_BS = 8192
DOWNSTREAM_SEED = 2


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ids_hash(ids: np.ndarray) -> Dict[str, Any]:
    ids = np.asarray(ids, dtype=np.int64)
    return {
        "n": int(ids.shape[0]),
        "n_unique": int(np.unique(ids).shape[0]),
        "edge_id_sum": int(ids.sum()) if ids.size else 0,
        "sha256_of_ids_bytes": hashlib.sha256(ids.tobytes()).hexdigest() if ids.size else None,
    }


def assert_not_legacy_write(path: Path) -> None:
    try:
        path.resolve().relative_to(LEGACY_FORBIDDEN.resolve())
    except ValueError:
        return
    raise RuntimeError(f"Refusing write under historical PaySim dir: {path}")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def metrics_block(y: np.ndarray, proba: np.ndarray, thr: float) -> Dict[str, float]:
    y = y.astype(np.int64)
    pred = (proba >= float(thr)).astype(np.int64)
    out = {
        "auroc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        "auprc": float(average_precision_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "threshold": float(thr),
        "positive_prediction_rate": float(pred.mean()) if y.size else 0.0,
        "tp": float(((pred == 1) & (y == 1)).sum()),
        "fp": float(((pred == 1) & (y == 0)).sum()),
        "tn": float(((pred == 0) & (y == 0)).sum()),
        "fn": float(((pred == 0) & (y == 1)).sum()),
        "n": float(y.shape[0]),
    }
    out.update(alert_budget_metrics(y, proba))
    return out


def tune_thr_max_f1(y: np.ndarray, proba: np.ndarray) -> float:
    y = y.astype(np.int64)
    if len(np.unique(y)) < 2:
        return 0.5
    prec, rec, thrs = precision_recall_curve(y, proba)
    if thrs.size == 0:
        return 0.5
    f1 = (2 * prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
    return float(thrs[int(np.argmax(f1))])


def gin_model_class_weight() -> Dict[int, float]:
    args = create_parser().parse_args(["--data", "PaySim", "--model", "gin", "--testing"])
    return {0: float(extract_param("w_ce1", args)), 1: float(extract_param("w_ce2", args))}


def run_logistic_on_dir(
    emb_dir: Path,
    *,
    class_weight_mode: str = "model",
    C: float = 1.0,
    seed: int = 1,
    label: str = "logistic",
) -> Dict[str, Any]:
    """Legacy H-only logistic protocol (June 2026)."""
    splits = {}
    for sp in ("train", "val", "test"):
        z, y, ids = load_embedding_npz(emb_dir / f"{sp}.npz")
        splits[sp] = {"Z": z, "y": y, "ids": ids}
    if class_weight_mode == "model":
        cw: Any = gin_model_class_weight()
    elif class_weight_mode == "none":
        cw = None
    elif class_weight_mode == "balanced":
        cw = "balanced"
    else:
        raise ValueError(class_weight_mode)

    set_seed(seed)
    # n_jobs=1: never fan out OpenMP on a shared login node; Slurm CPU cells are the
    # right place for this fit. Using -1 previously OOM/CPU-thrashed login hosts.
    clf = LogisticRegression(
        class_weight=cw, max_iter=1000, random_state=seed, solver="lbfgs", n_jobs=1, C=float(C)
    )
    clf.fit(splits["train"]["Z"], splits["train"]["y"])
    proba = {
        sp: clf.predict_proba(splits[sp]["Z"])[:, 1].astype(np.float64) for sp in splits
    }
    thr = tune_thr_max_f1(splits["val"]["y"], proba["val"])
    report = {
        "label": label,
        "embeddings_dir": str(emb_dir),
        "learner": "LogisticRegression",
        "class_weight_mode": class_weight_mode,
        "class_weight": cw if not isinstance(cw, dict) else {str(k): float(v) for k, v in cw.items()},
        "C": float(C),
        "seed": seed,
        "feature_stack": "H_only",
        "h_dim": int(splits["train"]["Z"].shape[1]),
        "validation_selected_threshold": thr,
        "ids": {sp: ids_hash(splits[sp]["ids"]) for sp in splits},
        "threshold_0.5": metrics_block(splits["test"]["y"], proba["test"], 0.5),
        "threshold_val_selected": metrics_block(splits["test"]["y"], proba["test"], thr),
        "val_ranking": {
            "auroc": float(roc_auc_score(splits["val"]["y"], proba["val"])),
            "auprc": float(average_precision_score(splits["val"]["y"], proba["val"])),
        },
    }
    return report


def load_paysim_x() -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "probe_feature_ablation", ROOT / "scripts" / "probe_feature_ablation.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["probe_feature_ablation"] = mod
    spec.loader.exec_module(mod)
    df, df_train, tr_ids, va_ids, te_ids, _spec = mod.load_dataset_frames(
        "PaySim", str(ROOT / "data_config.json")
    )
    x_raw, names, _, meta = mod.build_full_feature_matrix(
        df, df_train, ("edge_native",), categorical_encoding="one_hot"
    )
    y_all = df[_spec.label_col].to_numpy().astype(np.int64)
    return x_raw.astype(np.float32), y_all, {
        "train": np.asarray(tr_ids, dtype=np.int64),
        "val": np.asarray(va_ids, dtype=np.int64),
        "test": np.asarray(te_ids, dtype=np.int64),
        "feat_names": names,
        "meta": meta,
        "x_dim": int(x_raw.shape[1]),
    }


def train_mlp_val_auprc(x_tr, y_tr, x_va, y_va, x_te, y_te, device, epochs=MLP_EPOCHS):
    torch.manual_seed(DOWNSTREAM_SEED)
    np.random.seed(DOWNSTREAM_SEED)
    model = PaperStyleMLP(int(x_tr.shape[1])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=MLP_LR)
    x_t = torch.from_numpy(x_tr.astype(np.float32))
    y_t = torch.from_numpy(y_tr.astype(np.float32))
    n = x_tr.shape[0]
    best_auprc, best_state, best_ep = -1.0, None, -1
    for ep in range(epochs):
        model.train()
        perm = np.random.RandomState(DOWNSTREAM_SEED * 1009 + ep).permutation(n)
        for start in range(0, n, MLP_BS):
            idx = perm[start : start + MLP_BS]
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(x_t[idx].to(device)), y_t[idx].to(device)
            )
            loss.backward()
            opt.step()
        pva = _predict_proba(model, x_va, batch_size=MLP_BS, device=device)
        auprc = float(average_precision_score(y_va, pva))
        if auprc > best_auprc + 1e-12:
            best_auprc = auprc
            best_ep = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.to(device)
    pva = _predict_proba(model, x_va, batch_size=MLP_BS, device=device)
    pte = _predict_proba(model, x_te, batch_size=MLP_BS, device=device)
    thr = tune_thr_max_f1(y_va, pva)
    return {
        "best_epoch": best_ep,
        "best_val_auprc": best_auprc,
        "validation_selected_threshold": thr,
        "threshold_0.5": metrics_block(y_te, pte, 0.5),
        "threshold_val_selected": metrics_block(y_te, pte, thr),
        "val_ranking": {
            "auroc": float(roc_auc_score(y_va, pva)),
            "auprc": float(average_precision_score(y_va, pva)),
        },
    }


def run_mlp_stacks(
    emb_dir: Path,
    *,
    stacks: List[str],
    device_str: str = "cuda:0",
    label: str = "mlp",
    x_pack: Optional[Tuple] = None,
) -> Dict[str, Any]:
    """Current locked MLP evaluator on H / H+X (and X-only if requested)."""
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    if x_pack is None:
        x_raw, y_all, x_meta = load_paysim_x()
    else:
        x_raw, y_all, x_meta = x_pack

    splits = {}
    for sp in ("train", "val", "test"):
        z, y, ids = load_embedding_npz(emb_dir / f"{sp}.npz")
        if not np.array_equal(y, y_all[ids]):
            raise RuntimeError(f"label mismatch {sp}")
        splits[sp] = {"Z": z, "y": y, "ids": ids}

    out: Dict[str, Any] = {
        "label": label,
        "embeddings_dir": str(emb_dir),
        "learner": "PaperStyleMLP",
        "h_dim": int(splits["train"]["Z"].shape[1]),
        "x_dim": int(x_meta["x_dim"]),
        "ids": {sp: ids_hash(splits[sp]["ids"]) for sp in splits},
        "stacks": {},
    }
    for stack in stacks:
        feats = {}
        for sp in ("train", "val", "test"):
            ids = splits[sp]["ids"]
            z = splits[sp]["Z"]
            x = x_raw[ids]
            if stack == "H_only":
                mat = z
            elif stack == "HxX":
                mat = np.concatenate([z, x], axis=1)
            elif stack == "X_only":
                mat = x
            else:
                raise ValueError(stack)
            feats[sp] = {"X": mat.astype(np.float32), "y": splits[sp]["y"]}
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(feats["train"]["X"]).astype(np.float32)
        x_va = scaler.transform(feats["val"]["X"]).astype(np.float32)
        x_te = scaler.transform(feats["test"]["X"]).astype(np.float32)
        out["stacks"][stack] = train_mlp_val_auprc(
            x_tr, feats["train"]["y"], x_va, feats["val"]["y"], x_te, feats["test"]["y"], device
        )
        out["stacks"][stack]["feature_dim"] = int(x_tr.shape[1])
    return out


def cmd_provenance(_: argparse.Namespace) -> int:
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    legacy_ckpt = ROOT / "saved-models/checkpoint_hi_contrastive_proj_sym_20ep_bestckpt.tar"
    dplus_ckpt = ROOT / (
        "saved-models/checkpoint_gin_emlps_ports_tds_corrected_preserve_seed_"
        "asym_proj_8192neg_queue0_40ep_seed2.tar"
    )
    probe_hi = ROOT / (
        "embeddings/paysim/hi_contrastive_proj_sym_20ep_bestckpt/probe_results_cw_model_thr0.5.json"
    )
    probe_rand = ROOT / "embeddings/paysim/random_init_gin/probe_results_cw_model_thr0.5.json"
    final = ROOT / "results/diagnostics/paysim_dplus_transfer_final.json"
    role2 = ROOT / "results/diagnostics/paysim_dplus_transfer_final/role_seed2.json"
    role_rand = ROOT / "results/diagnostics/paysim_dplus_transfer_final/role_random_init.json"

    def loadj(p: Path):
        return json.loads(p.read_text()) if p.is_file() else None

    hi_probe = loadj(probe_hi)
    rand_probe = loadj(probe_rand)
    final_j = loadj(final)
    r2 = loadj(role2)
    rr = loadj(role_rand)

    ckpt = torch.load(legacy_ckpt, map_location="cpu") if legacy_ckpt.is_file() else {}
    sd = ckpt.get("model_state_dict", {})
    edge_shapes = {
        k: list(v.shape) for k, v in sd.items() if "edge_emb" in k and k.endswith("weight")
    }

    provenance = {
        "title": "paysim_transfer_regression_audit_stage0",
        "old_protocol": {
            "jobs": {
                "pretrained_extract_probe": 16036046,
                "random_extract_probe": 16043535,
                "probe_variants_documented": [16043589, 16043590, 16043591, 16043595, 16043596, 16043597],
                "probe_variants_missing_from_record": [16043592, 16043593, 16043594],
            },
            "unique_name_pretrained": "hi_contrastive_proj_sym_20ep_bestckpt",
            "unique_name_random": "random_init_gin",
            "checkpoint": {
                "path": str(legacy_ckpt),
                "exists": legacy_ckpt.is_file(),
                "sha256": sha256_file(legacy_ckpt) if legacy_ckpt.is_file() else None,
                "epoch": ckpt.get("epoch"),
                "edge_emb_shapes": edge_shapes,
                "edge_dim": 6,
            },
            "flags": {
                "ports": True,
                "tds": False,
                "emlps": False,
                "ego": True,
                "reverse_mp": True,
                "correct_reverse_edge_features": False,
                "preserve_seed_edges": False,
                "projection_at_extract": False,
                "note": "Projection head present in ckpt but extract uses embedding_head post-128 only",
            },
            "representation": "post_embedding_128",
            "normalization": {
                "node_x": "train-fit clone",
                "edge_attr": "independent per-graph z_norm (train / train+val / all-edges test)",
                "train_fit_edge_znorm": False,
                "test_edge_znorm_uses_test_graph_attrs": True,
                "classification": "PARTIAL_transductive_target_edge_znorm — not proven invalid, but not inductive",
            },
            "probe": {
                "learner": "sklearn LogisticRegression lbfgs",
                "feature_stack": "H_only",
                "class_weight_canonical": "model (GIN w_ce1/w_ce2)",
                "C": 1.0,
                "seed": 1,
                "threshold_rule_canonical": "max_f1_val",
            },
            "surviving_metrics_cw_model_thr0.5": {
                "pretrained_test_auroc": (hi_probe or {})
                .get("splits_at_selected_threshold", {})
                .get("test", {})
                .get("auroc"),
                "random_test_auroc": (rand_probe or {})
                .get("splits_at_selected_threshold", {})
                .get("test", {})
                .get("auroc"),
            },
            "notes_only_metrics_cw_model_maxf1val": {
                "pretrained_test_auroc_rounded": 0.866,
                "random_test_auroc_rounded": 0.730,
                "source": "notes/downstream-eval-plan.md (original probe_results.json overwritten)",
            },
            "npz_caches_present": False,
            "split_counts_from_probe_json": {
                "train": 3792814,
                "val": 1276273,
                "test": 1293522,
                "test_positive_rate": 0.003291787847442873,
            },
        },
        "current_protocol": {
            "jobs": {"A": 18855316, "B": 18855317, "C": 18855318, "D": 18855319},
            "primary_stack": "pre3h_HxX",
            "flags": [
                "ports",
                "tds",
                "emlps",
                "ego",
                "reverse_mp",
                "correct_reverse_edge_features",
                "train_fit_edge_znorm",
            ],
            "edge_dim": 8,
            "seed2_checkpoint": {
                "path": str(dplus_ckpt),
                "sha256": sha256_file(dplus_ckpt) if dplus_ckpt.is_file() else None,
                "epoch": 40,
            },
            "primary_metrics": (final_j or {}).get("primary_pre3h_HxX"),
            "x_only": (final_j or {}).get("x_only"),
            "random_control": {
                "pre3h_H_auprc": (rr or {})
                .get("stacks", {})
                .get("pre3h_H_only", {})
                .get("threshold_0.5", {})
                .get("auprc"),
                "pre3h_HxX_auprc": (rr or {})
                .get("stacks", {})
                .get("pre3h_HxX", {})
                .get("threshold_0.5", {})
                .get("auprc"),
            },
            "seed2_stacks_auroc_auprc": {
                k: {
                    "auroc": (r2 or {}).get("stacks", {}).get(k, {}).get("threshold_0.5", {}).get("auroc"),
                    "auprc": (r2 or {}).get("stacks", {}).get(k, {}).get("threshold_0.5", {}).get("auprc"),
                }
                for k in ("X_only", "pre3h_H_only", "pre3h_HxX", "post128_H_only", "post128_HxX")
            },
            "normalization": {
                "edge_attr": "train_fit_edge_znorm inductive",
                "test_edge_znorm_uses_test_graph_attrs": False,
            },
        },
        "cohort_overlap_hypothesis": {
            "same_formatted_csv": True,
            "same_hourly_step_602020": True,
            "old_test_n": 1293522,
            "current_test_n_typical": 1293522,
            "note": "Confirm with reproduced extract ID hashes in Stage 1",
        },
    }
    out = AUDIT_ROOT / "stage0_provenance.json"
    write_json(out, provenance)
    logging.info("Wrote %s", out)
    return 0


def cmd_logistic(args: argparse.Namespace) -> int:
    emb = Path(args.embeddings_dir)
    if not emb.is_dir():
        raise SystemExit(f"missing {emb}")
    report = run_logistic_on_dir(
        emb,
        class_weight_mode=args.class_weight,
        C=args.C,
        seed=args.seed,
        label=args.label,
    )
    out = Path(args.output_json)
    write_json(out, report)
    logging.info(
        "Wrote %s test_auroc@0.5=%.4f auprc=%.4f",
        out,
        report["threshold_0.5"]["auroc"],
        report["threshold_0.5"]["auprc"],
    )
    return 0


def cmd_mlp(args: argparse.Namespace) -> int:
    emb = Path(args.embeddings_dir)
    stacks = [s.strip() for s in args.stacks.split(",") if s.strip()]
    report = run_mlp_stacks(emb, stacks=stacks, device_str=args.device, label=args.label)
    out = Path(args.output_json)
    write_json(out, report)
    logging.info("Wrote %s stacks=%s", out, list(report["stacks"]))
    return 0


def cmd_assemble(args: argparse.Namespace) -> int:
    cells_dir = Path(args.cells_dir)
    cells = {}
    for p in sorted(cells_dir.glob("*.json")):
        cells[p.stem] = json.loads(p.read_text())
    provenance = json.loads((AUDIT_ROOT / "stage0_provenance.json").read_text())
    final_cur = json.loads(
        (ROOT / "results/diagnostics/paysim_dplus_transfer_final.json").read_text()
    )

    def _get(cell, *path, default=None):
        cur = cells.get(cell, {})
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    # Stage 1 reproduction numbers
    repro_pre = _get("A_legacy_sym_logistic_model", "threshold_0.5", "auroc")
    repro_rand = _get("A_legacy_random_logistic_model", "threshold_0.5", "auroc")
    hist_pre = 0.864195902135397  # thr0.5 surviving JSON
    hist_rand = 0.7300642153727568

    answers = {
        "1_old_0866_reproducible": {
            "target_notes_maxf1": 0.866,
            "target_thr05_json": hist_pre,
            "reproduced_thr05_auroc": repro_pre,
            "reproduced_random_thr05_auroc": repro_rand,
            "delta_vs_thr05_json": None if repro_pre is None else abs(float(repro_pre) - hist_pre),
            "verdict": None,
        },
        "2_same_paysim_cohort": None,
        "3_old_result_validity": (
            "preliminary_but_defensible_diagnostic: ports-only post-128 logistic H-only; "
            "per-graph edge z_norm includes test-graph attrs (PARTIAL/transductive); "
            "not D+; original max_f1 JSON overwritten but thr0.5 JSON intact at 0.8642"
        ),
        "4_regression_from_encoder": None,
        "5_regression_from_evaluator": None,
        "6_dplus_component_most_reducing_transfer": None,
        "7_old_encoder_beats_random_under_current_eval": None,
        "8_dplus_beats_random_under_legacy_eval": None,
        "9_feature_schema_mismatch_plausible": (
            "Yes as demonstrated protocol difference: currency/payment placeholders; "
            "TDS/ports recomputed; D+ adds TDS+emlps+corrected reverse absent from old stack. "
            "Masking cells fill quantitative support when present."
        ),
        "10_smallest_followup_training": (
            "If Stage 2–3 show encoder-driven loss concentrated at TDS/corrected-reverse, "
            "smallest follow-up is ONE seed-2 contrastive retrain with ports-only (edge_dim=6) "
            "matched to old sym+proj recipe for transfer re-measure — only after validation-gated "
            "decision; do not auto-submit."
        ),
    }

    # Fill crossover comparisons when cells exist
    old_cur_h = _get("B_legacy_sym_mlp_H_only", "stacks", "H_only", "threshold_0.5", "auroc")
    old_cur_hx = _get("B_legacy_sym_mlp_HxX", "stacks", "HxX", "threshold_0.5", "auroc")
    dplus_leg_post = _get("C_dplus_seed2_post128_logistic_model", "threshold_0.5", "auroc")
    dplus_leg_pre = _get("C_dplus_seed2_pre3h_logistic_model", "threshold_0.5", "auroc")
    dplus_rand_leg = _get("C_dplus_random_post128_logistic_model", "threshold_0.5", "auroc")
    old_rand_cur = _get("B_legacy_random_mlp_H_only", "stacks", "H_only", "threshold_0.5", "auroc")

    cur_seed2_post_h = (
        final_cur.get("per_seed", {})
        .get("seed2", {})
        .get("stacks", {})
        .get("post128_H_only", {})
        .get("threshold_0.5", {})
        .get("auroc")
    )
    # role json may be nested differently in assembled final
    if cur_seed2_post_h is None:
        role2 = json.loads(
            (ROOT / "results/diagnostics/paysim_dplus_transfer_final/role_seed2.json").read_text()
        )
        cur_seed2_post_h = role2["stacks"]["post128_H_only"]["threshold_0.5"]["auroc"]
        cur_seed2_pre_h = role2["stacks"]["pre3h_H_only"]["threshold_0.5"]["auroc"]
        cur_seed2_hx = role2["stacks"]["pre3h_HxX"]["threshold_0.5"]["auroc"]
    else:
        cur_seed2_pre_h = final_cur["per_seed"]["seed2"]["stacks"]["pre3h_H_only"]["threshold_0.5"]["auroc"]
        cur_seed2_hx = final_cur["per_seed"]["seed2"]["stacks"]["pre3h_HxX"]["threshold_0.5"]["auroc"]

    if repro_pre is not None:
        answers["1_old_0866_reproducible"]["verdict"] = (
            "YES_approx_thr05"
            if abs(float(repro_pre) - hist_pre) < 0.01
            else "PARTIAL_or_FAIL"
        )

    # Cohort: compare ID hashes if available
    legacy_ids = _get("A_legacy_sym_logistic_model", "ids", "test")
    dplus_ids = None
    dplus_post = ROOT / "embeddings/paysim_dplus_transfer_final/dplus_seed2/post_embedding_128/test.npz"
    if dplus_post.is_file():
        _, _, ids = load_embedding_npz(dplus_post)
        dplus_ids = ids_hash(ids)
    if legacy_ids and dplus_ids:
        answers["2_same_paysim_cohort"] = {
            "legacy_test_ids": legacy_ids,
            "dplus_test_ids": dplus_ids,
            "same_n": legacy_ids.get("n") == dplus_ids.get("n"),
            "same_hash": legacy_ids.get("sha256_of_ids_bytes") == dplus_ids.get("sha256_of_ids_bytes"),
        }

    if old_cur_h is not None and repro_pre is not None:
        answers["5_regression_from_evaluator"] = {
            "legacy_encoder_legacy_logistic_auroc": repro_pre,
            "legacy_encoder_current_mlp_H_auroc": old_cur_h,
            "delta_auroc": float(old_cur_h) - float(repro_pre),
            "note": "Positive delta means current evaluator helps old encoder; negative hurts",
        }
    if dplus_leg_post is not None and repro_pre is not None:
        answers["4_regression_from_encoder"] = {
            "legacy_encoder_legacy_logistic_auroc": repro_pre,
            "dplus_post128_legacy_logistic_auroc": dplus_leg_post,
            "delta_auroc": float(dplus_leg_post) - float(repro_pre),
            "note": "Large negative delta under matched legacy evaluator implicates encoder/protocol extract",
        }
    if old_cur_h is not None and old_rand_cur is not None:
        answers["7_old_encoder_beats_random_under_current_eval"] = bool(float(old_cur_h) > float(old_rand_cur))
    if dplus_leg_post is not None and dplus_rand_leg is not None:
        answers["8_dplus_beats_random_under_legacy_eval"] = bool(
            float(dplus_leg_post) > float(dplus_rand_leg)
        )

    # Lineage
    lineage = {k: cells[k] for k in cells if k.startswith("L_")}
    if lineage:
        # pick lowest AUROC under logistic H-only as "most reducing" among available
        scores = []
        for k, v in lineage.items():
            auroc = v.get("threshold_0.5", {}).get("auroc")
            if auroc is not None:
                scores.append((k, float(auroc)))
        scores.sort(key=lambda x: x[1])
        answers["6_dplus_component_most_reducing_transfer"] = {
            "ranked_logistic_H_auroc_ascending": scores,
            "lowest": scores[0] if scores else None,
        }

    payload = {
        "title": "paysim_transfer_regression_audit",
        "provenance": provenance,
        "cells": cells,
        "crossover_summary": {
            "A_old_enc_old_eval_auroc": repro_pre,
            "A_old_rand_old_eval_auroc": repro_rand,
            "B_old_enc_current_mlp_H_auroc": old_cur_h,
            "B_old_enc_current_mlp_HxX_auroc": old_cur_hx,
            "C_dplus_post128_legacy_logistic_auroc": dplus_leg_post,
            "C_dplus_pre3h_legacy_logistic_auroc": dplus_leg_pre,
            "D_dplus_seed2_current_post128_H_auroc": cur_seed2_post_h,
            "D_dplus_seed2_current_pre3h_H_auroc": cur_seed2_pre_h,
            "D_dplus_seed2_current_pre3h_HxX_auroc": cur_seed2_hx,
        },
        "answers": answers,
        "no_encoder_training": True,
        "historical_artifacts_overwritten": False,
    }
    write_json(FINAL_JSON, payload)

    lines = [
        "# PaySim transfer regression audit",
        "",
        "No encoder training or fine-tuning. Diagnostic extraction/evaluation only.",
        "",
        "## Stage 0 — provenance (summary)",
        "",
        "### Old (June 2026)",
        "- Jobs: pretrained **16036046**, random **16043535**; probe variants 16043589–91, 95–97",
        "- Encoder: `hi_contrastive_proj_sym_20ep_bestckpt` ep20, **edge_dim=6**, ports-only, **no** TDS/emlps/corrected reverse",
        "- Representation: **post-128**; probe: sklearn logistic **H-only**, `class_weight=model`, C=1, seed=1",
        "- Edge z-norm: **independent per-graph** (test graph includes all edges → test attrs enter test z-norm) — PARTIAL/transductive",
        "- Surviving thr0.5 JSON: pretrained AUROC **0.8642**, random **0.7301**; notes max_f1 **0.866 / 0.730**",
        "- `.npz` caches missing before this audit (re-extract to `embeddings/paysim_regression_audit/` only)",
        "",
        "### Current (July 2026 D+)",
        "- Jobs 18855316–18; primary **pre-3h H+X** MLP; edge_dim=**8**; ports+tds+emlps+corrected reverse; **train_fit_edge_znorm**",
        f"- Primary mean test AUPRC {final_cur['primary_pre3h_HxX']['test_auprc']['mean']:.4f}, AUROC {final_cur['primary_pre3h_HxX']['test_auroc']['mean']:.4f}",
        "",
        "## Stage 1–2 cells",
        "",
    ]
    for name, cell in sorted(cells.items()):
        if "threshold_0.5" in cell:
            t = cell["threshold_0.5"]
            lines.append(
                f"- **{name}**: AUROC {t.get('auroc'):.4f} AUPRC {t.get('auprc'):.4f} "
                f"F1@0.5 {t.get('f1'):.4f} (H-dim={cell.get('h_dim')})"
            )
        elif "stacks" in cell:
            for sk, sm in cell["stacks"].items():
                t = sm.get("threshold_0.5", {})
                lines.append(
                    f"- **{name}/{sk}**: AUROC {t.get('auroc'):.4f} AUPRC {t.get('auprc'):.4f} F1@0.5 {t.get('f1'):.4f}"
                )
    lines += ["", "## Decision answers", ""]
    for k, v in answers.items():
        lines.append(f"- **{k}**: `{json.dumps(v, default=str)[:500]}`")
    lines += [
        "",
        "## Artifacts",
        f"- `{FINAL_MD}`",
        f"- `{FINAL_JSON}`",
        f"- cells: `{cells_dir}/`",
        f"- embeddings: `{EMBED_ROOT}/` (never historical `embeddings/paysim/*`)",
        "",
    ]
    FINAL_MD.write_text("\n".join(lines) + "\n")
    logging.info("Wrote %s and %s", FINAL_JSON, FINAL_MD)
    return 0


def main() -> None:
    logger_setup()
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("provenance")
    sp.set_defaults(func=cmd_provenance)

    sl = sub.add_parser("logistic")
    sl.add_argument("--embeddings_dir", required=True)
    sl.add_argument("--output_json", required=True)
    sl.add_argument("--class_weight", default="model", choices=("model", "none", "balanced"))
    sl.add_argument("--C", type=float, default=1.0)
    sl.add_argument("--seed", type=int, default=1)
    sl.add_argument("--label", default="logistic")
    sl.set_defaults(func=cmd_logistic)

    sm = sub.add_parser("mlp")
    sm.add_argument("--embeddings_dir", required=True)
    sm.add_argument("--output_json", required=True)
    sm.add_argument("--stacks", default="H_only,HxX")
    sm.add_argument("--device", default="cuda:0")
    sm.add_argument("--label", default="mlp")
    sm.set_defaults(func=cmd_mlp)

    sa = sub.add_parser("assemble")
    sa.add_argument("--cells_dir", default=str(AUDIT_ROOT / "cells"))
    sa.set_defaults(func=cmd_assemble)

    args = p.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
