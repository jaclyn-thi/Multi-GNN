#!/usr/bin/env python3
"""Validation decision gate for edge D+ neighbor-positive 10ep scout.

Loads the Small-HI graph once, extracts pre-3h H for selected checkpoints with
loader_num_workers=0 (hang-safe), joins X+TF after a single H forward, and fits
the same paper-style MLP used by job 18678029. Selection = temporal val AUPRC.

Does not retrain any GNN. Does not submit 40ep continuation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_loading import get_data
from embedding_extraction import run_embedding_extraction
from linear_probe import load_embedding_npz
from scripts.probe_feature_ablation import build_full_feature_matrix, load_dataset_frames
from util import create_parser, logger_setup, set_seed

REF_VAL_AUPRC = 0.550
TF_CACHE = _ROOT / "results/cache/temporal_flow_causal/Small-HI"
ARMS = {
    "identity": "edge_dplus_identity_poscomplete_10ep_seed2",
    "neighbor": "edge_dplus_neighbor_supcon_poscomplete_10ep_seed2",
}
PROTECTED_EMB_PREFIXES = (
    "gin_emlps_ports_tds_corrected_preserve_seed_asym_proj_8192neg_queue0_40ep_seed2",
    "gcpal_",
)


def _load_challenge_mod():
    path = _ROOT / "scripts" / "gcpal_challenge_fullstack_eval.py"
    spec = importlib.util.spec_from_file_location("gcpal_challenge_fullstack_eval", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["gcpal_challenge_fullstack_eval"] = mod
    spec.loader.exec_module(mod)
    return mod


def _file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _emb_dir(arm: str, ep: int) -> Path:
    return (
        _ROOT
        / "embeddings"
        / f"edge_dplus_nb_valgate_{arm}_ep{ep:02d}_seed2"
        / "pre_embedding_3h"
    )


def _assert_safe_outdir(out: Path) -> None:
    try:
        rel = str(out.relative_to(_ROOT / "embeddings"))
    except ValueError:
        rel = str(out)
    for pref in PROTECTED_EMB_PREFIXES:
        if rel.startswith(pref) or f"/{pref}" in f"/{rel}":
            raise RuntimeError(f"Refusing to write into protected embedding path: {out}")


def _profile_batches(
    tr_data,
    val_data,
    te_data,
    tr_inds,
    val_inds,
    te_inds,
    args,
    data_config,
    *,
    max_batches: int,
) -> Dict[str, Any]:
    """Timed smoke: extract a few train batches under the hang-safe loader settings."""
    from types import SimpleNamespace

    from torch_geometric.nn import to_hetero

    from train_util import (
        AddEgoIds,
        add_arange_ids,
        extract_param,
        extract_seed_embeddings_hetero,
        get_loaders,
        load_checkpoint_weights,
        resolve_embedding_head_linear,
    )
    from training import get_model

    t0 = time.perf_counter()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = AddEgoIds() if args.ego else None
    add_arange_ids([tr_data, val_data, te_data])
    sample_args = SimpleNamespace(**vars(args))
    sample_args.loader_num_workers = 0
    sample_loader, _, _ = get_loaders(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, sample_args, train_shuffle=False
    )
    sample_batch = next(iter(sample_loader))
    del sample_loader
    config = SimpleNamespace(
        model=args.model,
        n_hidden=extract_param("n_hidden", args),
        n_gnn_layers=extract_param("n_gnn_layers", args),
        n_heads=None,
        dropout=extract_param("dropout", args),
        final_dropout=extract_param("final_dropout", args),
    )
    model = get_model(sample_batch, config, args)
    head_spec = resolve_embedding_head_linear(model, int(getattr(model, "embedding_dim", 128)))
    pre_dim = head_spec.in_features
    model = to_hetero(model, te_data.metadata(), aggr="mean")
    load_checkpoint_weights(model, device, args, data_config)
    model.eval()
    tr_loader, _, _ = get_loaders(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args, train_shuffle=False
    )

    class _Cap:
        def __init__(self, loader, n):
            self._loader = loader
            self._n = n
            self.data = loader.data

        def __iter__(self):
            for i, b in enumerate(self._loader):
                if i >= self._n:
                    break
                yield b

        def __len__(self):
            return self._n

    capped = _Cap(tr_loader, max_batches)
    t_ext0 = time.perf_counter()
    with torch.inference_mode():
        edge_ids, z, y = extract_seed_embeddings_hetero(
            capped,
            tr_inds,
            model,
            tr_data,
            device,
            args,
            representation_source="pre_embedding_3h",
            pre_dim=pre_dim,
            emb_dim=128,
            head_spec=head_spec,
        )
    t_ext = time.perf_counter() - t_ext0
    per = t_ext / max(max_batches, 1)
    n_train_batches = int(np.ceil(int(tr_inds.numel()) / int(args.batch_size)))
    n_val_batches = int(np.ceil(int(val_inds.numel()) / int(args.batch_size)))
    proj_train = per * n_train_batches
    proj_val = per * n_val_batches
    # 4 ckpts (2 arms × ep5/10) × (train+val) + data load overhead budget
    proj_four = 4.0 * (proj_train + proj_val)
    out = {
        "profile_batches": max_batches,
        "profile_extract_s": float(t_ext),
        "sec_per_batch": float(per),
        "rows_profiled": int(edge_ids.numel()),
        "z_dim": int(z.shape[1]),
        "n_train_batches": n_train_batches,
        "n_val_batches": n_val_batches,
        "projected_train_s": float(proj_train),
        "projected_val_s": float(proj_val),
        "projected_4ckpt_train_val_h": float(proj_four / 3600.0),
        "wall_s": float(time.perf_counter() - t0),
        "device": str(device),
        "loader_num_workers": int(args.loader_num_workers),
    }
    logging.info("PROFILE %s", json.dumps(out))
    return out


def _ensure_extract(
    *,
    arm_key: str,
    ep: int,
    splits: Sequence[str],
    tr_data,
    val_data,
    te_data,
    tr_inds,
    val_inds,
    te_inds,
    data_config,
) -> Tuple[Path, Dict[str, Any]]:
    arm = ARMS[arm_key]
    out = _emb_dir(arm_key, ep)
    _assert_safe_outdir(out)
    needed = [out / f"{s}.npz" for s in splits]
    meta_path = out / "meta.json"
    if all(p.is_file() for p in needed) and meta_path.is_file():
        logging.info("Reuse cached embeddings %s", out)
        return out, json.loads(meta_path.read_text())

    suffix = f"_ep{ep:02d}"
    ckpt = _ROOT / "saved-models" / f"checkpoint_{arm}{suffix}.tar"
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    emb_subdir = f"edge_dplus_nb_valgate_{arm_key}_ep{ep:02d}_seed2"
    args = create_parser().parse_args(
        [
            "--data",
            "Small-HI",
            "--model",
            "gin",
            "--tqdm",
            "--batch_size",
            "8192",
            "--num_neighs",
            "100",
            "100",
            "--loader_num_workers",
            "0",
            "--seed",
            "2",
            "--reverse_mp",
            "--ego",
            "--ports",
            "--emlps",
            "--tds",
            "--correct_reverse_edge_features",
            "--testing",
            "--unique_name",
            arm,
        ]
    )
    args.checkpoint_suffix = suffix
    args.embeddings_dir = str(_ROOT / "embeddings")
    args.embeddings_subdir = emb_subdir
    args.representation_source = "pre_embedding_3h"
    args.extract_splits = ",".join(splits)
    args.random_init = False
    args.finetune = False
    args.loader_num_workers = 0
    t0 = time.perf_counter()
    logging.info(
        "EXTRACT arm=%s ep=%d splits=%s ckpt_sha=%s",
        arm_key,
        ep,
        list(splits),
        _file_sha256(ckpt)[:16],
    )
    with torch.inference_mode():
        run_embedding_extraction(
            tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config
        )
    # run_embedding_extraction nests pre_embedding_3h under emb_subdir
    final = _ROOT / "embeddings" / emb_subdir / "pre_embedding_3h"
    if final != out and final.is_dir():
        # Expected path already matches _emb_dir layout.
        pass
    meta = json.loads((final / "meta.json").read_text())
    meta["valgate"] = {
        "arm_key": arm_key,
        "epoch": ep,
        "checkpoint": str(ckpt),
        "checkpoint_sha256": _file_sha256(ckpt),
        "extract_wall_s": float(time.perf_counter() - t0),
        "splits": list(splits),
        "loader_num_workers": 0,
        "representation_source": "pre_embedding_3h",
    }
    (final / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return final, meta


def _pack_stack(
    z: np.ndarray,
    ids: np.ndarray,
    stack: str,
    x_raw: np.ndarray,
    tf_feat: np.ndarray,
) -> np.ndarray:
    parts = [z]
    if "X" in stack:
        parts.append(x_raw[ids])
    if "TF" in stack:
        parts.append(tf_feat[ids])
    return np.concatenate(parts, axis=1).astype(np.float32)


def _eval_val_only(
    mod,
    *,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Fit on train; score validation only (dummy test = val so helpers stay unchanged)."""
    scaler = StandardScaler()
    x_tr_s = scaler.fit_transform(x_tr).astype(np.float32)
    x_va_s = scaler.transform(x_va).astype(np.float32)
    metrics = mod.train_mlp(
        x_tr_s,
        y_tr,
        x_va_s,
        y_va,
        x_va_s,
        y_va,
        device=device,
        seed=seed,
        focal=False,
    )
    return {
        "val_auprc": float(metrics["val_ranking"]["auprc"]),
        "val_auroc": float(metrics["val_ranking"]["auroc"]),
        "val_f1_at_selected": float(metrics["val_at_selected_threshold"]["f1"]),
        "val_precision_at_selected": float(metrics["val_at_selected_threshold"]["precision"]),
        "val_recall_at_selected": float(metrics["val_at_selected_threshold"]["recall"]),
        "validation_selected_threshold": float(
            metrics["threshold_val_selected"]["validation_selected_threshold"]
        ),
        "learner": "mlp",
        "weight": "none",
    }


def _eval_with_test(
    mod,
    *,
    x_tr,
    y_tr,
    x_va,
    y_va,
    x_te,
    y_te,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    scaler = StandardScaler()
    x_tr_s = scaler.fit_transform(x_tr).astype(np.float32)
    x_va_s = scaler.transform(x_va).astype(np.float32)
    x_te_s = scaler.transform(x_te).astype(np.float32)
    metrics = mod.train_mlp(
        x_tr_s, y_tr, x_va_s, y_va, x_te_s, y_te, device=device, seed=seed, focal=False
    )
    return {
        "val_auprc": float(metrics["val_ranking"]["auprc"]),
        "val_auroc": float(metrics["val_ranking"]["auroc"]),
        "val_f1_at_selected": float(metrics["val_at_selected_threshold"]["f1"]),
        "test_auroc": float(metrics["threshold_0.5"]["auroc"]),
        "test_auprc": float(metrics["threshold_0.5"]["auprc"]),
        "test_f1_0.5": float(metrics["threshold_0.5"]["f1"]),
        "test_p_0.5": float(metrics["threshold_0.5"]["precision"]),
        "test_r_0.5": float(metrics["threshold_0.5"]["recall"]),
        "test_f1_val_thr": float(metrics["threshold_val_selected"]["f1"]),
        "test_p_val_thr": float(metrics["threshold_val_selected"]["precision"]),
        "test_r_val_thr": float(metrics["threshold_val_selected"]["recall"]),
        "validation_selected_threshold": float(
            metrics["threshold_val_selected"]["validation_selected_threshold"]
        ),
        "p_at_100": float(metrics["threshold_0.5"]["precision_at_100"]),
        "p_at_500": float(metrics["threshold_0.5"]["precision_at_500"]),
        "p_at_1000": float(metrics["threshold_0.5"]["precision_at_1000"]),
        "ppr_0.5": float(metrics["threshold_0.5"]["positive_prediction_rate"]),
        "tp_0.5": float(metrics["threshold_0.5"]["tp"]),
        "fp_0.5": float(metrics["threshold_0.5"]["fp"]),
        "fn_0.5": float(metrics["threshold_0.5"]["fn"]),
        "tn_0.5": float(metrics["threshold_0.5"]["tn"]),
        "learner": "mlp",
        "weight": "none",
    }


def _select_row(rows: List[Dict[str, Any]], arm_key: str) -> Optional[Dict[str, Any]]:
    cand = [r for r in rows if r["arm_key"] == arm_key and r["stack"] == "H+X+TF"]
    if not cand:
        return None
    cand.sort(key=lambda r: (-r["val_auprc"], -r["val_f1_at_selected"], -r["epoch"]))
    return cand[0]


def _write_notes(path: Path, payload: Dict[str, Any]) -> None:
    diag = payload.get("timeout_diagnosis", {})
    val_rows = [r for r in payload.get("val_rows", []) if r["stack"] == "H+X+TF" or r["stack"] == "H"]
    lines = [
        "# Edge D+ neighbor-positive 10ep scout",
        "",
        "**NOT an exact GCPAL reproduction.** Matched identity poscomplete control required.",
        "",
        "## Training (no retrain in this gate)",
        "",
        "| Arm | Job | Elapsed | Status |",
        "|-----|-----|---------|--------|",
        "| Identity poscomplete control | **18719614** | 21m42s | COMPLETED |",
        "| Neighbor SupCon | **18719615** | 36m49s | COMPLETED |",
        f"| Val-gate extract+eval | **{payload.get('valgate_job_id', 'pending')}** | "
        f"{payload.get('valgate_elapsed', 'n/a')} | {payload.get('valgate_status', 'RUNNING')} |",
        "",
        "Smoke (passed): job **18719182**. Failed evaluator: job **18719616** TIMEOUT.",
        "",
        "## Timeout root cause (job 18719616)",
        "",
        diag.get(
            "summary",
            "See diagnostics JSON.",
        ),
        "",
        "## SSL train loss (do **not** use for checkpoint selection)",
        "",
        "| Epoch | Identity loss | Neighbor loss |",
        "|------:|-------------:|--------------:|",
        "| 1 | 5.940 | 6.049 |",
        "| 3 | 5.778 | 5.923 |",
        "| 5 | 5.733 | 5.909 |",
        "| 10 | 5.708 | 5.901 |",
        "",
        "Epoch 10 is the **latest / lowest training-loss** checkpoint for both arms — not a validation-selected best.",
        "",
        "## Validation decision table (pre-3h, MLP, temporal train→val)",
        "",
        "| arm | epoch | H val AUPRC | H+X+TF val AUPRC | val F1 |",
        "|-----|------:|------------:|-----------------:|-------:|",
    ]
    # Pivot by arm/epoch
    by = {}
    for r in payload.get("val_rows", []):
        key = (r["arm_key"], r["epoch"])
        by.setdefault(key, {})[r["stack"]] = r
    for (arm_key, ep) in sorted(by.keys(), key=lambda t: (t[0], t[1])):
        h = by[(arm_key, ep)].get("H", {})
        hx = by[(arm_key, ep)].get("H+X+TF", {})
        lines.append(
            f"| {arm_key} | {ep} | "
            f"{h.get('val_auprc', float('nan')):.4f} | "
            f"{hx.get('val_auprc', float('nan')):.4f} | "
            f"{hx.get('val_f1_at_selected', float('nan')):.4f} |"
        )
    sel_i = payload.get("selected_identity")
    sel_n = payload.get("selected_neighbor")
    winner = payload.get("winner")
    lines += [
        "",
        "Selection rule: max **H+X+TF** temporal validation AUPRC; validation F1 secondary; never SSL loss; never test.",
        "",
        f"- Identity selected: `{sel_i['tag'] if sel_i else None}` "
        f"(val AUPRC={sel_i['val_auprc'] if sel_i else None})",
        f"- Neighbor selected: `{sel_n['tag'] if sel_n else None}` "
        f"(val AUPRC={sel_n['val_auprc'] if sel_n else None})",
        f"- Winner (paired comparison): `{winner['tag'] if winner else None}`",
        f"- Neighbor beats matched identity: **{payload.get('neighbor_beats_identity')}**",
        f"- Reference D+ fullstack val AUPRC: **{REF_VAL_AUPRC}** (40ep horizon; unequal batching — contextual only)",
        f"- Recommend 40ep continuation: **{payload.get('recommend_40ep')}** — {payload.get('recommend_40ep_rationale')}",
        f"- Automatic 40ep submitted: **False**",
        f"- GNN retrained in this gate: **False**",
        "",
    ]
    if payload.get("test_rows"):
        lines.append("## Winner-only paired test (locked after validation)")
        lines.append("")
        for r in payload["test_rows"]:
            lines += [
                f"### `{r['tag']}`",
                f"- AUROC/AUPRC: {r['test_auroc']:.4f} / {r['test_auprc']:.4f}",
                f"- F1@0.5 P/R: {r['test_f1_0.5']:.4f} / {r['test_p_0.5']:.4f} / {r['test_r_0.5']:.4f}",
                f"- F1@val-thr P/R: {r['test_f1_val_thr']:.4f} / {r['test_p_val_thr']:.4f} / {r['test_r_val_thr']:.4f}",
                f"- PPR@0.5: {r['ppr_0.5']:.4f}; TP/FP/FN/TN: "
                f"{int(r['tp_0.5'])}/{int(r['fp_0.5'])}/{int(r['fn_0.5'])}/{int(r['tn_0.5'])}",
                f"- P@100/500/1000: {r['p_at_100']:.3f}/{r['p_at_500']:.3f}/{r['p_at_1000']:.3f}",
                "",
            ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    logger_setup()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--profile_batches", type=int, default=8)
    p.add_argument("--skip_profile", action="store_true")
    p.add_argument("--profile_only", action="store_true")
    p.add_argument(
        "--epochs",
        type=str,
        default="5,10",
        help="Comma-separated epochs for the first validation pass.",
    )
    p.add_argument(
        "--arms",
        type=str,
        default="identity,neighbor",
        help="Comma-separated arm keys.",
    )
    p.add_argument("--max_projected_hours", type=float, default=4.0)
    p.add_argument(
        "--output_json",
        default="results/diagnostics/edge_dplus_neighbor_positive_10ep_seed2.json",
    )
    p.add_argument(
        "--output_md",
        default="notes/edge_dplus_neighbor_positive_10ep_seed2.md",
    )
    p.add_argument("--run_test", action="store_true", help="After val selection, extract+eval test for winners.")
    p.add_argument("--force_early_epochs", action="store_true", help="Also evaluate epochs 1 and 3.")
    args_ns = p.parse_args()

    set_seed(args_ns.seed)
    mod = _load_challenge_mod()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    timeout_diagnosis = {
        "failed_job": 18719616,
        "successful_reference_extract": 18558352,
        "successful_fullstack_eval": 18678029,
        "summary": (
            "Job 18719616 never finished a single extraction: after ~8 min ports/TDS load and "
            "checkpoint load, `extract hetero` stayed at 0/397 for ~5h50m until the 6h TIME_LIMIT. "
            "The child command used `--loader_num_workers 8` (persistent_workers=True). "
            "`embedding_extraction.py` previously called `next(iter(tr_loader))` for model "
            "construction before CUDA init, then re-iterated the same multi-worker loader — a "
            "classic CUDA+fork / persistent-worker deadlock on the first batch. "
            "Healthy seed-edge pre-3h extraction (job 18558352) completes train (~397 batches) in "
            "~2 minutes at ~3 it/s; the timeout was not 'too many checkpoints' alone. "
            "Secondary waste: parent eval reloaded data without ports, then each subprocess "
            "reloaded ports/TDS (~8 min) and planned arms×epochs×(post+pre)×(train/val/test). "
            "Empty dir `embeddings/edge_dplus_identity_poscomplete_10ep_seed2_ep01/` has no valid cache. "
            "Fix: single-process data load, loader_num_workers=0, fresh loaders after sample batch, "
            "pre-3h train+val only for ep 5/10, join X+TF after one H extract."
        ),
        "not_causes": [
            "seed-edge-at-a-time extraction (batch_size=8192)",
            "missing model.eval/no_grad (both present; hang was before first batch)",
        ],
        "partial_cache_valid": False,
    }

    with open(_ROOT / "data_config.json", encoding="utf-8") as f:
        data_config = json.load(f)

    # Load X / TF once (row-aligned by global edge id = CSV row index).
    df, df_train, tr_ids, va_ids, te_ids, spec = load_dataset_frames("Small-HI", str(_ROOT / "data_config.json"))
    y_all = df[spec.label_col].to_numpy().astype(np.int64)
    x_raw, _, _, _ = build_full_feature_matrix(
        df, df_train, ("edge_native",), categorical_encoding="one_hot"
    )
    tf_feat = np.load(TF_CACHE / "features.npy").astype(np.float32)
    tf_ids = np.load(TF_CACHE / "edge_id.npy").astype(np.int64)
    assert tf_feat.shape[0] == len(df)
    assert np.array_equal(tf_ids, np.arange(len(tf_ids))) or tf_feat.shape[0] == len(tf_ids)

    # Graph once with D+ flags.
    graph_args = create_parser().parse_args(
        [
            "--data",
            "Small-HI",
            "--model",
            "gin",
            "--batch_size",
            "8192",
            "--num_neighs",
            "100",
            "100",
            "--loader_num_workers",
            "0",
            "--seed",
            str(args_ns.seed),
            "--reverse_mp",
            "--ego",
            "--ports",
            "--emlps",
            "--tds",
            "--correct_reverse_edge_features",
            "--testing",
            "--unique_name",
            ARMS["identity"],
            "--tqdm",
        ]
    )
    graph_args.checkpoint_suffix = "_ep05"
    graph_args.finetune = False
    graph_args.include_temporal_flow_edge_features = False
    graph_args.embedding_dim = 128
    logging.info("Loading Small-HI graph once (ports+tds)…")
    t_data0 = time.perf_counter()
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(graph_args, data_config)
    data_load_s = time.perf_counter() - t_data0
    logging.info("Retrieved data in %.1fs", data_load_s)

    profile = None
    if not args_ns.skip_profile:
        profile = _profile_batches(
            tr_data,
            val_data,
            te_data,
            tr_inds,
            val_inds,
            te_inds,
            graph_args,
            data_config,
            max_batches=int(args_ns.profile_batches),
        )
        if float(profile["projected_4ckpt_train_val_h"]) > float(args_ns.max_projected_hours):
            raise SystemExit(
                f"Projected 4-ckpt train+val extract "
                f"{profile['projected_4ckpt_train_val_h']:.2f}h exceeds budget "
                f"{args_ns.max_projected_hours}h; aborting before full run."
            )
    if args_ns.profile_only:
        Path(args_ns.output_json).write_text(
            json.dumps(
                {
                    "timeout_diagnosis": timeout_diagnosis,
                    "profile": profile,
                    "data_load_s": data_load_s,
                    "automatic_40ep_submitted": False,
                    "gnn_retrained": False,
                },
                indent=2,
            )
            + "\n"
        )
        logging.info("Profile-only complete")
        return

    epochs = [int(x) for x in args_ns.epochs.split(",") if x.strip()]
    arm_keys = [a.strip() for a in args_ns.arms.split(",") if a.strip()]
    val_rows: List[Dict[str, Any]] = []
    extract_meta: Dict[str, Any] = {}

    def run_epoch_set(eps: Sequence[int]) -> None:
        for arm_key in arm_keys:
            for ep in eps:
                emb_path, meta = _ensure_extract(
                    arm_key=arm_key,
                    ep=ep,
                    splits=("train", "val"),
                    tr_data=tr_data,
                    val_data=val_data,
                    te_data=te_data,
                    tr_inds=tr_inds,
                    val_inds=val_inds,
                    te_inds=te_inds,
                    data_config=data_config,
                )
                extract_meta[f"{arm_key}_ep{ep:02d}"] = meta.get("valgate", meta)
                splits = {}
                for sp, ids_ref in (("train", tr_ids), ("val", va_ids)):
                    z, y, ids = load_embedding_npz(emb_path / f"{sp}.npz")
                    assert np.array_equal(y, y_all[ids])
                    # Coverage vs temporal split ids
                    cov = float(np.isin(ids_ref, ids).mean()) if len(ids_ref) else 0.0
                    logging.info(
                        "Coverage %s ep%02d %s: emb=%d ref=%d overlap=%.4f",
                        arm_key,
                        ep,
                        sp,
                        len(ids),
                        len(ids_ref),
                        cov,
                    )
                    splits[sp] = {"z": z, "y": y, "ids": ids}
                # One H extract → both stacks
                for stack in ("H", "H+X+TF"):
                    x_tr = _pack_stack(
                        splits["train"]["z"], splits["train"]["ids"], stack, x_raw, tf_feat
                    )
                    x_va = _pack_stack(
                        splits["val"]["z"], splits["val"]["ids"], stack, x_raw, tf_feat
                    )
                    metrics = _eval_val_only(
                        mod,
                        x_tr=x_tr,
                        y_tr=splits["train"]["y"],
                        x_va=x_va,
                        y_va=splits["val"]["y"],
                        seed=args_ns.seed,
                        device=device,
                    )
                    row = {
                        "arm_key": arm_key,
                        "arm": ARMS[arm_key],
                        "epoch": ep,
                        "representation": "pre3h",
                        "stack": stack,
                        **metrics,
                        "emb_dir": str(emb_path),
                        "tag": f"{arm_key}|pre3h|ep{ep:02d}|{stack}|mlp|none",
                    }
                    val_rows.append(row)
                    logging.info(
                        "VAL %s auprc=%.4f f1=%.4f",
                        row["tag"],
                        row["val_auprc"],
                        row["val_f1_at_selected"],
                    )

    run_epoch_set(epochs)

    # Early epochs only if ep5 vs ep10 crossover / unresolved trend on H+X+TF
    need_early = bool(args_ns.force_early_epochs)
    if not need_early and set(epochs) >= {5, 10}:
        for arm_key in arm_keys:
            r5 = next(
                (
                    r
                    for r in val_rows
                    if r["arm_key"] == arm_key and r["epoch"] == 5 and r["stack"] == "H+X+TF"
                ),
                None,
            )
            r10 = next(
                (
                    r
                    for r in val_rows
                    if r["arm_key"] == arm_key and r["epoch"] == 10 and r["stack"] == "H+X+TF"
                ),
                None,
            )
            if r5 and r10:
                delta = float(r10["val_auprc"]) - float(r5["val_auprc"])
                # Crossover between arms or strong opposing trends
                if abs(delta) < 0.005:
                    need_early = True
                    logging.info("Unresolved ep5/ep10 trend for %s (Δ=%.4f); will eval 1,3", arm_key, delta)
        # Cross-arm crossover: neighbor ahead at 5 but behind at 10 or vice versa
        def _a(arm, ep):
            return next(
                r["val_auprc"]
                for r in val_rows
                if r["arm_key"] == arm and r["epoch"] == ep and r["stack"] == "H+X+TF"
            )

        if set(arm_keys) >= {"identity", "neighbor"} and set(epochs) >= {5, 10}:
            lead5 = _a("neighbor", 5) - _a("identity", 5)
            lead10 = _a("neighbor", 10) - _a("identity", 10)
            if lead5 * lead10 < 0:
                need_early = True
                logging.info("Arm crossover between ep5 and ep10; will eval 1,3")

    if need_early:
        already = {(r["arm_key"], r["epoch"]) for r in val_rows}
        early = [e for e in (1, 3) if any((a, e) not in already for a in arm_keys)]
        if early:
            run_epoch_set(early)

    sel_i = _select_row(val_rows, "identity")
    sel_n = _select_row(val_rows, "neighbor")
    winner = None
    if sel_i and sel_n:
        winner = sel_n if (
            float(sel_n["val_auprc"]) > float(sel_i["val_auprc"])
            or (
                abs(float(sel_n["val_auprc"]) - float(sel_i["val_auprc"])) < 1e-12
                and float(sel_n["val_f1_at_selected"]) > float(sel_i["val_f1_at_selected"])
            )
        ) else sel_i
    elif sel_n:
        winner = sel_n
    elif sel_i:
        winner = sel_i

    neighbor_beats = bool(
        sel_i
        and sel_n
        and float(sel_n["val_auprc"]) > float(sel_i["val_auprc"]) + 0.005
    )
    recommend_40 = neighbor_beats
    rationale = (
        "Neighbor H+X+TF val AUPRC materially above matched identity (+>0.005); 40ep may be justified"
        if recommend_40
        else "No material val AUPRC gain vs matched identity on this 10ep scout; do not auto-continue"
    )

    test_rows: List[Dict[str, Any]] = []
    if args_ns.run_test and sel_i and sel_n:
        paired = [
            ("identity", int(sel_i["epoch"])),
            ("neighbor", int(sel_n["epoch"])),
        ]
        # Deduplicate if same... always both for paired comparison
        seen = set()
        for arm_key, ep in paired:
            key = (arm_key, ep)
            if key in seen:
                continue
            seen.add(key)
            emb_path, meta = _ensure_extract(
                arm_key=arm_key,
                ep=ep,
                splits=("train", "val", "test"),
                tr_data=tr_data,
                val_data=val_data,
                te_data=te_data,
                tr_inds=tr_inds,
                val_inds=val_inds,
                te_inds=te_inds,
                data_config=data_config,
            )
            extract_meta[f"{arm_key}_ep{ep:02d}_with_test"] = meta.get("valgate", meta)
            splits = {}
            for sp in ("train", "val", "test"):
                z, y, ids = load_embedding_npz(emb_path / f"{sp}.npz")
                splits[sp] = {"z": z, "y": y, "ids": ids}
            stack = "H+X+TF"
            metrics = _eval_with_test(
                mod,
                x_tr=_pack_stack(splits["train"]["z"], splits["train"]["ids"], stack, x_raw, tf_feat),
                y_tr=splits["train"]["y"],
                x_va=_pack_stack(splits["val"]["z"], splits["val"]["ids"], stack, x_raw, tf_feat),
                y_va=splits["val"]["y"],
                x_te=_pack_stack(splits["test"]["z"], splits["test"]["ids"], stack, x_raw, tf_feat),
                y_te=splits["test"]["y"],
                seed=args_ns.seed,
                device=device,
            )
            row = {
                "arm_key": arm_key,
                "arm": ARMS[arm_key],
                "epoch": ep,
                "representation": "pre3h",
                "stack": stack,
                "role": "winner" if winner and winner["arm_key"] == arm_key and winner["epoch"] == ep else "paired_control",
                **metrics,
                "emb_dir": str(emb_path),
                "tag": f"{arm_key}|pre3h|ep{ep:02d}|{stack}|mlp|none",
            }
            test_rows.append(row)
            logging.info(
                "TEST %s auprc=%.4f f1@0.5=%.4f",
                row["tag"],
                row["test_auprc"],
                row["test_f1_0.5"],
            )

    payload = {
        "title": "edge_dplus_neighbor_positive_10ep_seed2",
        "not_exact_gcpal_reproduction": True,
        "matched_identity_control_required": True,
        "selection_rule": "max temporal H+X+TF val AUPRC; val F1 secondary; never SSL loss; never test",
        "reference_dplus_fullstack_val_auprc": REF_VAL_AUPRC,
        "reference_caveat": "40ep D+ horizon and unequal batching vs poscomplete 10ep scout",
        "timeout_diagnosis": timeout_diagnosis,
        "profile": profile,
        "data_load_s": data_load_s,
        "extract_meta": extract_meta,
        "val_rows": val_rows,
        "selected_identity": sel_i,
        "selected_neighbor": sel_n,
        "winner": winner,
        "neighbor_beats_identity": neighbor_beats,
        "recommend_40ep": recommend_40,
        "recommend_40ep_rationale": rationale,
        "automatic_40ep_submitted": False,
        "gnn_retrained": False,
        "test_rows": test_rows,
        "valgate_job_id": os.environ.get("SLURM_JOB_ID"),
        "training_jobs": {"identity": 18719614, "neighbor": 18719615, "failed_eval": 18719616},
        "epoch10_wording": "latest / lowest training loss (not validation-best)",
    }
    out_json = Path(args_ns.output_json)
    out_md = Path(args_ns.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    # Preserve prior interim if present
    if out_json.is_file():
        bak = out_json.with_suffix(out_json.suffix + f".bak_{int(time.time())}")
        bak.write_text(out_json.read_text())
        logging.info("Backed up prior JSON to %s", bak)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    _write_notes(out_md, payload)
    logging.info("Wrote %s and %s", out_json, out_md)


if __name__ == "__main__":
    main()
