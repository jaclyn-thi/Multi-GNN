#!/usr/bin/env python3
"""Eval edge D+ neighbor-positive 10ep scout: frozen H / H+X+TF + paper MLP.

Selection = temporal validation AUPRC only. NOT exact GCPAL reproduction.
Assumes embeddings already extracted (or --do_extract).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from sklearn.preprocessing import StandardScaler

CHECKPOINT_EPOCHS = (1, 3, 5, 10)
ARMS = (
    "edge_dplus_identity_poscomplete_10ep_seed2",
    "edge_dplus_neighbor_supcon_poscomplete_10ep_seed2",
)
GRAPH_COMMON = [
    "--data", "Small-HI", "--model", "gin", "--tqdm",
    "--batch_size", "8192", "--num_neighs", "100", "100",
    "--loader_num_workers", "8", "--seed", "2",
    "--reverse_mp", "--ego", "--ports", "--emlps", "--tds",
    "--correct_reverse_edge_features", "--testing",
]
TF_DIR = _ROOT / "results/cache/temporal_flow_causal/Small-HI"
REF_VAL_AUPRC = 0.55


def _load_challenge_mod():
    path = _ROOT / "scripts" / "gcpal_challenge_fullstack_eval.py"
    spec = importlib.util.spec_from_file_location("gcpal_challenge_fullstack_eval", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["gcpal_challenge_fullstack_eval"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_npz_split(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(path)
    return d["Z"].astype(np.float32), d["y"].astype(np.int64), d["edge_id"].astype(np.int64)


def _align_to_ids(
    z_map: Dict[int, np.ndarray],
    y_map: Dict[int, int],
    ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep = np.array([int(i) in z_map for i in ids], dtype=bool)
    ids_k = ids[keep]
    z = np.stack([z_map[int(i)] for i in ids_k], axis=0)
    y = np.asarray([y_map[int(i)] for i in ids_k], dtype=np.int64)
    return z, y, ids_k


def extract_if_needed(python: str, arm: str, ep: int, rep: str) -> Path:
    base = _ROOT / "embeddings" / f"{arm}_ep{ep:02d}"
    out = base / "pre_embedding_3h" if rep == "pre3h" else base
    if (out / "val.npz").is_file() and (out / "test.npz").is_file():
        return out
    suffix = f"_ep{ep:02d}"
    ckpt = _ROOT / "saved-models" / f"checkpoint_{arm}{suffix}.tar"
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    cmd = [python, "embedding_extraction.py", *GRAPH_COMMON, "--unique_name", arm]
    cmd += ["--checkpoint_suffix", suffix]
    if rep == "pre3h":
        cmd += [
            "--representation_source", "pre_embedding_3h",
            "--embeddings_subdir", f"{arm}_ep{ep:02d}/pre_embedding_3h",
        ]
    else:
        cmd += ["--embeddings_subdir", f"{arm}_ep{ep:02d}"]
    logging.info("EXTRACT: %s", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(_ROOT))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--python", default="/home/jthi/.conda/envs/multignn/bin/python")
    p.add_argument("--do_extract", action="store_true")
    p.add_argument("--seed", type=int, default=2)
    p.add_argument(
        "--output_json",
        default="results/diagnostics/edge_dplus_neighbor_positive_10ep_seed2.json",
    )
    p.add_argument(
        "--output_md",
        default="notes/edge_dplus_neighbor_positive_10ep_seed2.md",
    )
    args = p.parse_args()
    mod = _load_challenge_mod()

    # Temporal split ids + raw/TF via challenge loader pieces
    from data_loading import get_data
    from util import create_parser, set_seed

    set_seed(args.seed)
    apars = create_parser().parse_args(
        ["--data", "Small-HI", "--model", "gin", "--testing", "--seed", str(args.seed)]
    )
    with open(_ROOT / "data_config.json", encoding="utf-8") as f:
        data_config = json.load(f)
    # get_data returns inds into full edge table
    _tr_data, _va, _te, tr_inds, val_inds, te_inds = get_data(apars, data_config)
    tr = tr_inds.detach().cpu().numpy().astype(np.int64)
    va = val_inds.detach().cpu().numpy().astype(np.int64)
    te = te_inds.detach().cpu().numpy().astype(np.int64)
    # Labels / X / TF from challenge helpers
    y_all = np.load(TF_DIR / "edge_id.npy")  # may only be ids; load features carefully
    tf_feat = np.load(TF_DIR / "features.npy").astype(np.float32)
    tf_ids = np.load(TF_DIR / "edge_id.npy").astype(np.int64)
    if not np.array_equal(tf_ids, np.arange(len(tf_ids))):
        # build dense map
        tf_map = {int(i): tf_feat[j] for j, i in enumerate(tf_ids)}
    else:
        tf_map = None

    # Raw X from formatted csv columns used in challenge — reuse mod if available
    x_raw = None
    if hasattr(mod, "load_raw_edge_features"):
        x_raw = mod.load_raw_edge_features(str(data_config["paths"]["aml_data"]), "Small-HI")
    else:
        # Fall back: one-hot-ish numeric from dataset via probe_feature_ablation
        import pandas as pd
        from dataset_specs import get_dataset_spec

        spec = get_dataset_spec("Small-HI")
        csv = Path(data_config["paths"]["aml_data"]) / "Small-HI" / spec.formatted_csv_name()
        df = pd.read_csv(csv, usecols=list(spec.edge_feature_cols) + [spec.label_col])
        y_full = df[spec.label_col].to_numpy().astype(np.int64)
        x_raw = df[list(spec.edge_feature_cols)].to_numpy().astype(np.float32)
    y_full = None
    if x_raw is not None:
        import pandas as pd
        from dataset_specs import get_dataset_spec

        spec = get_dataset_spec("Small-HI")
        csv = Path(data_config["paths"]["aml_data"]) / "Small-HI" / spec.formatted_csv_name()
        y_full = pd.read_csv(csv, usecols=[spec.label_col])[spec.label_col].to_numpy().astype(np.int64)

    rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        for ep in CHECKPOINT_EPOCHS:
            for rep in ("post128", "pre3h"):
                try:
                    emb_dir = (
                        extract_if_needed(args.python, arm, ep, rep)
                        if args.do_extract
                        else (
                            _ROOT / "embeddings" / f"{arm}_ep{ep:02d}" / "pre_embedding_3h"
                            if rep == "pre3h"
                            else _ROOT / "embeddings" / f"{arm}_ep{ep:02d}"
                        )
                    )
                except FileNotFoundError as e:
                    logging.warning("%s", e)
                    continue
                if not (emb_dir / "val.npz").is_file():
                    logging.warning("Missing embeddings %s", emb_dir)
                    continue
                splits = {}
                for sp in ("train", "val", "test"):
                    z, y, ids = _load_npz_split(emb_dir / f"{sp}.npz")
                    splits[sp] = {"z": z, "y": y, "ids": ids}
                for stack in ("H", "H+X+TF"):
                    def pack(sp: str):
                        z, y, ids = splits[sp]["z"], splits[sp]["y"], splits[sp]["ids"]
                        parts = [z]
                        if "X" in stack:
                            parts.append(x_raw[ids])
                        if "TF" in stack:
                            if tf_map is None:
                                parts.append(tf_feat[ids])
                            else:
                                parts.append(np.stack([tf_map[int(i)] for i in ids], axis=0))
                        return np.concatenate(parts, axis=1).astype(np.float32), y

                    x_tr, y_tr = pack("train")
                    x_va, y_va = pack("val")
                    x_te, y_te = pack("test")
                    scaler = StandardScaler()
                    x_tr_s = scaler.fit_transform(x_tr)
                    x_va_s = scaler.transform(x_va)
                    x_te_s = scaler.transform(x_te)
                    metrics = mod.fit_eval(
                        "mlp",
                        "none",
                        x_tr_s,
                        y_tr,
                        x_va_s,
                        y_va,
                        x_te_s,
                        y_te,
                        seed=args.seed,
                    )
                    val_a = float(metrics["val_ranking"]["auprc"])
                    row = {
                        "arm": arm,
                        "epoch": ep,
                        "representation": rep,
                        "stack": stack,
                        "learner": "mlp",
                        "weight": "none",
                        "val_auprc": val_a,
                        "val_f1_at_selected": float(
                            metrics["val_at_selected_threshold"]["f1"]
                        ),
                        "test_auprc": float(metrics["threshold_0.5"]["auprc"]),
                        "test_auroc": float(metrics["threshold_0.5"]["auroc"]),
                        "test_f1_0.5": float(metrics["threshold_0.5"]["f1"]),
                        "test_f1_val_thr": float(
                            metrics["threshold_val_selected"]["f1"]
                        ),
                        "test_p_0.5": float(metrics["threshold_0.5"]["precision"]),
                        "test_r_0.5": float(metrics["threshold_0.5"]["recall"]),
                        "test_p_val_thr": float(
                            metrics["threshold_val_selected"]["precision"]
                        ),
                        "test_r_val_thr": float(
                            metrics["threshold_val_selected"]["recall"]
                        ),
                        "p_at_100": float(metrics["threshold_0.5"]["precision_at_100"]),
                        "p_at_500": float(metrics["threshold_0.5"]["precision_at_500"]),
                        "p_at_1000": float(metrics["threshold_0.5"]["precision_at_1000"]),
                        "ppr_0.5": float(
                            metrics["threshold_0.5"]["positive_prediction_rate"]
                        ),
                        "tp_0.5": float(metrics["threshold_0.5"]["tp"]),
                        "fp_0.5": float(metrics["threshold_0.5"]["fp"]),
                        "fn_0.5": float(metrics["threshold_0.5"]["fn"]),
                        "tn_0.5": float(metrics["threshold_0.5"]["tn"]),
                        "emb_dir": str(emb_dir),
                        "tag": f"{arm}|{rep}|ep{ep:02d}|{stack}|mlp|none",
                    }
                    rows.append(row)
                    logging.info(
                        "EVAL %s valA=%.4f testF1@0.5=%.4f",
                        row["tag"],
                        val_a,
                        row["test_f1_0.5"],
                    )

    rows.sort(key=lambda r: (-r["val_auprc"], -r["val_f1_at_selected"]))
    selected = rows[0] if rows else None
    neigh = [r for r in rows if "neighbor_supcon" in r["arm"]]
    ident = [r for r in rows if "identity_poscomplete" in r["arm"]]
    best_n = neigh[0] if neigh else None
    best_i = ident[0] if ident else None
    improved = bool(
        best_n
        and best_i
        and float(best_n["val_auprc"]) > float(best_i["val_auprc"]) + 0.005
    )
    payload = {
        "title": "edge_dplus_neighbor_positive_10ep_seed2",
        "not_exact_gcpal_reproduction": True,
        "matched_identity_control_required": True,
        "selection_rule": "max temporal val AUPRC; never test",
        "reference_dplus_fullstack_val_auprc": REF_VAL_AUPRC,
        "n_rows": len(rows),
        "selected": selected,
        "best_neighbor": best_n,
        "best_identity": best_i,
        "recommend_40ep": improved,
        "recommend_40ep_rationale": (
            "Neighbor val AUPRC materially above matched identity (+>0.005)"
            if improved
            else "No material val AUPRC gain vs matched identity; do not auto-continue"
        ),
        "automatic_40ep_submitted": False,
        "rows": rows,
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Edge D+ neighbor-positive 10ep scout",
        "",
        "**NOT an exact GCPAL reproduction.** Poscomplete matched identity control required.",
        "",
        f"- Rows evaluated: {len(rows)}",
        f"- Selected (val AUPRC): `{selected['tag'] if selected else None}` "
        f"valA={selected['val_auprc'] if selected else None}",
        f"- Best neighbor: `{best_n['tag'] if best_n else None}`",
        f"- Best identity: `{best_i['tag'] if best_i else None}`",
        f"- Recommend 40ep: **{improved}** — {payload['recommend_40ep_rationale']}",
        f"- Automatic 40ep submitted: **False**",
        "",
        f"Reference fullstack D+ pre3h H+X+TF val AUPRC: {REF_VAL_AUPRC} (40ep horizon; unequal).",
        "",
    ]
    if selected:
        lines += [
            "## Selected test metrics (after val selection)",
            f"- AUROC/AUPRC: {selected['test_auroc']:.4f} / {selected['test_auprc']:.4f}",
            f"- F1@0.5 / @val-thr: {selected['test_f1_0.5']:.4f} / {selected['test_f1_val_thr']:.4f}",
            f"- P@100/500/1000: {selected['p_at_100']:.3f}/{selected['p_at_500']:.3f}/{selected['p_at_1000']:.3f}",
            "",
        ]
    Path(args.output_md).write_text("\n".join(lines) + "\n")
    logging.info("Wrote %s", args.output_json)


if __name__ == "__main__":
    main()
