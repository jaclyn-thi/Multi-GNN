#!/usr/bin/env python3
"""4-way paired probe: exported embedding dimension 128 vs 198 (Small-LI emb198 scout).

Compares four frozen representations, all paired on a common ``edge_id`` inner-join per split so
every representation is probed on identical rows / labels / order:

  orig_post_128   original seed-1 128-d model, embedding_head output (128-d)
  orig_pre_3h_198 original seed-1 128-d model, pre-embedding (3*n_hidden = 198-d)
  new_post_198    new emb198 model, embedding_head output (198-d)
  new_pre_3h_198  new emb198 model, pre-embedding (198-d)

Same frozen linear-probe pipeline as compare_representation_source.py (class weights, C, val-tuned
threshold, seed). Embedding-only primary; ``--with_raw`` adds a secondary block.

Reported contrasts (each note-worthy effect isolated as far as possible):
  * new_post_198 vs orig_post_128   -> exported-dimension change **conflated with** retraining a
                                       different head (do NOT attribute solely to dimension)
  * new_pre_3h_198 vs new_post_198  -> pre vs post within the new model
  * orig_pre_3h_198 vs orig_post_128-> pre vs post within the original model (replicates prior run)
  * new_pre_3h_198 vs orig_pre_3h_198 -> effect of retraining on the (same-dim) 3h representation
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from linear_probe import load_embedding_npz, resolve_class_weight, serialize_class_weight
from scripts.compare_representation_source import (
    _build_raw_features,
    _probe_one_representation,
)
from util import logger_setup, set_seed

SPLITS = ("train", "val", "test")


def _load(dirpath: Path, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return load_embedding_npz(dirpath / f"{split}.npz")


def _align_common(rep_arrays: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], split: str) -> Dict[str, Any]:
    """Inner-join all representations on a common edge_id set (ascending), verifying label agreement."""
    edge_id_sets = [arrs[2] for arrs in rep_arrays.values()]
    common = edge_id_sets[0]
    for e in edge_id_sets[1:]:
        common = np.intersect1d(common, e)
    aligned_z: Dict[str, np.ndarray] = {}
    ref_y: Optional[np.ndarray] = None
    for name, (z, y, eid) in rep_arrays.items():
        pos = {int(e): i for i, e in enumerate(eid)}
        idx = np.array([pos[int(e)] for e in common], dtype=np.int64)
        aligned_z[name] = z[idx].astype(np.float32)
        y_c = y[idx].astype(np.int64)
        if ref_y is None:
            ref_y = y_c
        elif not np.array_equal(ref_y, y_c):
            raise ValueError(f"{split}: labels disagree across representations for joined edge_ids")
    return {
        "z": aligned_z,
        "y": ref_y,
        "edge_id": common.astype(np.int64),
        "coverage": {
            "joined_rows": int(common.shape[0]),
            "per_rep_rows": {name: int(arrs[2].shape[0]) for name, arrs in rep_arrays.items()},
            "positives": int(ref_y.sum()) if ref_y is not None else 0,
            "positive_rate": float(ref_y.mean()) if ref_y is not None and ref_y.shape[0] else float("nan"),
        },
    }


def _contrast(reps: Dict[str, Any], a: str, b: str, note: str) -> Dict[str, Any]:
    ta = reps[a]["test"]
    tb = reps[b]["test"]
    return {
        "a": a,
        "b": b,
        "note": note,
        "delta_auprc_a_minus_b": float(ta["auprc"] - tb["auprc"]),
        "delta_auroc_a_minus_b": float(ta["auroc"] - tb["auroc"]),
        "delta_f1_a_minus_b": float(ta["f1_at_selected_threshold"] - tb["f1_at_selected_threshold"]),
        "auprc_winner": a if ta["auprc"] > tb["auprc"] else (b if tb["auprc"] > ta["auprc"] else "tie"),
        "f1_winner": a if ta["f1_at_selected_threshold"] > tb["f1_at_selected_threshold"] else (
            b if tb["f1_at_selected_threshold"] > ta["f1_at_selected_threshold"] else "tie"),
    }


def run(args) -> Dict[str, Any]:
    rep_dirs = {
        "orig_post_128": Path(args.orig_post_dir),
        "orig_pre_3h_198": Path(args.orig_pre_dir),
        "new_post_198": Path(args.new_post_dir),
        "new_pre_3h_198": Path(args.new_pre_dir),
    }
    for name, d in rep_dirs.items():
        for split in SPLITS:
            if not (d / f"{split}.npz").is_file():
                raise FileNotFoundError(f"{name}: missing {d / f'{split}.npz'}")

    class_weight = resolve_class_weight(args)

    aligned: Dict[str, Dict[str, Any]] = {}
    edge_ids_by_split: Dict[str, np.ndarray] = {}
    for split in SPLITS:
        rep_arrays = {name: _load(d, split) for name, d in rep_dirs.items()}
        aligned[split] = _align_common(rep_arrays, split)
        edge_ids_by_split[split] = aligned[split]["edge_id"]
        logging.info("split=%s common pairing: %s", split, aligned[split]["coverage"])

    def _feature_block(mode: str, raw_by_split: Optional[Dict[str, np.ndarray]]) -> Dict[str, Any]:
        def _mat(split: str, name: str) -> np.ndarray:
            z = aligned[split]["z"][name]
            if raw_by_split is None:
                return z
            return np.concatenate([z, raw_by_split[split]], axis=1).astype(np.float32)

        reps: Dict[str, Any] = {}
        for name in rep_dirs:
            reps[name] = _probe_one_representation(
                x_train=_mat("train", name), y_train=aligned["train"]["y"],
                x_val=_mat("val", name), y_val=aligned["val"]["y"],
                x_test=_mat("test", name), y_test=aligned["test"]["y"],
                class_weight=class_weight, seed=int(args.seed),
                max_iter=int(args.probe_max_iter), probe_c=float(args.probe_C),
                n_jobs=int(args.probe_n_jobs),
            )
            t = reps[name]["test"]
            logging.info("[%s] %s: dim=%d AUROC=%.4f AUPRC=%.4f F1=%.4f",
                         mode, name, reps[name]["feature_dim"], t["auroc"], t["auprc"],
                         t["f1_at_selected_threshold"])
        contrasts = [
            _contrast(reps, "new_post_198", "orig_post_128",
                      "exported-dim change CONFLATED with retraining a different head"),
            _contrast(reps, "new_pre_3h_198", "new_post_198", "pre vs post within the new emb198 model"),
            _contrast(reps, "orig_pre_3h_198", "orig_post_128", "pre vs post within the original model"),
            _contrast(reps, "new_pre_3h_198", "orig_pre_3h_198",
                      "retraining effect on the same-dim (198) 3h representation"),
        ]
        return {"feature_mode": mode, "representations": reps, "contrasts": contrasts}

    blocks: Dict[str, Any] = {"embedding_only": _feature_block("embedding_only", None)}
    if args.with_raw:
        raw_by_split = _build_raw_features(args.data, args.data_config, edge_ids_by_split)
        blocks["embedding_plus_raw"] = _feature_block("embedding_plus_raw", raw_by_split)

    payload = {
        "diagnostic": "small_li_embedding_dim_128_vs_198",
        "no_ssl_retraining_of_original": True,
        "new_model_is_retrained": True,
        "paired": True,
        "pairing": "common edge_id inner-join across all four representations per split",
        "data": args.data,
        "seed": int(args.seed),
        "representation_dirs": {k: str(v) for k, v in rep_dirs.items()},
        "representation_dims": {
            "orig_post_128": int(aligned["train"]["z"]["orig_post_128"].shape[1]),
            "orig_pre_3h_198": int(aligned["train"]["z"]["orig_pre_3h_198"].shape[1]),
            "new_post_198": int(aligned["train"]["z"]["new_post_198"].shape[1]),
            "new_pre_3h_198": int(aligned["train"]["z"]["new_pre_3h_198"].shape[1]),
        },
        "probe": {
            "impl": "sklearn LogisticRegression (lbfgs)",
            "class_weight_mode": str(args.class_weight),
            "class_weight": serialize_class_weight(class_weight),
            "probe_C": float(args.probe_C),
            "probe_max_iter": int(args.probe_max_iter),
            "seed": int(args.seed),
            "threshold_tuning": "max_f1_on_val",
            "alert_budget_ks": [100, 500, 1000],
        },
        "split_pairing": {s: aligned[s]["coverage"] for s in SPLITS},
        "blocks": blocks,
    }
    for name, d in rep_dirs.items():
        mp = d / "meta.json"
        if mp.is_file():
            with mp.open("r", encoding="utf-8") as f:
                payload.setdefault("extraction_meta", {})[name] = json.load(f)
    return payload


def write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    dims = payload["representation_dims"]
    lines = [
        f"# Small-LI exported embedding dim: 128 vs 198 (emb198 scout)",
        "",
        "Four frozen representations, paired on a common `edge_id` join per split. The original "
        "128-d checkpoint was **not** retrained; the emb198 checkpoint is a **new** training run "
        "(seed 1, 20 ep) that changed only `embedding_dim: 128 → 198`. Because the altered head also "
        "changes SSL optimization, differences are **not** attributable to dimension alone.",
        "",
        f"- dims: orig_post_128={dims['orig_post_128']}, orig_pre_3h={dims['orig_pre_3h_198']}, "
        f"new_post_198={dims['new_post_198']}, new_pre_3h={dims['new_pre_3h_198']}",
        f"- probe: {payload['probe']['impl']}, class_weight={payload['probe']['class_weight_mode']}, "
        f"C={payload['probe']['probe_C']}, threshold={payload['probe']['threshold_tuning']}, "
        f"seed={payload['probe']['seed']}",
        "",
    ]
    for mode, block in payload["blocks"].items():
        lines.append(f"## {mode}")
        lines.append("")
        lines.append("| representation | dim | AUROC | AUPRC | F1@val-thr | F1@0.5 | P@val-thr | R@val-thr | R@1000 | lift@100 |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for name in ("orig_post_128", "orig_pre_3h_198", "new_post_198", "new_pre_3h_198"):
            r = block["representations"][name]
            t = r["test"]
            lines.append(
                f"| {name} | {r['feature_dim']} | {t['auroc']:.4f} | {t['auprc']:.4f} | "
                f"{t['f1_at_selected_threshold']:.4f} | {t['f1_at_threshold_0.5']:.4f} | "
                f"{t['precision_at_selected_threshold']:.4f} | {t['recall_at_selected_threshold']:.4f} | "
                f"{t.get('recall_at_1000', float('nan')):.4f} | {t.get('lift_at_100', float('nan')):.2f} |"
            )
        lines.append("")
        lines.append("**Contrasts (test):**")
        lines.append("")
        for c in block["contrasts"]:
            lines.append(
                f"- `{c['a']}` vs `{c['b']}` ({c['note']}): "
                f"ΔAUPRC={c['delta_auprc_a_minus_b']:+.4f}, ΔAUROC={c['delta_auroc_a_minus_b']:+.4f}, "
                f"ΔF1={c['delta_f1_a_minus_b']:+.4f} → AUPRC winner: **{c['auprc_winner']}**"
            )
        lines.append("")
    lines.extend([
        "## Interpretation guide",
        "",
        "- **Exported-dimension effect** is captured by `new_post_198` vs `orig_post_128`, but this "
        "conflates the wider export with retraining a different head (different SSL optimum).",
        "- **Removing the bottleneck within the new model**: compare `new_pre_3h_198` vs "
        "`new_post_198` (both 198-d) — a learned 198→198 head vs the raw 198-d pre-embedding.",
        "- **Retraining effect on the pre-embedding**: `new_pre_3h_198` vs `orig_pre_3h_198`.",
        "",
        "## Caveats",
        "",
        "- Single seed, single checkpoint per model; development/scout run.",
        "- The emb198 model is a distinct training run; do not attribute any delta solely to the "
        "exported dimension.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="Small-LI")
    p.add_argument("--data_config", default="data_config.json")
    p.add_argument("--orig_post_dir", required=True)
    p.add_argument("--orig_pre_dir", required=True)
    p.add_argument("--new_post_dir", required=True)
    p.add_argument("--new_pre_dir", required=True)
    p.add_argument("--model", default="gin")
    p.add_argument("--class_weight", default="model", choices=["balanced", "none", "model", "explicit"])
    p.add_argument("--class_weight_pos", type=float, default=None)
    p.add_argument("--probe_C", type=float, default=1.0)
    p.add_argument("--probe_max_iter", type=int, default=1000)
    p.add_argument("--probe_n_jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--with_raw", action="store_true")
    p.add_argument("--output_json", required=True)
    p.add_argument("--output_md", required=True)
    p.add_argument("--testing", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger_setup()
    set_seed(args.seed)
    payload = run(args)
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logging.info("Wrote %s", out_json)
    write_markdown(Path(args.output_md), payload)
    logging.info("Wrote %s", args.output_md)


if __name__ == "__main__":
    main()
