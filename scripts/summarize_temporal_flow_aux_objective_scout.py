#!/usr/bin/env python3
"""Aggregate temporal-flow aux scout probe JSONs into thesis-facing diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VARIANTS = ("tf_bins5_w0.10", "tf_reg_w0.10", "tf_bins10_w0.10", "tf_reg_w0.05")
PRIMARY_ARMS = ("A_embedding", "B_embedding_raw", "D_embedding_raw_temporal_flow")
METRIC_KEYS = (
    "auroc",
    "auprc",
    "f1",
    "precision",
    "recall",
    "precision_at_100",
    "recall_at_100",
    "lift_at_100",
    "precision_at_500",
    "recall_at_500",
    "lift_at_500",
    "precision_at_1000",
    "recall_at_1000",
    "lift_at_1000",
    "recall_at_precision_ge_0.95",
    "recall_at_precision_ge_0.90",
    "recall_at_precision_ge_0.80",
    "recall_at_precision_ge_0.70",
    "threshold_at_precision_ge_0.95",
    "threshold_at_precision_ge_0.90",
    "threshold_at_precision_ge_0.80",
    "threshold_at_precision_ge_0.70",
    "n_alerts_at_precision_ge_0.95",
    "n_alerts_at_precision_ge_0.90",
    "n_alerts_at_precision_ge_0.80",
    "n_alerts_at_precision_ge_0.70",
    "precision_achieved_at_precision_ge_0.95",
    "precision_achieved_at_precision_ge_0.90",
    "precision_achieved_at_precision_ge_0.80",
    "precision_achieved_at_precision_ge_0.70",
)


def _resolve(p: Path) -> Path:
    return p if p.is_absolute() else (ROOT / p)


def _pick_test(arm: Dict[str, Any]) -> Dict[str, Any]:
    t = arm.get("test") or {}
    out = {}
    for k in METRIC_KEYS:
        if k in t:
            out[k] = t[k]
        elif k == "f1" and "f1_at_selected_threshold" in t:
            out[k] = t["f1_at_selected_threshold"]
        elif k == "precision" and "precision_at_selected_threshold" in t:
            out[k] = t["precision_at_selected_threshold"]
        elif k == "recall" and "recall_at_selected_threshold" in t:
            out[k] = t["recall_at_selected_threshold"]
    # common aliases in probe JSON
    for src, dst in (
        ("f1_at_selected_threshold", "f1"),
        ("precision_at_selected_threshold", "precision"),
        ("recall_at_selected_threshold", "recall"),
        ("recall_at_precision_0.9", "recall_at_precision_ge_0.90"),
        ("recall_at_precision_0.8", "recall_at_precision_ge_0.80"),
        ("recall_at_precision_ge_090", "recall_at_precision_ge_0.90"),
        ("recall_at_precision_ge_080", "recall_at_precision_ge_0.80"),
    ):
        if dst not in out and src in t:
            out[dst] = t[src]
    return out


def _load_probe(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _preferred_probe_path(variant: str, rep: str, seed: int) -> Path:
    """Prefer enriched recall-metrics JSON when present; else original scout probe."""
    enriched = _resolve(
        Path(f"results/diagnostics/enriched/tf_aux_{variant}_{rep}_seed{seed}_recall_metrics.json")
    )
    if enriched.is_file():
        return enriched
    return _resolve(Path(f"results/diagnostics/tf_aux_{variant}_{rep}_seed{seed}.json"))


def _ckpt_epoch(run_name: str) -> Optional[int]:
    ckpt = ROOT / "saved-models" / f"checkpoint_{run_name}.tar"
    if not ckpt.is_file():
        return None
    try:
        import torch

        obj = torch.load(ckpt, map_location="cpu")
        for key in ("epoch", "best_epoch", "checkpoint_epoch"):
            if key in obj:
                return int(obj[key])
        hist = obj.get("history") or obj.get("epoch_history")
        if isinstance(hist, list) and hist:
            last = hist[-1]
            if isinstance(last, dict) and "epoch" in last:
                return int(last["epoch"])
        return None
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--baseline_run",
        default="hi_contrastive_gin_emlps_tds_proj_asym_8192neg_queue0_accum4_20ep",
        help="Baseline run name for comparison (seed1 strong recipe).",
    )
    ap.add_argument(
        "--output_json",
        default="results/diagnostics/temporal_flow_aux_objective_scout.json",
    )
    ap.add_argument(
        "--output_md",
        default="notes/temporal_flow_aux_objective_scout.md",
    )
    args = ap.parse_args()

    seed = int(args.seed)
    rows: List[Dict[str, Any]] = []
    pending: List[str] = []

    for variant in VARIANTS:
        run = f"hi_tf_aux_{variant}_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed{seed}"
        post_p = _preferred_probe_path(variant, "post128", seed)
        pre_p = _preferred_probe_path(variant, "pre3h", seed)
        post = _load_probe(post_p)
        pre = _load_probe(pre_p)
        if post is None or pre is None:
            pending.append(variant)
            rows.append(
                {
                    "variant": variant,
                    "run_name": run,
                    "status": "pending",
                    "missing": [str(p) for p in (post_p, pre_p) if not p.is_file()],
                }
            )
            continue
        entry: Dict[str, Any] = {
            "variant": variant,
            "run_name": run,
            "status": "complete",
            "checkpoint_epoch": _ckpt_epoch(run),
            "ssl_labels_used": False,
            "attach_point": "post_embedding_head_pre_projection",
            "post_embedding_128": {},
            "pre_embedding_3h": {},
            "primary_success_evidence": {
                "note": (
                    "Count embedding-only (A) or pre-3h+raw (B) gains vs baseline; "
                    "do not treat only D (+temporal-flow) gains as proof."
                )
            },
        }
        for label, payload in (("post_embedding_128", post), ("pre_embedding_3h", pre)):
            arms = payload.get("arms") or {}
            packed = {}
            for arm in PRIMARY_ARMS:
                if arm in arms:
                    packed[arm] = _pick_test(arms[arm])
            entry[label] = {
                "source_json": str(post_p if label.startswith("post") else pre_p),
                "arms": packed,
            }
        rows.append(entry)

    out = {
        "dataset": "Small-HI",
        "model": "gin",
        "seed": seed,
        "recipe": {
            "contrastive_asymmetric": True,
            "num_neg_samples": 8192,
            "memory_bank_size": 0,
            "temperature": 0.5,
            "graph": ["reverse_mp", "ego", "ports", "emlps", "tds"],
            "epochs": 20,
            "batch_size": 8192,
            "accum_steps": 4,
        },
        "baseline_run": args.baseline_run,
        "ssl_no_label_use": True,
        "variants": rows,
        "pending_variants": pending,
        "success_criterion": (
            "Primary evidence = improve A (embedding-only) or B (pre-3h+raw) over baseline; "
            "D-only gains are not sufficient."
        ),
    }

    out_json = _resolve(Path(args.output_json))
    out_md = _resolve(Path(args.output_md))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    if out_json.is_file():
        try:
            prev = json.loads(out_json.read_text(encoding="utf-8"))
            if isinstance(prev.get("submissions"), dict):
                out["submissions"] = prev["submissions"]
        except Exception:
            pass
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = [
        "# Temporal-flow auxiliary objective scout",
        "",
        f"- Dataset: Small-HI | Model: GIN | Seed: {seed}",
        f"- Baseline run: `{args.baseline_run}`",
        "- SSL aux uses causal `temporal_flow_causal` only; **no labels** in pretraining.",
        "- Attach point: `post_embedding_head_pre_projection` (z_seed / post-128 before projection).",
        "- Primary success: A (embedding-only) or B (pre-3h + raw) vs baseline — not D-only.",
        "",
    ]
    if pending:
        lines.append(f"**Pending variants:** {', '.join(pending)}")
        lines.append("")
    lines.append("| variant | rep | arm | AUROC | AUPRC | F1 | P@100 | R@100 | Lift@100 | ckpt_epoch |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        if row.get("status") != "complete":
            lines.append(f"| {row['variant']} | — | — | pending | | | | | | |")
            continue
        for rep in ("post_embedding_128", "pre_embedding_3h"):
            arms = row[rep]["arms"]
            for arm in PRIMARY_ARMS:
                m = arms.get(arm) or {}
                lines.append(
                    "| {v} | {rep} | {arm} | {auroc} | {auprc} | {f1} | {p100} | {r100} | {l100} | {ep} |".format(
                        v=row["variant"],
                        rep=rep,
                        arm=arm,
                        auroc=_fmt(m.get("auroc")),
                        auprc=_fmt(m.get("auprc")),
                        f1=_fmt(m.get("f1")),
                        p100=_fmt(m.get("precision_at_100")),
                        r100=_fmt(m.get("recall_at_100")),
                        l100=_fmt(m.get("lift_at_100")),
                        ep=row.get("checkpoint_epoch"),
                    )
                )
    lines.extend(
        [
            "",
            f"Full JSON: `{out_json.relative_to(ROOT)}`",
            "",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    if pending:
        print(f"PENDING: {pending}")
        return 2
    return 0


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)


if __name__ == "__main__":
    raise SystemExit(main())
