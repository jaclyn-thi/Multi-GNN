#!/usr/bin/env python3
"""Build formal three-seed aggregate for paper-faithful Multi-GIN+EU ports 50ep evals.

Uses only formal post-hoc paper_argmax test metrics. Train-time aggregate is referenced
separately and not mixed in.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    1: "small_hi_legacy_supervised_gin_emlps_ports_50ep_seed1",
    2: "small_hi_legacy_supervised_gin_emlps_ports_50ep_seed2",
    3: "small_hi_legacy_supervised_gin_emlps_ports_50ep_seed3",
}
PAPER_MEAN = 0.6479
PAPER_STD = 0.0122
TDS_ON_EVAL = ROOT / "results/diagnostics/eval_small_hi_legacy_supervised_gin_emlps_tds_100ep_seed1.json"
TRAIN_TIME_AGG = ROOT / "results/diagnostics/supervised_Small-HI_ports_50ep_seeds1-3_aggregate.json"
OUT_JSON = ROOT / "results/diagnostics/eval_small_hi_legacy_supervised_gin_emlps_ports_50ep_seeds1-3_formal_aggregate.json"
OUT_MD = ROOT / "notes/eval_small_hi_legacy_supervised_gin_emlps_ports_50ep_seeds1-3_formal_aggregate.md"


def _pred_counts(split: Dict[str, Any]) -> Dict[str, float]:
    """Derive TP/FP/FN/TN-ish counts from n, prevalence, and paper_argmax P/R."""
    n = float(split["n"])
    pos = float(split["positive_rate"]) * n
    neg = n - pos
    pa = split["paper_argmax"]
    precision = float(pa["precision"])
    recall = float(pa["recall"])
    tp = recall * pos
    fn = pos - tp
    if precision > 0:
        pred_pos = tp / precision
    else:
        pred_pos = 0.0
    fp = max(pred_pos - tp, 0.0)
    tn = max(neg - fp, 0.0)
    return {
        "n": n,
        "n_positive": pos,
        "n_negative": neg,
        "predicted_positive": pred_pos,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _seed_row(seed: int, run: str) -> Dict[str, Any]:
    ev = json.loads((ROOT / f"results/diagnostics/eval_{run}.json").read_text())
    train_sum = json.loads(
        (ROOT / f"results/diagnostics/supervised_Small-HI_{run}_summary.json").read_text()
    )
    hist = json.loads(
        (ROOT / f"results/diagnostics/supervised_Small-HI_{run}_epoch_history.json").read_text()
    )["epochs"]
    best_ep = int(ev["checkpoint_epoch"])
    train_time = next(r for r in hist if int(r["epoch"]) == best_ep)
    out_splits = {}
    for split_name in ("train", "val", "test"):
        s = ev["splits"][split_name]
        pa = s["paper_argmax"]
        out_splits[split_name] = {
            "coverage": ev["coverage"][split_name],
            "n": s["n"],
            "positive_rate": s["positive_rate"],
            "auroc": s["auroc"],
            "auprc": s["auprc"],
            "paper_argmax": {
                "f1": pa["f1"],
                "precision": pa["precision"],
                "recall": pa["recall"],
            },
            "prediction_counts_derived": _pred_counts(s),
        }
    return {
        "seed": seed,
        "run_name": run,
        "checkpoint_path": ev["checkpoint_path"],
        "checkpoint_epoch": best_ep,
        "checkpoint_best_validation_f1": ev.get("checkpoint_best_validation_f1"),
        "checkpoint_test_f1_at_selected_epoch_train_time": ev.get(
            "checkpoint_test_f1_at_selected_epoch"
        ),
        "splits": out_splits,
        "comparison_vs_training_time_at_best_val_epoch": {
            "note": "Training-time metrics are diagnostic only; not used in formal aggregate.",
            "train_time_val_f1": train_time["validation_minority_f1_argmax"],
            "train_time_test_f1": train_time["test_minority_f1_argmax"],
            "train_time_test_precision": train_time["test_precision_argmax"],
            "train_time_test_recall": train_time["test_recall_argmax"],
            "train_time_test_auroc": train_time["test_auroc"],
            "train_time_test_auprc": train_time["test_auprc"],
            "formal_test_f1": out_splits["test"]["paper_argmax"]["f1"],
            "formal_test_precision": out_splits["test"]["paper_argmax"]["precision"],
            "formal_test_recall": out_splits["test"]["paper_argmax"]["recall"],
            "formal_test_auroc": out_splits["test"]["auroc"],
            "formal_test_auprc": out_splits["test"]["auprc"],
            "delta_test_f1_formal_minus_train_time": (
                out_splits["test"]["paper_argmax"]["f1"] - train_time["test_minority_f1_argmax"]
            ),
            "summary_best_validation_epoch": train_sum.get("best_validation_epoch"),
        },
    }


def _stats(xs: List[float]) -> Dict[str, float]:
    arr = np.asarray(xs, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std_sample": float(arr.std(ddof=1)),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "range": float(arr.max() - arr.min()),
        "n": int(arr.size),
        "values": [float(x) for x in arr],
    }


def main() -> None:
    seeds = [_seed_row(s, RUNS[s]) for s in (1, 2, 3)]
    test_f1 = [r["splits"]["test"]["paper_argmax"]["f1"] for r in seeds]
    test_p = [r["splits"]["test"]["paper_argmax"]["precision"] for r in seeds]
    test_r = [r["splits"]["test"]["paper_argmax"]["recall"] for r in seeds]
    test_auroc = [r["splits"]["test"]["auroc"] for r in seeds]
    test_auprc = [r["splits"]["test"]["auprc"] for r in seeds]
    f1_stats = _stats(test_f1)

    tds_on = None
    if TDS_ON_EVAL.exists():
        e = json.loads(TDS_ON_EVAL.read_text())
        t = e["splits"]["test"]["paper_argmax"]
        tds_on = {
            "run_name": e["run_name"],
            "checkpoint_epoch": e.get("checkpoint_epoch"),
            "test_paper_argmax_f1": t["f1"],
            "test_precision": t["precision"],
            "test_recall": t["recall"],
            "test_auroc": e["splits"]["test"]["auroc"],
            "test_auprc": e["splits"]["test"]["auprc"],
            "note": "Old TDS-on formal Small-HI GINe+EU seed1 (not paper-matched feature set).",
        }

    mean_close = abs(f1_stats["mean"] - PAPER_MEAN) <= PAPER_STD
    variance_reproduced = f1_stats["std_sample"] <= 1.5 * PAPER_STD

    payload = {
        "title": "Formal post-hoc aggregate: Multi-GIN+EU ports TDS-off 50ep seeds 1–3",
        "protocol": {
            "checkpoint": "checkpoint_best_val_f1.tar",
            "decision_rule": "paper_argmax (argmax over two-class logits)",
            "model_eval": True,
            "tds": False,
            "edge_dim": 6,
            "supervised_head": "legacy",
            "features": ["ports", "ego", "reverse_mp", "emlps"],
            "threshold_tuning_on_test": False,
            "note": "Aggregate uses only formal post-hoc test paper_argmax metrics.",
        },
        "paper_multigin_eu": {"mean": PAPER_MEAN, "std": PAPER_STD},
        "seeds": seeds,
        "formal_test_aggregate": {
            "paper_argmax_f1": f1_stats,
            "precision": _stats(test_p),
            "recall": _stats(test_r),
            "auroc": _stats(test_auroc),
            "auprc": _stats(test_auprc),
            "vs_paper": {
                "delta_mean_f1": f1_stats["mean"] - PAPER_MEAN,
                "mean_within_paper_std": mean_close,
                "sample_std": f1_stats["std_sample"],
                "paper_std": PAPER_STD,
                "low_variance_reproduced": variance_reproduced,
                "mean_reproduced": mean_close,
                "criteria_note": (
                    "mean_reproduced: |mean - paper_mean| <= paper_std; "
                    "low_variance_reproduced: sample_std <= 1.5 * paper_std"
                ),
            },
            "vs_old_tds_on_formal_seed1": tds_on,
        },
        "train_time_aggregate_preserved_separately": {
            "path": str(TRAIN_TIME_AGG.relative_to(ROOT)) if TRAIN_TIME_AGG.exists() else None,
            "note": "Do not mix into formal aggregate; diagnostic only.",
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Formal aggregate: Multi-GIN+EU ports TDS-off 50ep (seeds 1–3)",
        "",
        "Formal post-hoc evaluations of `checkpoint_best_val_f1.tar` with `model.eval()`, "
        "`paper_argmax` decision rule, `tds=False` / edge_dim=6, legacy head, ports/ego/reverse_mp/emlps. "
        "**Test metrics only** enter the aggregate; train-time metrics are diagnostic.",
        "",
        "## Per-seed formal results (paper_argmax)",
        "",
        "| Seed | Epoch | Test F1 | P | R | AUROC | AUPRC | Train cov | Val cov | Test cov |",
        "|------|------:|--------:|--:|--:|------:|------:|----------:|--------:|---------:|",
    ]
    for r in seeds:
        t = r["splits"]["test"]
        pa = t["paper_argmax"]
        lines.append(
            "| {seed} | {ep} | {f1:.4f} | {p:.4f} | {r:.4f} | {auroc:.4f} | {auprc:.4f} | "
            "{trc:.6f} | {vc:.6f} | {tec:.6f} |".format(
                seed=r["seed"],
                ep=r["checkpoint_epoch"],
                f1=pa["f1"],
                p=pa["precision"],
                r=pa["recall"],
                auroc=t["auroc"],
                auprc=t["auprc"],
                trc=r["splits"]["train"]["coverage"]["coverage"],
                vc=r["splits"]["val"]["coverage"]["coverage"],
                tec=t["coverage"]["coverage"],
            )
        )

    lines += [
        "",
        "### Prediction counts (derived from n, prevalence, P/R)",
        "",
        "| Seed | Split | n | n_pos | pred_pos | TP | FP | FN |",
        "|------|-------|--:|------:|---------:|---:|---:|---:|",
    ]
    for r in seeds:
        for split_name in ("train", "val", "test"):
            c = r["splits"][split_name]["prediction_counts_derived"]
            lines.append(
                "| {seed} | {sp} | {n:.0f} | {npos:.1f} | {pp:.1f} | {tp:.1f} | {fp:.1f} | {fn:.1f} |".format(
                    seed=r["seed"],
                    sp=split_name,
                    n=c["n"],
                    npos=c["n_positive"],
                    pp=c["predicted_positive"],
                    tp=c["tp"],
                    fp=c["fp"],
                    fn=c["fn"],
                )
            )

    lines += [
        "",
        "### vs training-time metrics at best-val epoch (diagnostic)",
        "",
        "| Seed | Train-time test F1 | Formal test F1 | Δ |",
        "|------|-------------------:|---------------:|--:|",
    ]
    for r in seeds:
        c = r["comparison_vs_training_time_at_best_val_epoch"]
        lines.append(
            "| {seed} | {tt:.4f} | {ff:.4f} | {d:+.4f} |".format(
                seed=r["seed"],
                tt=c["train_time_test_f1"],
                ff=c["formal_test_f1"],
                d=c["delta_test_f1_formal_minus_train_time"],
            )
        )

    vs = payload["formal_test_aggregate"]["vs_paper"]
    lines += [
        "",
        "## Formal test aggregate (paper_argmax F1 only)",
        "",
        f"- **mean ± sample SD:** {f1_stats['mean']:.4f} ± {f1_stats['std_sample']:.4f}",
        f"- **median / range:** {f1_stats['median']:.4f} / [{f1_stats['min']:.4f}, {f1_stats['max']:.4f}] "
        f"(range={f1_stats['range']:.4f})",
        f"- **Paper Multi-GIN+EU:** {PAPER_MEAN:.4f} ± {PAPER_STD:.4f}",
        f"- **Δ mean vs paper:** {vs['delta_mean_f1']:+.4f}",
        f"- **Mean reproduced?** {'yes' if vs['mean_reproduced'] else 'no'} "
        f"(|Δ|≤paper σ → {mean_close})",
        f"- **Paper low variance reproduced?** {'yes' if vs['low_variance_reproduced'] else 'no'} "
        f"(sample σ={f1_stats['std_sample']:.4f} vs paper σ={PAPER_STD:.4f})",
        "",
    ]
    if tds_on:
        lines += [
            "## vs old TDS-on formal (seed 1)",
            "",
            f"- Run: `{tds_on['run_name']}` @ epoch {tds_on['checkpoint_epoch']}",
            f"- Test paper_argmax F1 **{tds_on['test_paper_argmax_f1']:.4f}** "
            f"(P {tds_on['test_precision']:.4f}, R {tds_on['test_recall']:.4f}, "
            f"AUROC {tds_on['test_auroc']:.4f}, AUPRC {tds_on['test_auprc']:.4f})",
            f"- Formal TDS-off mean F1 − TDS-on: "
            f"**{f1_stats['mean'] - tds_on['test_paper_argmax_f1']:+.4f}**",
            "",
        ]
    lines += [
        "## Artifacts",
        "",
        f"- Aggregate JSON: `{OUT_JSON.relative_to(ROOT)}`",
        f"- Per-seed notes: `notes/eval_{{run}}.md`",
        f"- Train-time diagnostic aggregate (separate): `{TRAIN_TIME_AGG.relative_to(ROOT) if TRAIN_TIME_AGG.exists() else 'n/a'}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(OUT_JSON)
    print(OUT_MD)
    print(json.dumps(payload["formal_test_aggregate"]["paper_argmax_f1"], indent=2))


if __name__ == "__main__":
    main()
