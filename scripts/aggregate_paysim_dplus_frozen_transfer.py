#!/usr/bin/env python3
"""Aggregate PaySim frozen D+ transfer role JSONs into final thesis artifacts.

Does not submit jobs. Optionally documents registry rows (no concurrent worker writes).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Repo-root imports (Slurm ``python scripts/...`` does not put ROOT on sys.path by default).
from ranking_metrics import alert_budget_metrics  # noqa: E402

RESULTS_DIR = ROOT / "results" / "diagnostics" / "paysim_dplus_transfer_final"
PROBA_DIR = RESULTS_DIR / "proba"
FINAL_JSON = ROOT / "results" / "diagnostics" / "paysim_dplus_transfer_final.json"
FINAL_MD = ROOT / "notes" / "paysim_dplus_transfer_final.md"


def _mean_std(xs: List[float]) -> Dict[str, Any]:
    xs = [float(x) for x in xs]
    return {
        "mean": float(statistics.mean(xs)),
        "sample_std": float(statistics.stdev(xs)) if len(xs) > 1 else 0.0,
        "median": float(statistics.median(xs)),
        "n": len(xs),
        "values": xs,
    }


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _stack_metric(role: Dict[str, Any], stack: str, *keys, default=None):
    cur = role.get("stacks", {}).get(stack, {})
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _metrics_block(y: np.ndarray, proba: np.ndarray, thr: float) -> Dict[str, float]:
    y = y.astype(np.int64)
    pred = (proba >= float(thr)).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    n = int(y.shape[0])
    out = {
        "auroc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        "auprc": float(average_precision_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "threshold": float(thr),
        "positive_prediction_rate": float(pred.mean()) if n else 0.0,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "n": float(n),
    }
    out.update(alert_budget_metrics(y, proba))
    return out


def _select_threshold_f1(y_val: np.ndarray, proba_val: np.ndarray) -> float:
    """Validation F1 grid search (vectorized counts; safe on ~1e6 rows)."""
    y = y_val.astype(np.int64)
    if len(np.unique(y)) < 2:
        return 0.5
    # Quantile-based candidates keep walltime bounded on large PaySim splits.
    qs = np.linspace(0.01, 0.99, 99)
    thrs = np.unique(np.quantile(proba_val.astype(np.float64), qs))
    best_thr, best_f1 = 0.5, -1.0
    for thr in thrs:
        pred = (proba_val >= float(thr)).astype(np.int64)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 0.0 if (prec + rec) == 0 else (2.0 * prec * rec / (prec + rec))
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr

def equal_weight_ensemble(
    run_tags: List[str],
    stack: str,
) -> Dict[str, Any]:
    """Mean of aligned val/test probabilities on common ID intersection."""
    packs = []
    for tag in run_tags:
        path = PROBA_DIR / tag / f"{stack}_proba.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        packs.append(np.load(path))

    def _common_sorted(arrays: List[np.ndarray]) -> np.ndarray:
        common = arrays[0].astype(np.int64)
        for a in arrays[1:]:
            common = np.intersect1d(common, a.astype(np.int64), assume_unique=False)
        return np.sort(common)

    common_val_arr = _common_sorted([p["val_ids"] for p in packs])
    common_te_arr = _common_sorted([p["test_ids"] for p in packs])

    def align(pack, split: str, common: np.ndarray):
        ids = pack[f"{split}_ids"].astype(np.int64)
        proba = pack[f"{split}_proba"].astype(np.float64)
        y = pack[f"{split}_y"].astype(np.int64)
        # Stable align via searchsorted on sorted unique ids; IDs may be unsorted in npz.
        order = np.argsort(ids)
        ids_s = ids[order]
        # Deduplicate if needed (keep first)
        if ids_s.size and np.any(ids_s[1:] == ids_s[:-1]):
            uniq, first = np.unique(ids_s, return_index=True)
            ids_s = uniq
            order = order[first]
        idx_s = np.searchsorted(ids_s, common)
        if np.any(idx_s >= ids_s.size) or not np.array_equal(ids_s[idx_s], common):
            raise RuntimeError(f"Failed to align {split} IDs for ensemble")
        idx = order[idx_s]
        return proba[idx], y[idx]

    val_probs = []
    te_probs = []
    y_va = None
    y_te = None
    for p in packs:
        pv, yv = align(p, "val", common_val_arr)
        pt, yt = align(p, "test", common_te_arr)
        val_probs.append(pv)
        te_probs.append(pt)
        if y_va is None:
            y_va, y_te = yv, yt
        else:
            if not np.array_equal(y_va, yv) or not np.array_equal(y_te, yt):
                raise RuntimeError("Label mismatch on ensemble ID intersection")

    p_va = np.mean(np.stack(val_probs, axis=0), axis=0)
    p_te = np.mean(np.stack(te_probs, axis=0), axis=0)
    thr = _select_threshold_f1(y_va, p_va)
    return {
        "stack": stack,
        "run_tags": run_tags,
        "weights": "equal",
        "learned_weights": False,
        "logit_average": False,
        "n_val_intersection": int(len(common_val_arr)),
        "n_test_intersection": int(len(common_te_arr)),
        "val_ids_sha256": hashlib.sha256(common_val_arr.tobytes()).hexdigest(),
        "test_ids_sha256": hashlib.sha256(common_te_arr.tobytes()).hexdigest(),
        "n_test_positives": int(y_te.sum()),
        "validation_selected_threshold": float(thr),
        "val_ranking": {
            "auprc": float(average_precision_score(y_va, p_va)),
            "auroc": float(roc_auc_score(y_va, p_va)),
        },
        "threshold_0.5": _metrics_block(y_te, p_te, 0.5),
        "threshold_val_selected": {
            **_metrics_block(y_te, p_te, thr),
            "validation_selected_threshold": float(thr),
        },
        "classification": (
            "Predeclared equal-weight ensemble of three frozen AML-pretrained D+ transfer models."
            if stack == "pre3h_HxX"
            else "Post-128 H+X equal-weight ensemble (representation sensitivity; not primary)."
        ),
    }


def _fmt_pm(agg: Dict[str, Any], key: str = "mean") -> str:
    return f"{agg['mean']:.4f} ± {agg['sample_std']:.4f}"


def build_answers(payload: Dict[str, Any]) -> Dict[str, Any]:
    primary = payload["primary_pre3h_HxX"]
    x_only = payload.get("x_only") or {}
    pre3h_h = payload.get("ablation_pre3h_H_only") or {}
    post128_hx = payload.get("ablation_post128_HxX") or {}
    post128_h = payload.get("ablation_post128_H_only") or {}
    ens_pre = payload.get("ensemble_pre3h_HxX") or {}
    ens_post = payload.get("ensemble_post128_HxX") or {}
    random = payload.get("control_random") or {}
    ft = payload.get("control_ft_seed2") or {}
    seed2 = payload.get("per_seed", {}).get("seed2", {})

    def _auprc(stack_agg):
        return (stack_agg or {}).get("test_auprc", {}).get("mean")

    x_auprc = x_only.get("test_auprc")
    h_auprc = _auprc(pre3h_h)
    hx_auprc = primary.get("test_auprc", {}).get("mean")
    post_hx_auprc = _auprc(post128_hx)
    rand_h = _stack_metric(random, "pre3h_H_only", "threshold_0.5", "auprc") if random else None
    seed2_h = _stack_metric(seed2, "pre3h_H_only", "threshold_0.5", "auprc") if seed2 else None
    ft_hx = _stack_metric(ft, "pre3h_HxX", "threshold_0.5", "auprc") if ft else None
    seed2_hx = _stack_metric(seed2, "pre3h_HxX", "threshold_0.5", "auprc") if seed2 else None

    transfer_supported = None
    if hx_auprc is not None and x_auprc is not None:
        transfer_supported = bool(hx_auprc > x_auprc)

    return {
        "1_primary_pre3h_HxX_mean_pm_sd": {
            "test_auprc": _fmt_pm(primary["test_auprc"]),
            "test_auroc": _fmt_pm(primary["test_auroc"]),
            "test_f1_0.5": _fmt_pm(primary["test_f1_0.5"]),
            "test_f1_val_thr": _fmt_pm(primary["test_f1_val_thr"]),
            "median_test_auprc": primary["test_auprc"]["median"],
        },
        "2_x_only": x_only,
        "3_pre3h_H_only_outperforms_X_only": (
            None if h_auprc is None or x_auprc is None else bool(h_auprc > x_auprc)
        ),
        "4_pre3h_H_improves_HxX_over_X": (
            None if hx_auprc is None or x_auprc is None else bool(hx_auprc > x_auprc)
        ),
        "5_post128_vs_pre3h": {
            "pre3h_HxX_auprc_mean": hx_auprc,
            "post128_HxX_auprc_mean": post_hx_auprc,
            "post128_better": (
                None
                if hx_auprc is None or post_hx_auprc is None
                else bool(post_hx_auprc > hx_auprc)
            ),
            "note": "Do not select representation using test; post-128 is sensitivity only.",
        },
        "6_seed_variability": {
            "test_auprc_sample_std": primary["test_auprc"]["sample_std"],
            "test_f1_0.5_sample_std": primary["test_f1_0.5"]["sample_std"],
            "per_seed_auprc": primary["test_auprc"]["values"],
        },
        "7_pretrained_H_vs_random_H": {
            "seed2_pre3h_H_auprc": seed2_h,
            "random_pre3h_H_auprc": rand_h,
            "pretrained_better": (
                None if seed2_h is None or rand_h is None else bool(seed2_h > rand_h)
            ),
        },
        "8_pre3h_ensemble": {
            "test_auprc": ens_pre.get("threshold_0.5", {}).get("auprc"),
            "test_f1_0.5": ens_pre.get("threshold_0.5", {}).get("f1"),
            "improves_vs_mean": (
                None
                if ens_pre.get("threshold_0.5", {}).get("auprc") is None or hx_auprc is None
                else bool(ens_pre["threshold_0.5"]["auprc"] > hx_auprc)
            ),
        },
        "9_post128_ensemble": {
            "test_auprc": ens_post.get("threshold_0.5", {}).get("auprc"),
            "test_f1_0.5": ens_post.get("threshold_0.5", {}).get("f1"),
            "improves_vs_post128_mean": (
                None
                if ens_post.get("threshold_0.5", {}).get("auprc") is None or post_hx_auprc is None
                else bool(ens_post["threshold_0.5"]["auprc"] > post_hx_auprc)
            ),
        },
        "10_ft_vs_frozen_seed2": {
            "frozen_seed2_pre3h_HxX_auprc": seed2_hx,
            "ft_seed2_pre3h_HxX_auprc": ft_hx,
            "ft_helps": (
                None if seed2_hx is None or ft_hx is None else bool(ft_hx > seed2_hx)
            ),
            "included_in_primary_aggregate": False,
        },
        "11_cross_dataset_transfer_supported": transfer_supported,
        "12_schema_preprocessing_caveats": (
            "PaySim type→currency/payment slots are schema placeholders, not AML-semantic "
            "equivalence; ports/TDS recomputed on PaySim; train-fit edge z-norm (inductive); "
            "test MP graph includes all edges (Multi-GNN inherent scope); no TF; AML scalers "
            "not transferred."
        ),
        "13_published_comparisons": (
            "None numerically protocol-compatible (FAIL). Methodological PARTIAL only "
            "(Papagei-style frozen probe / GFM narrative). Do not cite historical ~0.866 "
            "ports-only logistic PaySim AUROC as D+ transfer."
        ),
        "14_no_paysim_labels_updated_encoder": True,
        "15_no_automatic_followup_training_submitted": True,
    }


def render_md(payload: Dict[str, Any]) -> str:
    a = payload["answers"]
    primary = payload["primary_pre3h_HxX"]
    lines = [
        "# PaySim frozen D+ transfer — final",
        "",
        "**Protocol:** frozen AMLWorld Small-HI D+ encoders (seeds 1–3) → PaySim dual extract "
        "(pre-3h + post-128) → PaperStyleMLP; primary stack = **pre-3h H+X**; "
        "best epoch by val AUPRC; threshold by val F1; `--train_fit_edge_znorm`.",
        "",
        "## Exact transfer claim",
        "",
        payload["claim_language"],
        "",
        "## Limitations",
        "",
        payload["limitations_language"],
        "",
        "## Primary three-seed pre-3h H+X",
        "",
        "| Metric | Mean ± sample SD | Median |",
        "|--------|-----------------:|-------:|",
        f"| test AUPRC | {_fmt_pm(primary['test_auprc'])} | {primary['test_auprc']['median']:.4f} |",
        f"| test AUROC | {_fmt_pm(primary['test_auroc'])} | {primary['test_auroc']['median']:.4f} |",
        f"| test F1@0.5 | {_fmt_pm(primary['test_f1_0.5'])} | {primary['test_f1_0.5']['median']:.4f} |",
        f"| test F1@val-thr | {_fmt_pm(primary['test_f1_val_thr'])} | {primary['test_f1_val_thr']['median']:.4f} |",
        "",
        "### Thesis table (Markdown)",
        "",
        "```markdown",
        payload["tables"]["primary_md"],
        "```",
        "",
        "### Thesis table (LaTeX)",
        "",
        "```latex",
        payload["tables"]["primary_tex"],
        "```",
        "",
        "## Controls and ablations",
        "",
        f"- X-only: `{json.dumps(payload.get('x_only'), indent=2)}`",
        f"- pre-3h H-only aggregate: `{json.dumps(payload.get('ablation_pre3h_H_only'), indent=2)}`",
        f"- post-128 H+X aggregate: `{json.dumps(payload.get('ablation_post128_HxX'), indent=2)}`",
        f"- Ensembles: pre3h={bool(payload.get('ensemble_pre3h_HxX'))} post128={bool(payload.get('ensemble_post128_HxX'))}",
        "",
        "## Final answers (1–15)",
        "",
    ]
    for k, v in a.items():
        lines.append(f"**{k}:** `{json.dumps(v)}`")
        lines.append("")
    lines.extend(
        [
            "## Registry rows (document for later ingest)",
            "",
            "Worker jobs must not write the thesis registry. Suggested rows after this aggregate:",
            "",
            "```json",
            json.dumps(payload.get("registry_rows_suggested", []), indent=2),
            "```",
            "",
            "Optional append was skipped by default (`--append_registry` not implied) to avoid "
            "concurrent registry writes.",
            "",
        ]
    )
    return "\n".join(lines)


def build_tables(per_seed: Dict[str, Any], primary: Dict[str, Any]) -> Dict[str, str]:
    rows = []
    for seed in ("seed1", "seed2", "seed3"):
        r = per_seed[seed]
        rows.append(
            {
                "seed": seed[-1],
                "auprc": _stack_metric(r, "pre3h_HxX", "threshold_0.5", "auprc"),
                "auroc": _stack_metric(r, "pre3h_HxX", "threshold_0.5", "auroc"),
                "f1": _stack_metric(r, "pre3h_HxX", "threshold_0.5", "f1"),
                "f1_thr": _stack_metric(r, "pre3h_HxX", "threshold_val_selected", "f1"),
            }
        )
    md = [
        "| Seed | Test AUPRC | Test AUROC | F1@0.5 | F1@val-thr |",
        "|-----:|-----------:|-----------:|-------:|-----------:|",
    ]
    for row in rows:
        md.append(
            f"| {row['seed']} | {row['auprc']:.4f} | {row['auroc']:.4f} | "
            f"{row['f1']:.4f} | {row['f1_thr']:.4f} |"
        )
    md.append(
        f"| mean±sd | {_fmt_pm(primary['test_auprc'])} | {_fmt_pm(primary['test_auroc'])} | "
        f"{_fmt_pm(primary['test_f1_0.5'])} | {_fmt_pm(primary['test_f1_val_thr'])} |"
    )
    tex = [
        r"\begin{tabular}{rcccc}",
        r"\toprule",
        r"Seed & Test AUPRC & Test AUROC & F1@0.5 & F1@val-thr \\",
        r"\midrule",
    ]
    for row in rows:
        tex.append(
            f"{row['seed']} & {row['auprc']:.4f} & {row['auroc']:.4f} & "
            f"{row['f1']:.4f} & {row['f1_thr']:.4f} \\\\"
        )
    tex.append(
        f"mean$\\pm$sd & {_fmt_pm(primary['test_auprc'])} & {_fmt_pm(primary['test_auroc'])} & "
        f"{_fmt_pm(primary['test_f1_0.5'])} & {_fmt_pm(primary['test_f1_val_thr'])} \\\\"
    )
    tex.extend([r"\bottomrule", r"\end{tabular}"])
    return {"primary_md": "\n".join(md), "primary_tex": "\n".join(tex)}


def aggregate_stack(per_seed: Dict[str, Any], stack: str) -> Dict[str, Any]:
    def collect(path_keys):
        vals = []
        for seed in ("seed1", "seed2", "seed3"):
            v = _stack_metric(per_seed[seed], stack, *path_keys)
            if v is None:
                raise KeyError(f"missing {stack} {path_keys} on {seed}")
            vals.append(float(v))
        return _mean_std(vals)

    return {
        "stack": stack,
        "test_auprc": collect(("threshold_0.5", "auprc")),
        "test_auroc": collect(("threshold_0.5", "auroc")),
        "test_f1_0.5": collect(("threshold_0.5", "f1")),
        "test_f1_val_thr": collect(("threshold_val_selected", "f1")),
        "val_auprc": collect(("val_ranking", "auprc")),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results_dir", default=str(RESULTS_DIR))
    p.add_argument("--output_json", default=str(FINAL_JSON))
    p.add_argument("--output_md", default=str(FINAL_MD))
    p.add_argument(
        "--append_registry",
        action="store_true",
        help="Reserved; default is document-only registry rows (safer).",
    )
    args = p.parse_args()
    results_dir = Path(args.results_dir)

    required = {
        "seed1": results_dir / "role_seed1.json",
        "seed2": results_dir / "role_seed2.json",
        "seed3": results_dir / "role_seed3.json",
        "random_init": results_dir / "role_random_init.json",
        "ft_seed2": results_dir / "role_ft_seed2.json",
    }
    missing = [str(v) for v in required.values() if not v.is_file()]
    if missing:
        raise SystemExit("Missing role JSONs:\n" + "\n".join(missing))

    per_seed = {k: load_json(required[k]) for k in ("seed1", "seed2", "seed3")}
    random = load_json(required["random_init"])
    ft = load_json(required["ft_seed2"])

    primary = aggregate_stack(per_seed, "pre3h_HxX")
    ablation_h = aggregate_stack(per_seed, "pre3h_H_only")
    ablation_post_h = aggregate_stack(per_seed, "post128_H_only")
    ablation_post_hx = aggregate_stack(per_seed, "post128_HxX")

    x_only = {
        "test_auprc": _stack_metric(per_seed["seed2"], "X_only", "threshold_0.5", "auprc"),
        "test_auroc": _stack_metric(per_seed["seed2"], "X_only", "threshold_0.5", "auroc"),
        "test_f1_0.5": _stack_metric(per_seed["seed2"], "X_only", "threshold_0.5", "f1"),
        "test_f1_val_thr": _stack_metric(per_seed["seed2"], "X_only", "threshold_val_selected", "f1"),
        "source_role": "seed2",
    }

    ens_pre = equal_weight_ensemble(
        [per_seed[s]["run_tag"] for s in ("seed1", "seed2", "seed3")],
        "pre3h_HxX",
    )
    ens_post = equal_weight_ensemble(
        [per_seed[s]["run_tag"] for s in ("seed1", "seed2", "seed3")],
        "post128_HxX",
    )

    tables = build_tables(per_seed, primary)
    payload: Dict[str, Any] = {
        "title": "paysim_dplus_transfer_final",
        "primary_stack": "pre3h_HxX",
        "primary_pre3h_HxX": primary,
        "ablation_pre3h_H_only": ablation_h,
        "ablation_post128_H_only": ablation_post_h,
        "ablation_post128_HxX": ablation_post_hx,
        "x_only": x_only,
        "ensemble_pre3h_HxX": ens_pre,
        "ensemble_post128_HxX": ens_post,
        "control_random": random,
        "control_ft_seed2": ft,
        "per_seed": per_seed,
        "tables": tables,
        "claim_language": (
            "A self-supervised contrastive Multi-GIN encoder (D+) pretrained on AMLWorld "
            "Small-HI and evaluated frozen on PaySim with a supervised downstream MLP on "
            "pre-3h H+X (train-fit edge z-norm; ports+tds+emlps+corrected reverse) yields "
            f"test AUPRC {_fmt_pm(primary['test_auprc'])} and F1@0.5 {_fmt_pm(primary['test_f1_0.5'])} "
            "over three encoder seeds. This is frozen encoder transfer with target-graph "
            "structural featurization, not pure feature-space zero-shot and not the historical "
            "ports-only logistic PaySim diagnostic."
        ),
        "limitations_language": (
            "Schema placeholders map PaySim transaction type into AML currency/payment slots "
            "without semantic alignment; ports/TDS are recomputed on PaySim; test message-passing "
            "uses the full timeline graph (Multi-GNN inherent scope); TF deferred; no "
            "protocol-compatible published PaySim numerical baseline; random and AML-supervised "
            "FT encoders are secondary and excluded from the primary mean."
        ),
        "registry_rows_suggested": [
            {
                "family": "paysim_frozen_dplus_transfer",
                "role": "primary_three_seed_pre3h_HxX",
                "metric_summary": _fmt_pm(primary["test_auprc"]),
                "source_json": str(FINAL_JSON),
            }
        ],
        "append_registry_executed": False,
    }
    payload["answers"] = build_answers(payload)

    if args.append_registry:
        payload["append_registry_executed"] = False
        payload["append_registry_note"] = (
            "Safe append not wired automatically; ingest suggested rows via "
            "scripts/build_thesis_experiment_registry.py in a dedicated pass."
        )

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    out_md.write_text(render_md(payload))
    # Also drop table files
    (ROOT / "tables" / "paysim_frozen_dplus_transfer_primary.md").write_text(tables["primary_md"] + "\n")
    (ROOT / "tables" / "paysim_frozen_dplus_transfer_primary.tex").write_text(tables["primary_tex"] + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
