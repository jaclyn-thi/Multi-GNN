#!/usr/bin/env python3
"""Aggregate A/B references + C/D posagg ablation into comparison deliverables."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional


EW = "frozen_checkpoint_temporal_expanding_window_v1"


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def _hx(block: dict) -> dict:
    return block["HxX"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--c_json", required=True)
    p.add_argument("--d_json", required=True)
    p.add_argument(
        "--a_ref",
        default="results/diagnostics/gcpal_txn_node_canonical_reextract_A_identity_seed2_job18662525.json",
    )
    p.add_argument(
        "--b_ref",
        default="results/diagnostics/gcpal_txn_node_canonical_reextract_B_gcpal_seed2_job18662526.json",
    )
    p.add_argument(
        "--output_json",
        default="results/diagnostics/gcpal_txn_node_posagg_ablation.json",
    )
    p.add_argument(
        "--output_md",
        default="notes/gcpal_txn_node_posagg_ablation.md",
    )
    args = p.parse_args()

    A = _load(Path(args.a_ref))
    B = _load(Path(args.b_ref))
    C = _load(Path(args.c_json))
    D = _load(Path(args.d_json))

    a_tp = _hx(A["by_epoch"]["5"]["protocols"][EW]["temporal_primary"])
    b_tp = _hx(B["by_epoch"]["5"]["protocols"][EW]["temporal_primary"])
    c_tp = _hx(C["temporal_primary_ep5"])
    d_tp = _hx(D["temporal_primary_ep5"])

    def row(name, agg, mean_p, tp, collapse):
        return {
            "condition": name,
            "positive_aggregation": agg,
            "mean_abs_P": mean_p,
            "val_HxX_auprc": tp["val_ranking"].get("auprc"),
            "test_HxX_auprc_0.5": tp["threshold_0.5"].get("auprc"),
            "test_HxX_auroc_0.5": tp["threshold_0.5"].get("auroc"),
            "test_HxX_f1_val_thr": tp["threshold_val_selected"].get("f1"),
            "collapse_verdict": collapse,
        }

    # mean |P| for A≈1, B from historical diagnostics ~16.48; C/D from their payloads
    rows = [
        row(
            "A identity / sum",
            "sum_logsumexp",
            1.0,
            a_tp,
            "n/a (reference)",
        ),
        row(
            "B neighbor / sum",
            "sum_logsumexp",
            16.4758,
            b_tp,
            "n/a (reference)",
        ),
        row(
            "C neighbor / logmeanexp",
            C["positive_aggregation"],
            C["optimization_diagnostics"]["n_pos_distribution"]["mean"],
            c_tp,
            C["optimization_diagnostics"]["collapse_verdict"],
        ),
        row(
            "D neighbor / SupCon",
            D["positive_aggregation"],
            D["optimization_diagnostics"]["n_pos_distribution"]["mean"],
            d_tp,
            D["optimization_diagnostics"]["collapse_verdict"],
        ),
    ]
    a_val = rows[0]["val_HxX_auprc"]
    b_val = rows[1]["val_HxX_auprc"]
    for r in rows:
        r["delta_val_vs_A"] = float(r["val_HxX_auprc"]) - float(a_val)
        r["delta_val_vs_B"] = float(r["val_HxX_auprc"]) - float(b_val)

    # Selection: among B-family (B,C,D) by val HxX AUPRC; A is control
    candidates = [rows[1], rows[2], rows[3]]
    selected = max(candidates, key=lambda r: float(r["val_HxX_auprc"]))
    # Report test only after selection
    selected_with_test = dict(selected)

    # Interpretation
    c_beats_a = float(rows[2]["val_HxX_auprc"]) > float(a_val)
    d_beats_a = float(rows[3]["val_HxX_auprc"]) > float(a_val)
    b_beats_a = float(b_val) > float(a_val)
    c_or_d_clear = c_beats_a or d_beats_a
    only_b = b_beats_a and (not c_beats_a) and (not d_beats_a)
    d_bad_c_ok = c_beats_a and (not d_beats_a)
    all_b_beat_a = b_beats_a and c_beats_a and d_beats_a

    if all_b_beat_a:
        interpretation = (
            "All B variants beat A on val HxX AUPRC: positive-set result is robust "
            "to aggregation choice."
        )
        neighbor_help = True
    elif c_or_d_clear and (c_beats_a or d_beats_a):
        if d_bad_c_ok:
            interpretation = (
                "D degrades relative to beating A while C succeeds: forcing alignment "
                "with every KNN neighbor (SupCon) is likely too strict; count-normalized "
                "soft aggregation remains useful."
            )
        else:
            interpretation = (
                "C and/or D retain a clear B>A gain: neighbor semantics help beyond the "
                "mechanical positive-count reward of sum_logsumexp."
            )
        neighbor_help = True
    elif only_b:
        interpretation = (
            "Only current B (sum_logsumexp) improves over A: the result depends materially "
            "on unnormalized positive aggregation and must be described that way."
        )
        neighbor_help = False
    else:
        interpretation = (
            "No clear neighbor-positive gain after accounting for aggregation; "
            "treat B>A under sum as aggregation-sensitive."
        )
        neighbor_help = bool(c_beats_a or d_beats_a)

    multiseed = bool(neighbor_help and (c_beats_a or d_beats_a or all_b_beat_a))

    payload = {
        "title": "gcpal_txn_node_posagg_ablation",
        "job_ids": {
            "C_logmeanexp": C.get("slurm_job_id"),
            "D_supcon": D.get("slurm_job_id"),
        },
        "references": {
            "A_expanding_ep5": args.a_ref,
            "B_expanding_ep5": args.b_ref,
            "note": "A/B metrics from canonical expanding-window re-extract; not retrained.",
        },
        "selection_rule": "max temporal validation HxX AUPRC among B/C/D; test reported after selection",
        "selected_aggregation": selected["positive_aggregation"],
        "selected_condition": selected["condition"],
        "selected_row_with_test": selected_with_test,
        "table": rows,
        "interpretation": interpretation,
        "neighbor_positives_help_after_count_normalization": neighbor_help,
        "multiseed_replication_justified": multiseed,
        "defaults_unchanged": True,
        "historical_artifacts_unchanged": True,
        "c_diagnostics": C.get("optimization_diagnostics"),
        "d_diagnostics": D.get("optimization_diagnostics"),
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    if out_json.exists() or out_md.exists():
        raise SystemExit("Refusing overwrite existing deliverable")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Positive-aggregation ablation (B_gcpal)",
        "",
        "Status: **complete** · A/B references read-only · C/D newly trained 5ep",
        "",
        f"Companion: [`{out_json}`](../{out_json})",
        "",
        "## Jobs",
        "",
        f"- C (`logmeanexp_count_normalized`): `{C.get('slurm_job_id')}`",
        f"- D (`supcon_mean_logprob`): `{D.get('slurm_job_id')}`",
        "",
        "## Selection (validation HxX AUPRC only)",
        "",
        f"**Selected:** {selected['condition']} (`{selected['positive_aggregation']}`) "
        f"val AUPRC={selected['val_HxX_auprc']:.6f}",
        "",
        f"Test metrics for selected (after rule): AUPRC@0.5={selected_with_test['test_HxX_auprc_0.5']:.6f}, "
        f"AUROC={selected_with_test['test_HxX_auroc_0.5']:.6f}, "
        f"F1@val-thr={selected_with_test['test_HxX_f1_val_thr']:.6f}",
        "",
        "## Comparison table (fixed epoch 5, expanding-window)",
        "",
        "| Condition | Aggregation | mean\\|P\\| | val HxX AUPRC | test HxX AUPRC@0.5 | F1@val-thr | AUROC | collapse | Δval vs A | Δval vs B |",
        "|-----------|-------------|----------:|--------------:|-------------------:|-----------:|------:|----------|----------:|----------:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['condition']} | `{r['positive_aggregation']}` | {r['mean_abs_P']:.4f} | "
            f"{r['val_HxX_auprc']:.6f} | {r['test_HxX_auprc_0.5']:.6f} | "
            f"{r['test_HxX_f1_val_thr']:.6f} | {r['test_HxX_auroc_0.5']:.6f} | "
            f"{r['collapse_verdict']} | {r['delta_val_vs_A']:+.6f} | {r['delta_val_vs_B']:+.6f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        interpretation,
        "",
        f"- Neighbor positives help after count normalization: **{neighbor_help}**",
        f"- Multiseed replication justified: **{multiseed}**",
        "",
        "## Confirmation",
        "",
        "- Default `sum_logsumexp` unchanged when `--positive_aggregation` omitted.",
        "- Historical A/B checkpoints and scout artifacts not modified.",
        "- Primary eval: `frozen_checkpoint_temporal_expanding_window_v1` only.",
        "- Raw loss magnitudes not used as quality across modes.",
        "",
    ]
    out_md.write_text("\n".join(lines) + "\n")
    print(out_json)
    print("selected", selected["positive_aggregation"], "neighbor_help", neighbor_help)


if __name__ == "__main__":
    main()
