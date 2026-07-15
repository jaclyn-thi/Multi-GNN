#!/usr/bin/env python3
"""Compare width65 PNA best-stack temporal-flow probe vs GIN main result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PNA_PROBE = ROOT / "results/diagnostics/pna_width65_temporal_flow_probe.json"
PNA_POST_D = ROOT / "results/diagnostics/pna_width65_temporal_flow_post128_armD.json"
GIN_REF = ROOT / "results/diagnostics/temporal_flow_ablation_small_hi_40ep_seed2_maxiter5000.json"
OUT_JSON = ROOT / "results/diagnostics/pna_width65_best_stack_comparison.json"
OUT_MD = ROOT / "notes/pna_width65_best_stack_comparison.md"

CAVEATS = [
    "Downstream-only probe; PNA SSL checkpoint was not retrained.",
    "PNA width65 is a one-seed scout (seed 1, 20ep); not a full architecture sweep.",
    "GIN comparison uses Small-HI 40ep seed2; epochs and seeds are not matched.",
    "Useful for architecture diagnostics, not a definitive architecture ranking.",
]


def _load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _metrics_from_arm(arm: Dict[str, Any]) -> Dict[str, Any]:
    t = arm.get("test") or {}
    conv = arm.get("convergence") or {}
    return {
        "auroc": t.get("auroc"),
        "auprc": t.get("auprc"),
        "f1_at_selected_threshold": t.get("f1_at_selected_threshold"),
        "precision_at_selected_threshold": t.get("precision_at_selected_threshold"),
        "recall_at_selected_threshold": t.get("recall_at_selected_threshold"),
        "selected_threshold": arm.get("selected_threshold"),
        "precision_at_100": t.get("precision_at_100"),
        "recall_at_100": t.get("recall_at_100"),
        "lift_at_100": t.get("lift_at_100"),
        "precision_at_500": t.get("precision_at_500"),
        "recall_at_500": t.get("recall_at_500"),
        "lift_at_500": t.get("lift_at_500"),
        "precision_at_1000": t.get("precision_at_1000"),
        "recall_at_1000": t.get("recall_at_1000"),
        "lift_at_1000": t.get("lift_at_1000"),
        "convergence_status": conv.get("status"),
        "n_iter": conv.get("n_iter"),
        "max_iter": conv.get("max_iter"),
        "test_n": t.get("n"),
        "feature_dim": arm.get("feature_dim"),
    }


def _row(
    label: str,
    *,
    encoder: str,
    run_name: str,
    representation: str,
    stack: str,
    arm: Dict[str, Any],
    source_json: str,
    seed: int,
    training_epochs: int,
    split_pairing: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    m = _metrics_from_arm(arm)
    return {
        "label": label,
        "encoder": encoder,
        "run_name": run_name,
        "representation_source": representation,
        "probe_feature_stack": stack,
        "seed": seed,
        "training_epochs": training_epochs,
        "source_json": source_json,
        "split_pairing": split_pairing,
        **m,
    }


def _delta(a: Dict[str, Any], b: Dict[str, Any], key: str) -> Optional[float]:
    av, bv = a.get(key), b.get(key)
    if av is None or bv is None:
        return None
    return float(bv) - float(av)


def build_payload() -> Dict[str, Any]:
    pna = _load(PNA_PROBE)
    gin = _load(GIN_REF)
    post_d_payload = _load(PNA_POST_D)
    if not pna:
        raise FileNotFoundError(f"Missing PNA probe JSON: {PNA_PROBE}")
    if not gin:
        raise FileNotFoundError(f"Missing GIN reference JSON: {GIN_REF}")

    arms = pna.get("arms") or {}
    gin_arms = gin.get("arms") or {}
    rows: List[Dict[str, Any]] = []

    for arm_key, label, stack in (
        ("A_embedding", "PNA width65 pre-3h", "embedding"),
        ("B_embedding_raw", "PNA width65 pre-3h + raw", "embedding+raw"),
        ("D_embedding_raw_temporal_flow", "PNA width65 pre-3h + raw + temporal_flow_causal", "embedding+raw+temporal_flow_causal"),
    ):
        arm = arms.get(arm_key)
        if not arm:
            raise KeyError(f"Missing arm {arm_key} in {PNA_PROBE}")
        rows.append(_row(
            label,
            encoder="pna",
            run_name=pna["run_name"],
            representation=pna.get("representation", "pre_embedding_3h"),
            stack=stack,
            arm=arm,
            source_json=str(PNA_PROBE.relative_to(ROOT)),
            seed=1,
            training_epochs=20,
            split_pairing=pna.get("split_pairing"),
        ))

    gin_d = gin_arms.get("D_embedding_raw_temporal_flow")
    if not gin_d:
        raise KeyError("Missing GIN arm D in reference JSON")
    rows.append(_row(
        "GIN Small-HI 40ep seed2 pre-3h + raw + temporal_flow_causal",
        encoder="gin",
        run_name=gin["run_name"],
        representation=gin.get("representation", "pre_embedding_3h"),
        stack="embedding+raw+temporal_flow_causal",
        arm=gin_d,
        source_json=str(GIN_REF.relative_to(ROOT)),
        seed=2,
        training_epochs=40,
        split_pairing=gin.get("split_pairing"),
    ))

    post_d_arm = None
    if post_d_payload:
        post_d_arm = (post_d_payload.get("arms") or {}).get("D_embedding_raw_temporal_flow")
    if post_d_arm:
        rows.append(_row(
            "PNA width65 post-128 + raw + temporal_flow_causal",
            encoder="pna",
            run_name=post_d_payload["run_name"],
            representation=post_d_payload.get("representation", "post_embedding_128"),
            stack="embedding+raw+temporal_flow_causal",
            arm=post_d_arm,
            source_json=str(PNA_POST_D.relative_to(ROOT)),
            seed=1,
            training_epochs=20,
            split_pairing=post_d_payload.get("split_pairing"),
        ))

    pna_primary = next(r for r in rows if "pre-3h + raw + temporal_flow" in r["label"] and r["encoder"] == "pna")
    gin_primary = next(r for r in rows if r["encoder"] == "gin")
    comparison = {
        "primary": "PNA width65 pre-3h + raw + temporal_flow_causal vs GIN Small-HI 40ep seed2 (same stack)",
        "delta_auprc_pna_minus_gin": _delta(gin_primary, pna_primary, "auprc"),
        "delta_f1_pna_minus_gin": _delta(gin_primary, pna_primary, "f1_at_selected_threshold"),
        "delta_precision_at_100_pna_minus_gin": _delta(gin_primary, pna_primary, "precision_at_100"),
        "delta_recall_at_100_pna_minus_gin": _delta(gin_primary, pna_primary, "recall_at_100"),
        "delta_lift_at_100_pna_minus_gin": _delta(gin_primary, pna_primary, "lift_at_100"),
    }

    return {
        "diagnostic": "pna_width65_best_stack_comparison",
        "no_ssl_retraining": True,
        "no_embedding_regeneration": True,
        "caveats": CAVEATS,
        "probe_settings": {
            "class_weight": "model (gin weights)",
            "probe_C": 1.0,
            "probe_max_iter": 5000,
            "probe_seed": 1,
            "threshold_tuning": "max_f1_on_val",
            "temporal_flow_cache": "results/cache/temporal_flow_causal/Small-HI",
            "pairing": "exact common edge_id intersections across compared stacks",
        },
        "rows": rows,
        "comparison": comparison,
        "pna_probe_source": str(PNA_PROBE.relative_to(ROOT)),
        "gin_reference_source": str(GIN_REF.relative_to(ROOT)),
    }


def write_markdown(payload: Dict[str, Any], path: Path) -> None:
    lines = [
        "# PNA width65 best-stack comparison (diagnostic)",
        "",
        "Downstream-only probe using validated temporal-flow stack "
        "`pre-3h + raw + temporal_flow_causal`. PNA was **not** retrained.",
        "",
        "## Caveats",
        "",
    ]
    for c in payload.get("caveats", []):
        lines.append(f"- {c}")
    lines.extend(["", "## Compared rows", ""])
    lines.append(
        "| label | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 | conv | n_iter | test_n |"
    )
    lines.append("|-------|------:|------:|---:|------:|------:|---------:|------|-------:|-------:|")
    for r in payload["rows"]:
        lines.append(
            "| {label} | {auroc:.4f} | {auprc:.4f} | {f1:.4f} | {p100:.4f} | {r100:.4f} | {l100:.2f} | {conv} | {n_iter} | {test_n} |".format(
                label=r["label"],
                auroc=float(r.get("auroc") or 0),
                auprc=float(r.get("auprc") or 0),
                f1=float(r.get("f1_at_selected_threshold") or 0),
                p100=float(r.get("precision_at_100") or 0),
                r100=float(r.get("recall_at_100") or 0),
                l100=float(r.get("lift_at_100") or 0),
                conv=r.get("convergence_status") or "—",
                n_iter=r.get("n_iter") or "—",
                test_n=int(r.get("test_n") or 0),
            )
        )
    comp = payload.get("comparison") or {}
    lines.extend([
        "",
        "## Primary comparison (PNA − GIN, best stack)",
        "",
        f"- ΔAUPRC: **{comp.get('delta_auprc_pna_minus_gin', float('nan')):+.4f}**",
        f"- ΔF1: {comp.get('delta_f1_pna_minus_gin', float('nan')):+.4f}",
        f"- ΔP@100: {comp.get('delta_precision_at_100_pna_minus_gin', float('nan')):+.4f}",
        f"- ΔR@100: {comp.get('delta_recall_at_100_pna_minus_gin', float('nan')):+.4f}",
        f"- Δlift@100: {comp.get('delta_lift_at_100_pna_minus_gin', float('nan')):+.2f}",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output_json", default=str(OUT_JSON))
    ap.add_argument("--output_md", default=str(OUT_MD))
    args = ap.parse_args()

    payload = build_payload()
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_markdown(payload, out_md)
    print(f"Wrote {out_json} and {out_md}")


if __name__ == "__main__":
    main()
