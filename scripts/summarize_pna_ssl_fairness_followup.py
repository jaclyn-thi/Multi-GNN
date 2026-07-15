#!/usr/bin/env python3
"""Consolidated PNA SSL fairness follow-up summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GIN_JSON = ROOT / "results/diagnostics/architecture_sweep_shared_probe_weights.json"
ORIG_PNA_JSON = ROOT / "results/diagnostics/pre_embedding_3h_vs_post_embedding_pna_emlps_tds_seed1.json"
WIDTH_PNA_JSON = ROOT / "results/diagnostics/pre_embedding_3h_vs_post_embedding_pna_width65_seed1.json"
WIDTH_ALIGNED_PROBE_JSON = ROOT / "results/diagnostics/pna_width_aligned_probe.json"
PARAM_AUDIT = ROOT / "results/diagnostics/pna_width_param_audit.json"
DEG_AUDIT = ROOT / "results/diagnostics/pna_degree_histogram_audit.json"
PARITY_NOTE = "See tests/test_pna_upstream_parity.py results in Slurm log or pytest output."


def _load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _row_from_compare(payload: Dict, *, model: str, hidden: int, pre: int, post: int, params: int, rep: str, stack: str) -> Dict:
    comp = payload["comparisons"][stack]
    side = "pre_embedding_3h" if rep == "pre" else "post_embedding_128"
    t = comp["representations"][side]["test"]
    return {
        "model": model,
        "hidden": hidden,
        "pre_dim": pre,
        "post_dim": post,
        "params": params,
        "representation": rep,
        "stack": stack,
        "auroc": t["auroc"],
        "auprc": t["auprc"],
        "f1": t["f1_at_selected_threshold"],
        "precision_at_100": t.get("precision_at_100"),
        "lift_at_100": t.get("lift_at_100"),
    }


def _gin_row(gin_payload: Dict, stack: str) -> Dict:
    for run in gin_payload["runs"]:
        if run["encoder"] == "gin":
            # architecture sweep is post-only embedding-only probe
            return {
                "model": "GIN reference",
                "hidden": 66,
                "pre_dim": 198,
                "post_dim": 128,
                "params": None,
                "representation": "post",
                "stack": stack,
                "auroc": run["test_auroc"],
                "auprc": run["test_auprc"],
                "f1": run["test_f1"],
                "precision_at_100": None,
                "lift_at_100": None,
            }
    raise KeyError("GIN row missing")


def build_width_aligned_probe_payload(width: Dict[str, Any]) -> Dict[str, Any]:
    """Focused JSON for architecture appendix / registry pending source."""
    emb = width["comparisons"]["embedding_only"]["representations"]
    post = emb["post_embedding_128"]["test"]
    pre = emb["pre_embedding_3h"]["test"]
    dims = width.get("representation_dims") or {}
    return {
        "diagnostic": "pna_width_aligned_probe",
        "run_name": width.get("run_name"),
        "checkpoint_path": width.get("checkpoint_path"),
        "data": width.get("data"),
        "hidden": 65,
        "pre_dim": dims.get("pre_embedding_3h", 195),
        "post_dim": dims.get("post_embedding_128", 128),
        "seed": (width.get("probe") or {}).get("seed", 1),
        "caveats": [
            "single-seed scout (seed 1)",
            "GIN-matched LR/dropout recipe; not a full hyperparameter sweep",
            "main thesis result does not depend on PNA",
        ],
        "embedding_only": {
            "post_embedding_128": post,
            "pre_embedding_3h": pre,
        },
        "embedding_plus_raw": {
            "post_embedding_128": width["comparisons"]["embedding_plus_raw"]["representations"]["post_embedding_128"]["test"],
            "pre_embedding_3h": width["comparisons"]["embedding_plus_raw"]["representations"]["pre_embedding_3h"]["test"],
        },
        "source_json": str(WIDTH_PNA_JSON.relative_to(ROOT)),
    }


def build_table(payload: Dict) -> List[Dict]:
    rows: List[Dict] = []
    pa = _load(PARAM_AUDIT) or {}
    orig_params = next((r["total_params"] for r in pa.get("pna_width_sweep", []) if r["actual_n_hidden"] == 20), None)
    w65_params = next((r["total_params"] for r in pa.get("pna_width_sweep", []) if r["actual_n_hidden"] == 65), None)
    gin_params = pa.get("gin_total_params_reference")

    orig = _load(ORIG_PNA_JSON)
    if orig:
        for stack in ("embedding_only", "embedding_plus_raw"):
            for rep in ("pre", "post"):
                rows.append(_row_from_compare(orig, model="original PNA", hidden=20, pre=60, post=128, params=orig_params or 0, rep=rep, stack=stack))

    width = _load(WIDTH_PNA_JSON)
    if width:
        for stack in ("embedding_only", "embedding_plus_raw"):
            for rep in ("pre", "post"):
                rows.append(_row_from_compare(width, model="width-aligned PNA", hidden=65, pre=195, post=128, params=w65_params or 0, rep=rep, stack=stack))

    gin = _load(GIN_JSON)
    if gin:
        rows.append(_gin_row(gin, "embedding"))
        # GIN +raw from feature ablation not in arch sweep — note in md

    payload_out = {
        "rows": rows,
        "gin_params": gin_params,
        "metadata": {
            "orig_pna_json": str(ORIG_PNA_JSON),
            "width_pna_json": str(WIDTH_PNA_JSON),
            "gin_json": str(GIN_JSON),
        },
    }
    return payload_out


def _md_table(rows: List[Dict]) -> str:
    lines = [
        "| model | hidden | pre dim | post dim | params | representation | stack | AUROC | AUPRC | F1 | P@100 | lift@100 |",
        "|-------|-------:|--------:|---------:|-------:|----------------|-------|------:|------:|---:|------:|---------:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['hidden']} | {r['pre_dim']} | {r['post_dim']} | {r.get('params','—')} | {r['representation']} | {r['stack']} | "
            f"{r['auroc']:.3f} | {r['auprc']:.3f} | {r['f1']:.3f} | "
            f"{r.get('precision_at_100') if r.get('precision_at_100') is not None else '—'} | "
            f"{r.get('lift_at_100') if r.get('lift_at_100') is not None else '—'} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_json", default=str(ROOT / "results/diagnostics/pna_ssl_fairness_followup.json"))
    parser.add_argument("--output_md", default=str(ROOT / "notes/pna_ssl_fairness_followup.md"))
    parser.add_argument("--width_aligned_json", default=str(WIDTH_ALIGNED_PROBE_JSON))
    parser.add_argument("--parity_passed", default="unknown")
    args = parser.parse_args()

    table_payload = build_table({})
    deg = _load(DEG_AUDIT)
    width = _load(WIDTH_PNA_JSON)
    gin = _load(GIN_JSON)
    pa = _load(PARAM_AUDIT) or {}
    gin_params = pa.get("gin_total_params_reference")
    w65_params = next(
        (r["total_params"] for r in pa.get("pna_width_sweep", []) if r.get("actual_n_hidden") == 65),
        None,
    )

    md = [
        "# PNA SSL fairness follow-up — consolidated interpretation",
        "",
        "This note compares three PNA-related baselines against the GIN current-protocol reference. "
        "The width-aligned PNA scout is **one seed** and **not** a full architecture sweep; do not "
        "treat it as proof that PNA is universally better or worse than GIN. The main thesis result "
        "does not depend on PNA.",
        "",
        "## Comparison table",
        "",
        _md_table(table_payload["rows"]),
        "",
        "## Model geometry",
        "",
        "| model | hidden | pre dim | post dim | params | notes |",
        "|-------|-------:|--------:|---------:|-------:|-------|",
        f"| GIN reference | 66 | 198 | 128 | {gin_params or '—'} | architecture-sweep post-128 embedding-only |",
        f"| default PNA | 20 | 60 | 128 | {next((r['total_params'] for r in pa.get('pna_width_sweep', []) if r.get('actual_n_hidden') == 20), '—')} | not capacity/hyperparameter matched |",
        f"| width-aligned PNA | 65 | 195 | 128 | {w65_params or '—'} | GIN-matched LR/dropout; seed 1 scout |",
        "",
        "## Answers",
        "",
        f"1. **Upstream parity:** {args.parity_passed} ({PARITY_NOTE})",
        f"2. **Degree histogram:** {deg.get('classification') if deg else 'audit pending'} — inherited minibatch behavior; see `notes/pna_degree_histogram_audit.md`.",
        "3. **Pre-3h on original 60-d PNA:** expansion 60→128 (not GIN-style compression); see `notes/pre_embedding_3h_vs_post_embedding_pna_emlps_tds_seed1.md`.",
        "4. **Width/hyperparameter alignment:** width65 scout uses GIN-matched LR/dropout; still one seed and not fully tuned PNA.",
        f"5. **Vs GIN:** GIN architecture-sweep post embedding AUPRC = {next((r['test_auprc'] for r in (gin or {}).get('runs', []) if r.get('encoder') == 'gin'), '—') if gin else '—'} (embedding-only).",
        "6. **AUROC vs AUPRC:** compare width-aligned rows in per-run notes; high AUROC with low AUPRC remains possible under imbalance.",
        "7. **Best downstream representation:** per-row winners in per-run notes (`pre` vs `post`, embedding-only vs +raw).",
        "8. **Remaining gap:** conservative read — objective + hyperparameter confounds unless width-aligned scout closes most of the GIN gap without parity failures.",
        "",
        f"JSON: `{args.output_json}`",
        f"Width-aligned probe JSON: `{args.width_aligned_json}`",
    ]

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(table_payload, indent=2), encoding="utf-8")
    Path(args.output_md).write_text("\n".join(md), encoding="utf-8")

    if width:
        wa_path = Path(args.width_aligned_json)
        wa_path.write_text(json.dumps(build_width_aligned_probe_payload(width), indent=2), encoding="utf-8")
        print(f"Wrote {wa_path}")
    else:
        print(f"WARNING: missing {WIDTH_PNA_JSON}; skipped {args.width_aligned_json}")

    print(f"Wrote {out_json} and {args.output_md}")


if __name__ == "__main__":
    main()
