#!/usr/bin/env python3
"""Merge label-scarcity probe JSONs and write diagnostic + compact tables."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_HI = ROOT / "results/diagnostics/label_scarcity_temporal_flow_hi.json"
DEFAULT_LI = [
    ROOT / f"results/diagnostics/label_scarcity_temporal_flow_li_seed{s}.json"
    for s in (1, 2, 3)
]
OUT_JSON = ROOT / "results/diagnostics/label_scarcity_temporal_flow_probe.json"
OUT_MD = ROOT / "notes/label_scarcity_temporal_flow_probe.md"
OUT_TABLE_MD = ROOT / "tables/label_scarcity_auprc.md"
OUT_TABLE_TEX = ROOT / "tables/label_scarcity_auprc.tex"


def _load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _mean_std(vals: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    xs = [float(v) for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not xs:
        return None, None
    mean = sum(xs) / len(xs)
    if len(xs) == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    return mean, math.sqrt(var)


def _fmt(mean: Optional[float], std: Optional[float] = None, decimals: int = 3) -> str:
    if mean is None:
        return "—"
    s = f"{mean:.{decimals}f}"
    if std is None or abs(std) < 1e-12:
        return s
    return f"{s} ± {std:.{decimals}f}"


def _arm_auprc(payload: Dict[str, Any], fraction: float, arm: str) -> Optional[float]:
    for fr in payload.get("fraction_results") or []:
        if abs(float(fr.get("fraction", -1)) - float(fraction)) < 1e-9:
            test = ((fr.get("arms") or {}).get(arm) or {}).get("test") or {}
            val = test.get("auprc")
            return float(val) if val is not None else None
    return None


def _fractions_union(payloads: Sequence[Dict[str, Any]]) -> List[float]:
    seen = []
    for p in payloads:
        for fr in p.get("fraction_results") or []:
            f = float(fr["fraction"])
            if not any(abs(f - x) < 1e-9 for x in seen):
                seen.append(f)
        for sk in p.get("skipped_fractions") or []:
            f = float(sk["fraction"])
            if not any(abs(f - x) < 1e-9 for x in seen):
                seen.append(f)
    return sorted(seen)


def build_table_rows(
    hi: Optional[Dict[str, Any]],
    li_payloads: Sequence[Dict[str, Any]],
) -> Tuple[List[str], List[List[str]], List[Dict[str, Any]]]:
    headers = [
        "Label fraction",
        "Small-HI pre-3h+raw AUPRC",
        "Small-HI pre-3h+raw+temporal AUPRC",
        "Small-LI pre-3h+raw AUPRC",
        "Small-LI pre-3h+raw+temporal AUPRC",
    ]
    fracs = _fractions_union([p for p in [hi, *li_payloads] if p])
    body: List[List[str]] = []
    structured: List[Dict[str, Any]] = []
    for frac in fracs:
        hi_b = _arm_auprc(hi, frac, "B_embedding_raw") if hi else None
        hi_d = _arm_auprc(hi, frac, "D_embedding_raw_temporal_flow") if hi else None
        li_b_vals = [_arm_auprc(p, frac, "B_embedding_raw") for p in li_payloads]
        li_d_vals = [_arm_auprc(p, frac, "D_embedding_raw_temporal_flow") for p in li_payloads]
        li_b_mean, li_b_std = _mean_std([v for v in li_b_vals if v is not None])
        li_d_mean, li_d_std = _mean_std([v for v in li_d_vals if v is not None])
        row = [
            f"{frac:.0%}" if frac >= 0.01 else f"{frac:.2%}",
            _fmt(hi_b),
            _fmt(hi_d),
            _fmt(li_b_mean, li_b_std),
            _fmt(li_d_mean, li_d_std),
        ]
        # Prefer cleaner percent labels for common fractions
        if abs(frac - 0.01) < 1e-12:
            row[0] = "1%"
        elif abs(frac - 0.05) < 1e-12:
            row[0] = "5%"
        elif abs(frac - 0.10) < 1e-12:
            row[0] = "10%"
        elif abs(frac - 0.25) < 1e-12:
            row[0] = "25%"
        elif abs(frac - 0.50) < 1e-12:
            row[0] = "50%"
        elif abs(frac - 1.0) < 1e-12:
            row[0] = "100%"
        body.append(row)
        structured.append({
            "fraction": frac,
            "small_hi_pre3h_raw_auprc": hi_b,
            "small_hi_pre3h_raw_temporal_auprc": hi_d,
            "small_li_pre3h_raw_auprc_mean": li_b_mean,
            "small_li_pre3h_raw_auprc_std": li_b_std,
            "small_li_pre3h_raw_temporal_auprc_mean": li_d_mean,
            "small_li_pre3h_raw_temporal_auprc_std": li_d_std,
            "small_hi_delta_d_minus_b": (None if hi_b is None or hi_d is None else hi_d - hi_b),
            "small_li_delta_d_minus_b_mean": (
                None if li_b_mean is None or li_d_mean is None else li_d_mean - li_b_mean
            ),
        })
    return headers, body, structured


def _interpret(structured: List[Dict[str, Any]]) -> List[str]:
    notes: List[str] = []
    hi_deltas = [r["small_hi_delta_d_minus_b"] for r in structured if r["small_hi_delta_d_minus_b"] is not None]
    li_deltas = [
        r["small_li_delta_d_minus_b_mean"]
        for r in structured
        if r["small_li_delta_d_minus_b_mean"] is not None
    ]
    if hi_deltas:
        pos = sum(1 for d in hi_deltas if d > 0)
        notes.append(
            f"Small-HI: temporal-flow improved AUPRC in {pos}/{len(hi_deltas)} label fractions "
            f"(mean ΔAUPRC={sum(hi_deltas)/len(hi_deltas):+.3f})."
        )
        low = [r for r in structured if r["fraction"] <= 0.10 and r["small_hi_delta_d_minus_b"] is not None]
        high = [r for r in structured if r["fraction"] >= 0.50 and r["small_hi_delta_d_minus_b"] is not None]
        if low and high:
            low_m = sum(r["small_hi_delta_d_minus_b"] for r in low) / len(low)
            high_m = sum(r["small_hi_delta_d_minus_b"] for r in high) / len(high)
            if low_m > high_m + 0.01:
                notes.append("Small-HI: temporal-flow gain appears larger at low label fractions.")
            elif high_m > low_m + 0.01:
                notes.append("Small-HI: temporal-flow gain appears larger at high label fractions.")
            else:
                notes.append("Small-HI: temporal-flow gain is broadly similar across low vs high label fractions.")
    if li_deltas:
        pos = sum(1 for d in li_deltas if d > 0)
        notes.append(
            f"Small-LI: temporal-flow improved mean AUPRC in {pos}/{len(li_deltas)} label fractions "
            f"(mean ΔAUPRC={sum(li_deltas)/len(li_deltas):+.3f})."
        )
    # Collapse check: very low AUPRC at 1%
    for r in structured:
        if abs(r["fraction"] - 0.01) < 1e-12:
            vals = [
                v for v in (
                    r["small_hi_pre3h_raw_temporal_auprc"],
                    r["small_li_pre3h_raw_temporal_auprc_mean"],
                )
                if v is not None
            ]
            if vals and max(vals) < 0.02:
                notes.append(
                    "At 1% labels, ranking performance is near-collapsed (AUPRC very low); "
                    "interpret low-fraction results cautiously."
                )
            elif vals:
                notes.append(
                    "At 1% labels, performance is degraded vs 100% but not fully collapsed "
                    "on the available datasets."
                )
    if not notes:
        notes.append("Insufficient completed results to interpret temporal-flow under label scarcity.")
    return notes


def _md_table(headers: List[str], body: List[List[str]], notes: List[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    if notes:
        lines.append("")
        lines.append("**Notes:**")
        for n in notes:
            lines.append(f"- {n}")
    return "\n".join(lines)


def _tex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def _tex_table(caption: str, headers: List[str], body: List[List[str]], notes: List[str]) -> str:
    colspec = "l" + "r" * (len(headers) - 1)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{_tex_escape(caption)}}}",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        " & ".join(_tex_escape(h) for h in headers) + " \\\\",
        "\\midrule",
    ]
    for row in body:
        lines.append(" & ".join(_tex_escape(c) if c != "—" else "---" for c in row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    for i, n in enumerate(notes, 1):
        lines.append(f"\\footnotetext[{i}]{{{_tex_escape(n)}}}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def _rel(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    p = path if path.is_absolute() else (ROOT / path)
    p = p.resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def build_payload(
    hi: Optional[Dict[str, Any]],
    li_payloads: List[Dict[str, Any]],
    hi_path: Optional[Path],
    li_paths: List[Path],
) -> Dict[str, Any]:
    headers, body, structured = build_table_rows(hi, li_payloads)
    interpretation = _interpret(structured)
    skipped = []
    for p in [hi, *li_payloads]:
        if not p:
            continue
        for sk in p.get("skipped_fractions") or []:
            skipped.append({
                "data": p.get("data"),
                "run_name": p.get("run_name"),
                **sk,
            })
    return {
        "diagnostic": "label_scarcity_temporal_flow_probe",
        "no_ssl_retraining": True,
        "no_embedding_regeneration": True,
        "caveats": [
            "Downstream-only probe on frozen pre-3h embeddings.",
            "Validation/test sets unchanged; only train labels subsampled.",
            "Small-HI uses one strong run (40ep seed2) × scarcity_seed=1.",
            "Small-LI uses model seeds 1–3 × scarcity_seed=1.",
            "Diagnostic appendix only; not inserted into main thesis tables.",
        ],
        "sources": {
            "small_hi": _rel(hi_path) if hi_path and (hi_path if hi_path.is_absolute() else ROOT / hi_path).is_file() else None,
            "small_li": [_rel(p) for p in li_paths if (p if p.is_absolute() else ROOT / p).is_file()],
        },
        "auprc_table": {
            "headers": headers,
            "rows": body,
            "structured": structured,
        },
        "interpretation": interpretation,
        "skipped_fractions": skipped,
        "small_hi": hi,
        "small_li_by_seed": li_payloads,
    }


def write_outputs(payload: Dict[str, Any], out_json: Path, out_md: Path, out_md_table: Path, out_tex: Path) -> None:
    headers = payload["auprc_table"]["headers"]
    body = payload["auprc_table"]["rows"]
    notes = list(payload.get("interpretation") or [])
    notes.extend([
        "Baseline = pre-3h + raw; +temporal = pre-3h + raw + temporal_flow_causal.",
        "Small-LI values are mean ± sample SD over model seeds 1–3 when available.",
    ])

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    md_table = _md_table(headers, body, notes)
    out_md_table.parent.mkdir(parents=True, exist_ok=True)
    out_md_table.write_text(
        "# Label-scarcity AUPRC\n\n" + md_table + "\n",
        encoding="utf-8",
    )
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text(
        _tex_table("Label-scarcity AUPRC", headers, body, notes) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Label-scarcity temporal-flow probe",
        "",
        "Downstream-only evaluation of frozen `pre-3h + raw` vs "
        "`pre-3h + raw + temporal_flow_causal` under subsampled train labels.",
        "",
        "## Caveats",
        "",
    ]
    for c in payload.get("caveats") or []:
        lines.append(f"- {c}")
    lines.extend(["", "## AUPRC by label fraction", "", md_table, ""])
    if payload.get("skipped_fractions"):
        lines.extend(["## Skipped fractions", ""])
        for sk in payload["skipped_fractions"]:
            lines.append(
                f"- {sk.get('data')} `{sk.get('run_name')}` frac={sk.get('fraction')}: {sk.get('reason')}"
            )
        lines.append("")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hi_json", default=str(DEFAULT_HI))
    ap.add_argument("--li_json", nargs="*", default=[str(p) for p in DEFAULT_LI])
    ap.add_argument("--output_json", default=str(OUT_JSON))
    ap.add_argument("--output_md", default=str(OUT_MD))
    ap.add_argument("--output_table_md", default=str(OUT_TABLE_MD))
    ap.add_argument("--output_table_tex", default=str(OUT_TABLE_TEX))
    args = ap.parse_args()

    hi_path = Path(args.hi_json)
    li_paths = [Path(p) for p in args.li_json]
    hi = _load(hi_path)
    li_payloads = []
    for p in li_paths:
        data = _load(p)
        if data:
            li_payloads.append(data)
    if hi is None and not li_payloads:
        raise FileNotFoundError("No label-scarcity probe JSONs found to summarize.")

    payload = build_payload(hi, li_payloads, hi_path, li_paths)
    write_outputs(
        payload,
        Path(args.output_json),
        Path(args.output_md),
        Path(args.output_table_md),
        Path(args.output_table_tex),
    )
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_table_md}")
    print(f"Wrote {args.output_table_tex}")


if __name__ == "__main__":
    main()
