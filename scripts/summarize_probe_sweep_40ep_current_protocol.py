#!/usr/bin/env python3
"""Consolidate 40ep targeted probe sweep results + comparison tables."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]

SSL_MODES = ("embedding", "embedding+raw", "embedding+raw+morph")
REFERENCE_ROWS = (
    {
        "label": "GIN 20ep seed1 embedding",
        "features": "embedding",
        "test_f1": 0.2593,
        "test_auprc": 0.2133,
        "test_f1_at_0_5": 0.2573,
        "source": "probe_feature_ablation_current_protocol_gin_emlps_tds_seed1.json",
    },
    {
        "label": "GIN 20ep seed1 embedding+raw+morph",
        "features": "embedding+raw+morph",
        "test_f1": 0.2982,
        "test_auprc": 0.2755,
        "test_f1_at_0_5": 0.3271,
        "source": "probe_feature_ablation_current_protocol_gin_emlps_tds_seed1.json",
    },
    {
        "label": "FNF seed1 embedding+raw+morph",
        "features": "embedding+raw+morph",
        "test_f1": 0.3188,
        "test_auprc": 0.2763,
        "test_f1_at_0_5": 0.3031,
        "source": "probe_feature_ablation_current_protocol_fnf_emlps_tds_seed1.json",
    },
    {
        "label": "FNF seed2 embedding+raw+morph",
        "features": "embedding+raw+morph",
        "test_f1": 0.2624,
        "test_auprc": 0.2425,
        "test_f1_at_0_5": 0.1700,
        "source": "probe_feature_ablation_current_protocol_fnf_emlps_tds_seed2.json",
    },
)


def _load_seed_json(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [c for c in data.get("cells", []) if c.get("status") == "completed"]


def _completed_rows(seed_nums: Sequence[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    diag = ROOT / "results/diagnostics"
    for n in seed_nums:
        final = diag / f"probe_sweep_40ep_seed{n}.json"
        partial = diag / f"probe_sweep_40ep_seed{n}_partial.json"
        cells = _load_seed_json(final) or _load_seed_json(partial)
        rows.extend(cells)
    return rows


def _metric(row: Dict[str, Any], split: str, key: str) -> float:
    return float(row[split][key])


def _row_line(r: Dict[str, Any]) -> str:
    test = r["test"]
    return (
        f"| {r.get('run_label', r.get('run_name'))} | `{r['feature_mode']}` | "
        f"{r['class_weight_policy']} | {r['probe_C']} | "
        f"{test['auroc']:.4f} | {test['auprc']:.4f} | {test['f1']:.4f} | "
        f"{test['f1_at_0_5']:.4f} | {r.get('threshold', test.get('threshold', float('nan'))):.4f} |"
    )


def _top_n(rows: List[Dict[str, Any]], metric: str, n: int = 10) -> List[Dict[str, Any]]:
    ssl = [r for r in rows if r.get("feature_mode") in SSL_MODES]
    return sorted(ssl, key=lambda r: _metric(r, "test", metric), reverse=True)[:n]


def _best_per_seed(rows: List[Dict[str, Any]], metric: str) -> List[Dict[str, Any]]:
    out = []
    for seed in (1, 2, 3, 4):
        run_name = f"gin_40ep_seed{seed}"
        cand = [r for r in rows if r.get("run_name") == run_name and r.get("feature_mode") in SSL_MODES]
        if not cand:
            continue
        out.append(max(cand, key=lambda r: _metric(r, "test", metric)))
    return out


def _shared_setting_stats(
    rows: List[Dict[str, Any]],
    feature_mode: str,
    class_weight_policy: str = "model",
    probe_C: float = 1.0,
) -> Optional[Dict[str, Any]]:
    vals_f1: List[float] = []
    vals_auprc: List[float] = []
    vals_f1_05: List[float] = []
    for seed in (1, 2, 3, 4):
        run_name = f"gin_40ep_seed{seed}"
        match = [
            r
            for r in rows
            if r.get("run_name") == run_name
            and r.get("feature_mode") == feature_mode
            and r.get("class_weight_policy") == class_weight_policy
            and float(r.get("probe_C", -1)) == probe_C
        ]
        if match:
            vals_f1.append(_metric(match[0], "test", "f1"))
            vals_auprc.append(_metric(match[0], "test", "auprc"))
            vals_f1_05.append(_metric(match[0], "test", "f1_at_0_5"))
    if not vals_f1:
        return None

    def _mean_std(xs: List[float]) -> Tuple[float, float]:
        if len(xs) == 1:
            return xs[0], 0.0
        return statistics.mean(xs), statistics.pstdev(xs)

    mf1, sf1 = _mean_std(vals_f1)
    ma, sa = _mean_std(vals_auprc)
    m05, s05 = _mean_std(vals_f1_05)
    return {
        "feature_mode": feature_mode,
        "class_weight_policy": class_weight_policy,
        "probe_C": probe_C,
        "n_seeds": len(vals_f1),
        "test_f1_mean": mf1,
        "test_f1_std": sf1,
        "test_auprc_mean": ma,
        "test_auprc_std": sa,
        "test_f1_at_0_5_mean": m05,
        "test_f1_at_0_5_std": s05,
    }


def _raw_vs_morph_wins(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary = []
    for seed in (1, 2, 3, 4):
        run_name = f"gin_40ep_seed{seed}"
        f1_raw = f1_morph = auprc_raw = 0
        total = 0
        for cw in ("model", "none"):
            for c in (0.1, 1.0, 10.0):
                raw = [
                    r
                    for r in rows
                    if r.get("run_name") == run_name
                    and r.get("feature_mode") == "embedding+raw"
                    and r.get("class_weight_policy") == cw
                    and float(r.get("probe_C")) == c
                ]
                morph = [
                    r
                    for r in rows
                    if r.get("run_name") == run_name
                    and r.get("feature_mode") == "embedding+raw+morph"
                    and r.get("class_weight_policy") == cw
                    and float(r.get("probe_C")) == c
                ]
                if not raw or not morph:
                    continue
                total += 1
                rf, mf = _metric(raw[0], "test", "f1"), _metric(morph[0], "test", "f1")
                if rf > mf:
                    f1_raw += 1
                elif mf > rf:
                    f1_morph += 1
                if _metric(raw[0], "test", "auprc") > _metric(morph[0], "test", "auprc"):
                    auprc_raw += 1
        summary.append(
            {
                "run_name": run_name,
                "pairs_compared": total,
                "embedding_raw_wins_f1": f1_raw,
                "embedding_raw_morph_wins_f1": f1_morph,
                "embedding_raw_wins_auprc": auprc_raw,
            }
        )
    return summary


def consolidate(seed_nums: Sequence[int] = (1, 2, 3, 4)) -> Path:
    rows = _completed_rows(seed_nums)
    diag = ROOT / "results/diagnostics"
    out_json = diag / "probe_sweep_40ep_current_protocol.json"
    out_md = diag / "probe_sweep_40ep_current_protocol.md"
    # notes/probe_sweep_40ep_current_protocol.md is curated by hand (adds context and
    # takeaways). This script writes only the raw diagnostics artifact so it does not
    # clobber the interpreted note.

    payload = {
        "description": "Consolidated 40ep targeted probe sweep (current protocol).",
        "included_seeds": list(seed_nums),
        "cells_completed": len(rows),
        "rows": rows,
        "reference_rows": list(REFERENCE_ROWS),
        "top10_test_f1": _top_n(rows, "f1", 10),
        "top10_test_auprc": _top_n(rows, "auprc", 10),
        "top10_test_f1_at_0_5": _top_n(rows, "f1_at_0_5", 10),
        "best_per_seed": {
            "f1": _best_per_seed(rows, "f1"),
            "auprc": _best_per_seed(rows, "auprc"),
            "f1_at_0_5": _best_per_seed(rows, "f1_at_0_5"),
        },
        "shared_setting_stats": [
            _shared_setting_stats(rows, m, "model", 1.0) for m in SSL_MODES
        ],
        "embedding_raw_vs_morph": _raw_vs_morph_wins(rows),
    }
    payload["shared_setting_stats"] = [s for s in payload["shared_setting_stats"] if s]

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    header = (
        "| Run | Features | cw | C | AUROC | AUPRC | F1 | F1@0.5 | Thr |"
        "\n|-----|----------|----|---|------:|------:|---:|-------:|----:|"
    )
    lines = [
        "# 40ep targeted probe sweep — consolidated",
        "",
        f"Completed cells merged: **{len(rows)}**",
        "",
        "## Best per seed (test F1)",
        "",
        header,
    ]
    for r in payload["best_per_seed"]["f1"]:
        lines.append(_row_line(r))

    lines.extend(["", "## Shared setting @ cw=model, C=1.0 (mean ± std over seeds)", ""])
    for stat in payload["shared_setting_stats"]:
        if stat["class_weight_policy"] != "model" or stat["probe_C"] != 1.0:
            continue
        lines.append(
            f"- **`{stat['feature_mode']}`** (n={stat['n_seeds']}): "
            f"F1 {stat['test_f1_mean']:.4f} ± {stat['test_f1_std']:.4f}, "
            f"AUPRC {stat['test_auprc_mean']:.4f} ± {stat['test_auprc_std']:.4f}, "
            f"F1@0.5 {stat['test_f1_at_0_5_mean']:.4f} ± {stat['test_f1_at_0_5_std']:.4f}"
        )

    lines.extend(["", "## `embedding+raw` vs `embedding+raw+morph` win counts (F1)", ""])
    lines.append("| Seed | pairs | raw wins F1 | morph wins F1 | raw wins AUPRC |")
    lines.append("|------|------:|------------:|--------------:|---------------:|")
    for s in payload["embedding_raw_vs_morph"]:
        lines.append(
            f"| {s['run_name']} | {s['pairs_compared']} | {s['embedding_raw_wins_f1']} | "
            f"{s['embedding_raw_morph_wins_f1']} | {s['embedding_raw_wins_auprc']} |"
        )

    lines.extend(["", "## Reference comparisons (prior ablations)", ""])
    lines.append("| Label | F1 | AUPRC | F1@0.5 |")
    lines.append("|-------|---:|------:|-------:|")
    for ref in REFERENCE_ROWS:
        lines.append(
            f"| {ref['label']} | {ref['test_f1']:.4f} | {ref['test_auprc']:.4f} | "
            f"{ref['test_f1_at_0_5']:.4f} |"
        )

    lines.extend(["", "## Top 10 test F1", "", header])
    for r in payload["top10_test_f1"]:
        lines.append(_row_line(r))

    text = "\n".join(lines) + "\n"
    out_md.write_text(text, encoding="utf-8")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    return out_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="1,2,3,4", help="Comma-separated seed numbers.")
    args = parser.parse_args()
    seed_nums = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    consolidate(seed_nums)


if __name__ == "__main__":
    main()
