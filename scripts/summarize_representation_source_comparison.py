#!/usr/bin/env python3
"""Merge per-dataset pre_embedding_3h vs post_embedding_128 comparisons.

Reads the per-dataset diagnostic JSONs written by ``compare_representation_source.py`` and
produces a combined JSON plus a concise interpretation note. Robust to 1 or 2 datasets present.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

# AUPRC gap below which the two representations are called "similar".
SIMILAR_AUPRC_EPS = 0.005


def _load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(x: Any) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if math.isnan(v) else f"{v:.4f}"


def _interpret(delta_auprc: float, delta_f1: float) -> str:
    if math.isnan(delta_auprc):
        return "inconclusive (AUPRC undefined)"
    if abs(delta_auprc) < SIMILAR_AUPRC_EPS:
        return (
            "similar: the 128-d embedding preserves the useful information in the 3h "
            "representation roughly as efficiently"
        )
    if delta_auprc > 0:
        return (
            "3h better: the embedding_head compression appears to discard some information "
            "useful for the probe"
        )
    return (
        "128-d better: the embedding_head may denoise / reorganize the representation into a "
        "more linearly separable space"
    )


def _dataset_block(payload: Dict[str, Any]) -> Dict[str, Any]:
    comp = payload["comparisons"]["embedding_only"]
    post = comp["representations"]["post_embedding_128"]["test"]
    pre = comp["representations"]["pre_embedding_3h"]["test"]
    d_auprc = float(pre["auprc"] - post["auprc"])
    d_f1 = float(pre["f1_at_selected_threshold"] - post["f1_at_selected_threshold"])
    return {
        "data": payload["data"],
        "run_name": payload["run_name"],
        "representation_dims": payload["representation_dims"],
        "embedding_only": comp,
        "embedding_plus_raw": payload["comparisons"].get("embedding_plus_raw"),
        "delta_auprc_pre_minus_post": d_auprc,
        "delta_f1_pre_minus_post": d_f1,
        "interpretation": _interpret(d_auprc, d_f1),
    }


def write_note(path: Path, blocks: List[Dict[str, Any]]) -> None:
    lines = [
        "# pre_embedding_3h vs post_embedding_128 (current-protocol contrastive GINe)",
        "",
        "Diagnostic comparing two frozen representations from the **same** contrastively trained "
        "checkpoint: the exported 128-d `embedding_head` output (`post_embedding_128`) vs the "
        "`3 * n_hidden` tensor fed into `embedding_head` (`pre_embedding_3h` = "
        "`cat(src_node, dst_node, edge_attr)`). No contrastive retraining occurred; this is an "
        "extraction-location probe, not a new training method.",
        "",
        "**Protocol:** identical frozen linear-probe pipeline for both representations, paired by "
        "an `edge_id` inner-join per split (same rows/labels/order), same class weights / "
        "regularization / val-tuned threshold / seed. Primary comparison is embedding-only.",
        "",
    ]
    for b in blocks:
        dims = b["representation_dims"]
        lines.append(f"## {b['data']} — `{b['run_name']}`")
        lines.append("")
        lines.append(
            f"- dims: post_embedding_128 = {dims['post_embedding_128']}, "
            f"pre_embedding_3h = {dims['pre_embedding_3h']}"
        )
        lines.append(
            f"- ΔAUPRC (pre − post) = {b['delta_auprc_pre_minus_post']:+.4f}, "
            f"ΔF1 = {b['delta_f1_pre_minus_post']:+.4f}"
        )
        lines.append(f"- **interpretation:** {b['interpretation']}")
        lines.append("")
        post = b["embedding_only"]["representations"]["post_embedding_128"]["test"]
        pre = b["embedding_only"]["representations"]["pre_embedding_3h"]["test"]
        win = b["embedding_only"]["winners"]
        lines.append("| metric (test, embedding-only) | post_embedding_128 | pre_embedding_3h | winner |")
        lines.append("|---|---|---|---|")
        for label, key, wkey in [
            ("AUROC", "auroc", "auroc"),
            ("AUPRC", "auprc", "auprc"),
            ("F1 @ val-thr", "f1_at_selected_threshold", "f1_at_selected_threshold"),
            ("precision @ val-thr", "precision_at_selected_threshold", None),
            ("recall @ val-thr", "recall_at_selected_threshold", None),
            ("recall@100", "recall_at_100", "recall_at_100"),
            ("recall@1000", "recall_at_1000", "recall_at_1000"),
            ("precision@100", "precision_at_100", None),
            ("lift@100", "lift_at_100", None),
        ]:
            w = win.get(wkey, "") if wkey else ""
            lines.append(f"| {label} | {_fmt(post.get(key))} | {_fmt(pre.get(key))} | {w} |")
        lines.append("")
        if b["embedding_plus_raw"]:
            rpost = b["embedding_plus_raw"]["representations"]["post_embedding_128"]["test"]
            rpre = b["embedding_plus_raw"]["representations"]["pre_embedding_3h"]["test"]
            lines.append(
                f"- secondary (embedding+raw): AUPRC post={_fmt(rpost.get('auprc'))} "
                f"pre={_fmt(rpre.get('auprc'))}; F1@val-thr post={_fmt(rpost.get('f1_at_selected_threshold'))} "
                f"pre={_fmt(rpre.get('f1_at_selected_threshold'))}"
            )
            lines.append("")
    lines.extend([
        "## Caveats",
        "",
        "- Single seed / single checkpoint per dataset.",
        "- This is an extraction-location diagnostic, not a new training method or a claim of "
        "universal superiority.",
        "- `pre_embedding_3h` has more dimensions (198 vs 128), which can help a linear probe "
        "independent of information content; read alongside AUROC/AUPRC.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--small_hi_json", default="results/diagnostics/pre_embedding_3h_vs_post_embedding_small_hi.json")
    p.add_argument("--small_li_json", default="results/diagnostics/pre_embedding_3h_vs_post_embedding_small_li.json")
    p.add_argument("--output_json", default="results/diagnostics/pre_embedding_3h_vs_post_embedding_current_protocol.json")
    p.add_argument("--output_md", default="notes/pre_embedding_3h_vs_post_embedding_current_protocol.md")
    args = p.parse_args()

    blocks: List[Dict[str, Any]] = []
    for path in (Path(args.small_hi_json), Path(args.small_li_json)):
        payload = _load(path)
        if payload is not None:
            blocks.append(_dataset_block(payload))

    if not blocks:
        raise FileNotFoundError(
            "No per-dataset comparison JSONs found; run compare_representation_source.py first."
        )

    combined = {
        "diagnostic": "pre_embedding_3h_vs_post_embedding_128",
        "no_ssl_retraining": True,
        "paired": True,
        "datasets": blocks,
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    write_note(Path(args.output_md), blocks)
    print(f"Wrote {out_json} and {args.output_md} ({len(blocks)} dataset(s))")


if __name__ == "__main__":
    main()
