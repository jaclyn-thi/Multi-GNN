#!/usr/bin/env python3
"""Multiseed summary: pre_embedding_3h vs post_embedding_128 on Small-LI plain GINe baselines.

Reads per-seed JSONs from ``compare_representation_source.py`` (seeds 1–3) and reports per-seed
metrics, mean/std, pre-vs-post deltas, win counts, and whether the seed-1 representation advantage
replicates. Includes seed 1 from the existing current-protocol comparison when present.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MODES = ("embedding_only", "embedding_plus_raw")
SEEDS = (1, 2, 3)
SIMILAR_AUPRC_EPS = 0.005


def _load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(x: Any, nd: int = 4) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if math.isnan(v) else f"{v:.{nd}f}"


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    clean = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
    if not clean:
        return float("nan"), float("nan")
    m = sum(clean) / len(clean)
    if len(clean) < 2:
        return m, float("nan")
    var = sum((x - m) ** 2 for x in clean) / (len(clean) - 1)
    return m, math.sqrt(var)


def _seed_block(payload: Dict[str, Any], seed: int) -> Dict[str, Any]:
    modes: Dict[str, Any] = {}
    for mode in MODES:
        comp = payload.get("comparisons", {}).get(mode)
        if comp is None:
            continue
        post = comp["representations"]["post_embedding_128"]["test"]
        pre = comp["representations"]["pre_embedding_3h"]["test"]
        d_auprc = float(pre["auprc"] - post["auprc"])
        d_f1 = float(pre["f1_at_selected_threshold"] - post["f1_at_selected_threshold"])
        d_lift100 = float(pre.get("lift_at_100", float("nan")) - post.get("lift_at_100", float("nan")))
        if abs(d_auprc) < SIMILAR_AUPRC_EPS:
            verdict = "similar"
        elif d_auprc > 0:
            verdict = "pre_3h better"
        else:
            verdict = "post_128 better"
        modes[mode] = {
            "post": post,
            "pre": pre,
            "delta_auprc_pre_minus_post": d_auprc,
            "delta_f1_pre_minus_post": d_f1,
            "delta_lift_at_100_pre_minus_post": d_lift100,
            "verdict_auprc": verdict,
        }
    min_cov = min(
        (
            cov.get("joined_fraction_of_post", float("nan"))
            for cov in payload.get("split_pairing", {}).values()
        ),
        default=float("nan"),
    )
    return {
        "seed": seed,
        "run_name": payload.get("run_name"),
        "representation_dims": payload.get("representation_dims", {}),
        "min_pairing_coverage": float(min_cov),
        "modes": modes,
    }


def _default_seed_path(seed: int) -> Path:
    if seed == 1:
        return Path("results/diagnostics/pre_embedding_3h_vs_post_embedding_small_li.json")
    return Path(f"results/diagnostics/pre_embedding_3h_vs_post_embedding_small_li_seed{seed}.json")


def build_summary(seed_payloads: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    blocks = [_seed_block(seed_payloads[s], s) for s in sorted(seed_payloads)]
    aggregate: Dict[str, Any] = {}
    for mode in MODES:
        d_auprcs: List[float] = []
        d_lifts: List[float] = []
        pre_wins = 0
        n = 0
        for b in blocks:
            mb = b["modes"].get(mode)
            if mb is None:
                continue
            n += 1
            d_auprcs.append(mb["delta_auprc_pre_minus_post"])
            d_lifts.append(mb["delta_lift_at_100_pre_minus_post"])
            if mb["verdict_auprc"] == "pre_3h better":
                pre_wins += 1
        m_a, s_a = _mean_std(d_auprcs)
        m_l, s_l = _mean_std(d_lifts)
        aggregate[mode] = {
            "n_seeds": n,
            "pre_3h_wins_auprc": pre_wins,
            "mean_delta_auprc_pre_minus_post": m_a,
            "std_delta_auprc_pre_minus_post": s_a,
            "mean_delta_lift_at_100_pre_minus_post": m_l,
            "std_delta_lift_at_100_pre_minus_post": s_l,
        }

    seed1 = seed_payloads.get(1)
    replicated = None
    if seed1 and len(blocks) >= 2:
        rep = all(
            blocks[i]["modes"].get(mode, {}).get("verdict_auprc") == "pre_3h better"
            for i in range(1, len(blocks))
            for mode in MODES
            if mode in blocks[i]["modes"]
        )
        replicated = bool(rep) if blocks[1:] else None

    return {
        "diagnostic": "pre_embedding_3h_vs_post_embedding_128_small_li_multiseed",
        "data": "Small-LI",
        "no_ssl_retraining_for_probes": True,
        "paired": True,
        "seeds_included": sorted(seed_payloads.keys()),
        "per_seed": blocks,
        "aggregate": aggregate,
        "seed1_replicated": replicated,
        "conclusions": _conclusions(blocks, aggregate, replicated),
    }


def _conclusions(
    blocks: List[Dict[str, Any]],
    aggregate: Dict[str, Any],
    replicated: Optional[bool],
) -> Dict[str, str]:
    n = len(blocks)
    lines_emb = aggregate.get("embedding_only", {})
    lines_raw = aggregate.get("embedding_plus_raw", {})
    rep_txt = (
        "yes: every additional seed shows pre-3h winning AUPRC in both embedding-only and +raw"
        if replicated is True
        else (
            "no / mixed: at least one seed or stack does not replicate seed-1's pre-3h AUPRC win"
            if replicated is False
            else "insufficient seeds to assess replication beyond seed 1"
        )
    )
    return {
        "multiseed_auprc_embedding_only": (
            f"pre-3h wins AUPRC in {lines_emb.get('pre_3h_wins_auprc', 0)}/{lines_emb.get('n_seeds', 0)} seeds; "
            f"mean ΔAUPRC={_fmt(lines_emb.get('mean_delta_auprc_pre_minus_post'))} "
            f"± {_fmt(lines_emb.get('std_delta_auprc_pre_minus_post'))}"
        ),
        "multiseed_auprc_embedding_plus_raw": (
            f"pre-3h wins AUPRC in {lines_raw.get('pre_3h_wins_auprc', 0)}/{lines_raw.get('n_seeds', 0)} seeds; "
            f"mean ΔAUPRC={_fmt(lines_raw.get('mean_delta_auprc_pre_minus_post'))} "
            f"± {_fmt(lines_raw.get('std_delta_auprc_pre_minus_post'))}"
        ),
        "multiseed_alert_budget_embedding_plus_raw": (
            f"mean Δlift@100={_fmt(lines_raw.get('mean_delta_lift_at_100_pre_minus_post'), 2)} "
            f"± {_fmt(lines_raw.get('std_delta_lift_at_100_pre_minus_post'), 2)}"
        ),
        "seed1_result_replicated": rep_txt,
        "conservative_read": (
            f"{n} seed(s); single checkpoint per seed; development comparison — treat replication "
            "as directional unless all three seeds agree with similar magnitude."
        ),
    }


def write_note(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# pre_embedding_3h vs post_embedding_128 — Small-LI multiseed (plain GINe baseline)",
        "",
        "Paired probe comparison (same frozen checkpoint, `edge_id` inner-join) across plain "
        "Small-LI contrastive seeds. Fair policy: `cw=model`, C=1.0, val-tuned F1; primary stacks: "
        "embedding-only and embedding+raw.",
        "",
        f"**Seeds included:** {summary['seeds_included']}",
        "",
        "## Aggregate (pre − post)",
        "",
        "| stack | n seeds | pre-3h AUPRC wins | mean ΔAUPRC | std ΔAUPRC | mean Δlift@100 | std Δlift@100 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        agg = summary["aggregate"].get(mode, {})
        lines.append(
            f"| {mode} | {agg.get('n_seeds', 0)} | {agg.get('pre_3h_wins_auprc', 0)} | "
            f"{_fmt(agg.get('mean_delta_auprc_pre_minus_post'))} | "
            f"{_fmt(agg.get('std_delta_auprc_pre_minus_post'))} | "
            f"{_fmt(agg.get('mean_delta_lift_at_100_pre_minus_post'), 2)} | "
            f"{_fmt(agg.get('std_delta_lift_at_100_pre_minus_post'), 2)} |"
        )
    lines.append("")
    lines.append("## Conclusions")
    lines.append("")
    for k, v in summary["conclusions"].items():
        lines.append(f"- **{k.replace('_', ' ')}:** {v}")
    lines.append("")
    for b in summary["per_seed"]:
        lines.append(f"## Seed {b['seed']} — `{b['run_name']}`")
        lines.append("")
        lines.append(f"- min pairing coverage: {_fmt(b.get('min_pairing_coverage'))}")
        lines.append("")
        lines.append("| stack | AUPRC post | AUPRC pre | ΔAUPRC | F1 post | F1 pre | lift@100 post | lift@100 pre | verdict |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for mode, mb in b["modes"].items():
            post, pre = mb["post"], mb["pre"]
            lines.append(
                f"| {mode} | {_fmt(post['auprc'])} | {_fmt(pre['auprc'])} | "
                f"{mb['delta_auprc_pre_minus_post']:+.4f} | "
                f"{_fmt(post['f1_at_selected_threshold'])} | {_fmt(pre['f1_at_selected_threshold'])} | "
                f"{_fmt(post.get('lift_at_100'), 2)} | {_fmt(pre.get('lift_at_100'), 2)} | "
                f"{mb['verdict_auprc']} |"
            )
        lines.append("")
    lines.extend([
        "## Caveats",
        "",
        "- Development numbers; single checkpoint per seed.",
        "- pre_3h is 198-d vs post_128 128-d (linear-probe width confounder).",
        "- Seed 1 probe uses the earlier `pre_embedding_3h_vs_post_embedding_small_li.json` artifact.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed1_json", default=str(_default_seed_path(1)))
    p.add_argument("--seed2_json", default=str(_default_seed_path(2)))
    p.add_argument("--seed3_json", default=str(_default_seed_path(3)))
    p.add_argument(
        "--output_json",
        default="results/diagnostics/pre_embedding_3h_vs_post_embedding_small_li_multiseed.json",
    )
    p.add_argument(
        "--output_md",
        default="notes/pre_embedding_3h_vs_post_embedding_small_li_multiseed.md",
    )
    args = p.parse_args()

    seed_payloads: Dict[int, Dict[str, Any]] = {}
    for seed, path_str in ((1, args.seed1_json), (2, args.seed2_json), (3, args.seed3_json)):
        payload = _load(Path(path_str))
        if payload is not None:
            seed_payloads[seed] = payload

    if len(seed_payloads) < 2:
        raise FileNotFoundError(
            "Need at least two per-seed comparison JSONs (seed1 + one of seed2/3)."
        )

    summary = build_summary(seed_payloads)
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_note(Path(args.output_md), summary)
    print(f"Wrote {out_json} and {args.output_md} ({len(seed_payloads)} seed(s))")


if __name__ == "__main__":
    main()
