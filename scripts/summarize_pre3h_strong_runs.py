#!/usr/bin/env python3
"""Combined summary of pre_embedding_3h vs post_embedding_128 across several strong checkpoints.

Reads the per-run diagnostic JSONs written by ``compare_representation_source.py`` (one per
checkpoint, each possibly containing embedding_only / embedding_plus_raw / embedding_plus_raw_morph
comparisons) and produces:

  * a combined JSON with per-run/per-mode deltas and global winners, and
  * a concise interpretation note answering the batch's guiding questions.

All inputs are paired (edge_id inner-join) 2-way comparisons between the exported 128-d
``post_embedding_128`` and the 198-d ``pre_embedding_3h`` from the SAME frozen checkpoint. No SSL
retraining occurred. Wording is deliberately conservative (few seeds / checkpoints).

Example:
  python scripts/summarize_pre3h_strong_runs.py \\
    --inputs results/diagnostics/pre3h_vs_post128_small_hi_40ep_seed2.json \\
             results/diagnostics/pre3h_vs_post128_small_hi_fnf_seed1.json \\
             results/diagnostics/pre3h_vs_post128_small_li_fnf_seed1.json \\
    --output_json results/diagnostics/pre3h_strong_run_comparison.json \\
    --output_md notes/pre3h_strong_run_comparison.md
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# AUPRC gap below which two representations are called "similar" (not a meaningful win).
SIMILAR_AUPRC_EPS = 0.005
MODES = ("embedding_only", "embedding_plus_raw", "embedding_plus_raw_morph")
REPS = ("post_embedding_128", "pre_embedding_3h")


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


def _classify(payload: Dict[str, Any]) -> str:
    """Human-friendly role label for a run, used to answer the guiding questions."""
    name = str(payload.get("run_name", "")).lower()
    data = str(payload.get("data", "")).lower()
    is_li = "small-li" in data or "small_li" in name or name.startswith("small_li")
    is_fnf = "fnf" in name
    if is_li and is_fnf:
        return "small_li_fnf"
    if is_fnf:
        return "small_hi_fnf"
    if "40ep_seed2" in name or ("small-hi" in data and not is_fnf):
        return "small_hi_ordinary"
    return "other"


def _rep_test(comp: Dict[str, Any], rep: str) -> Dict[str, Any]:
    return comp["representations"][rep]["test"]


def _mode_block(comp: Dict[str, Any]) -> Dict[str, Any]:
    post = _rep_test(comp, "post_embedding_128")
    pre = _rep_test(comp, "pre_embedding_3h")
    d_auprc = float(pre["auprc"] - post["auprc"])
    d_f1 = float(pre["f1_at_selected_threshold"] - post["f1_at_selected_threshold"])
    if abs(d_auprc) < SIMILAR_AUPRC_EPS:
        verdict = "similar"
    elif d_auprc > 0:
        verdict = "pre_3h better"
    else:
        verdict = "post_128 better"
    return {
        "post": post,
        "pre": pre,
        "delta_auprc_pre_minus_post": d_auprc,
        "delta_f1_pre_minus_post": d_f1,
        "verdict_auprc": verdict,
    }


def _run_block(payload: Dict[str, Any]) -> Dict[str, Any]:
    modes: Dict[str, Any] = {}
    for mode in MODES:
        comp = payload.get("comparisons", {}).get(mode)
        if comp is not None:
            modes[mode] = _mode_block(comp)
    return {
        "run_name": payload["run_name"],
        "data": payload["data"],
        "role": _classify(payload),
        "representation_dims": payload.get("representation_dims", {}),
        "checkpoint_epoch": (payload.get("extraction_meta", {}) or {})
        .get("pre", {})
        .get("checkpoint_epoch"),
        "split_pairing": payload.get("split_pairing", {}),
        "modes": modes,
    }


def _min_coverage(run: Dict[str, Any]) -> float:
    fracs = [
        cov.get("joined_fraction_of_post", float("nan"))
        for cov in run.get("split_pairing", {}).values()
    ]
    fracs = [f for f in fracs if isinstance(f, (int, float)) and not math.isnan(f)]
    return min(fracs) if fracs else float("nan")


def _candidates(runs: List[Dict[str, Any]], metric: str) -> List[Tuple[float, str, str, str]]:
    """All (value, run_name, mode, rep) tuples for a test metric across every run/mode/rep."""
    out: List[Tuple[float, str, str, str]] = []
    for run in runs:
        for mode, mb in run["modes"].items():
            for rep in REPS:
                t = mb["pre"] if rep == "pre_embedding_3h" else mb["post"]
                v = t.get(metric, float("nan"))
                if isinstance(v, (int, float)) and not math.isnan(v):
                    out.append((float(v), run["run_name"], mode, rep))
    out.sort(key=lambda x: -x[0])
    return out


def _answer_improves(run: Optional[Dict[str, Any]]) -> str:
    if run is None:
        return "not evaluated in this batch."
    parts = []
    for mode, mb in run["modes"].items():
        parts.append(
            f"{mode}: ΔAUPRC={mb['delta_auprc_pre_minus_post']:+.4f} "
            f"(post={_fmt(mb['post']['auprc'])}→pre={_fmt(mb['pre']['auprc'])}), "
            f"ΔF1={mb['delta_f1_pre_minus_post']:+.4f} [{mb['verdict_auprc']}]"
        )
    wins = sum(1 for mb in run["modes"].values() if mb["verdict_auprc"] == "pre_3h better")
    n = len(run["modes"])
    head = f"pre-3h wins AUPRC in {wins}/{n} feature stack(s). "
    return head + "; ".join(parts)


def build_summary(input_paths: List[Path]) -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = []
    for p in input_paths:
        payload = _load(p)
        if payload is not None:
            runs.append(_run_block(payload))
    if not runs:
        raise FileNotFoundError("No per-run comparison JSONs found; run the probes first.")

    by_role = {r["role"]: r for r in runs}
    best_auprc = _candidates(runs, "auprc")
    best_f1 = _candidates(runs, "f1_at_selected_threshold")
    best_lift100 = _candidates(runs, "lift_at_100")

    questions = {
        "q1_pre3h_improves_ordinary_small_hi": _answer_improves(by_role.get("small_hi_ordinary")),
        "q2_pre3h_improves_fnf_fullstack": _answer_improves(by_role.get("small_hi_fnf")),
        "q3_small_li_advantage_extends_to_fnf": _answer_improves(by_role.get("small_li_fnf")),
        "q4_best_auprc": (
            f"{_fmt(best_auprc[0][0])} — {best_auprc[0][3]} in {best_auprc[0][2]} "
            f"({best_auprc[0][1]})"
            if best_auprc
            else "n/a"
        ),
        "q5_best_f1": (
            f"{_fmt(best_f1[0][0])} — {best_f1[0][3]} in {best_f1[0][2]} ({best_f1[0][1]})"
            if best_f1
            else "n/a"
        ),
        "q6_best_alert_budget_lift_at_100": (
            f"{_fmt(best_lift100[0][0], 2)} — {best_lift100[0][3]} in {best_lift100[0][2]} "
            f"({best_lift100[0][1]})"
            if best_lift100
            else "n/a"
        ),
        "q7_consistency": _consistency_statement(runs),
    }

    return {
        "diagnostic": "pre_embedding_3h_vs_post_embedding_128_strong_runs",
        "no_ssl_retraining": True,
        "paired": True,
        "pairing": "inner-join on edge_id per split; identical rows/labels/order per run",
        "num_runs": len(runs),
        "min_pairing_coverage": {r["run_name"]: _min_coverage(r) for r in runs},
        "runs": runs,
        "leaderboards": {
            "auprc_top": best_auprc[:5],
            "f1_top": best_f1[:5],
            "lift_at_100_top": best_lift100[:5],
        },
        "questions": questions,
    }


def _consistency_statement(runs: List[Dict[str, Any]]) -> str:
    verdicts = []
    for run in runs:
        for mb in run["modes"].values():
            verdicts.append(mb["verdict_auprc"])
    if not verdicts:
        return "no comparisons available."
    n = len(verdicts)
    pre_wins = verdicts.count("pre_3h better")
    post_wins = verdicts.count("post_128 better")
    sim = verdicts.count("similar")
    if pre_wins == n:
        return f"consistent: pre-3h wins AUPRC in all {n} run×stack comparisons."
    if post_wins == n:
        return f"consistent: post-128 wins AUPRC in all {n} run×stack comparisons."
    return (
        f"mixed / run- and stack-dependent: across {n} run×stack comparisons, "
        f"pre-3h wins {pre_wins}, post-128 wins {post_wins}, similar {sim}."
    )


def write_note(path: Path, summary: Dict[str, Any]) -> None:
    q = summary["questions"]
    lines = [
        "# pre_embedding_3h vs post_embedding_128 — strong existing checkpoints",
        "",
        "Extraction/probe diagnostic (no SSL retraining) comparing two frozen representations from "
        "the **same** contrastive checkpoint: the exported 128-d `embedding_head` output "
        "(`post_embedding_128`) vs the `3 * n_hidden` = 198-d tensor fed into `embedding_head` "
        "(`pre_embedding_3h`). Each comparison is paired by an `edge_id` inner-join per split "
        "(identical rows/labels/order), with the same probe seed, class weights, regularization, "
        "val-tuned threshold, and alert-budget definitions.",
        "",
        "> **Conservative read:** one checkpoint per run, single probe seed. Treat directional "
        "signals, not precise magnitudes, as the takeaway.",
        "",
        "## Pairing coverage (min joined-fraction over splits)",
        "",
    ]
    for name, cov in summary["min_pairing_coverage"].items():
        flag = "" if (isinstance(cov, float) and cov >= 0.99) else "  ⚠ below 0.99 — interpret with care"
        lines.append(f"- `{name}`: {_fmt(cov)}{flag}")
    lines.append("")

    lines.append("## Guiding questions")
    lines.append("")
    lines.append(f"1. **Improves strongest ordinary Small-HI (40ep seed2)?** {q['q1_pre3h_improves_ordinary_small_hi']}")
    lines.append(f"2. **Improves strongest FNF full-stack (Small-HI FNF seed1)?** {q['q2_pre3h_improves_fnf_fullstack']}")
    lines.append(f"3. **Does the Small-LI advantage extend to FNF?** {q['q3_small_li_advantage_extends_to_fnf']}")
    lines.append(f"4. **Best AUPRC (any run/stack/representation):** {q['q4_best_auprc']}")
    lines.append(f"5. **Best val-tuned F1:** {q['q5_best_f1']}")
    lines.append(f"6. **Strongest alert-budget (lift@100):** {q['q6_best_alert_budget_lift_at_100']}")
    lines.append(f"7. **Are gains consistent?** {q['q7_consistency']}")
    lines.append("")

    for run in summary["runs"]:
        dims = run["representation_dims"]
        lines.append(f"## {run['data']} — `{run['run_name']}` ({run['role']})")
        lines.append("")
        lines.append(
            f"- dims: post_embedding_128 = {dims.get('post_embedding_128', '?')}, "
            f"pre_embedding_3h = {dims.get('pre_embedding_3h', '?')}"
            + (f"; checkpoint epoch {run['checkpoint_epoch']}" if run.get("checkpoint_epoch") else "")
        )
        lines.append("")
        lines.append(
            "| feature stack | AUPRC post | AUPRC pre | ΔAUPRC | F1 post | F1 pre | ΔF1 | "
            "lift@100 post | lift@100 pre | AUPRC verdict |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for mode, mb in run["modes"].items():
            post, pre = mb["post"], mb["pre"]
            lines.append(
                f"| {mode} | {_fmt(post['auprc'])} | {_fmt(pre['auprc'])} | "
                f"{mb['delta_auprc_pre_minus_post']:+.4f} | "
                f"{_fmt(post['f1_at_selected_threshold'])} | {_fmt(pre['f1_at_selected_threshold'])} | "
                f"{mb['delta_f1_pre_minus_post']:+.4f} | "
                f"{_fmt(post.get('lift_at_100'), 2)} | {_fmt(pre.get('lift_at_100'), 2)} | "
                f"{mb['verdict_auprc']} |"
            )
        lines.append("")

    lines.extend([
        "## Caveats",
        "",
        "- Extraction-location diagnostic only; **no contrastive retraining**. Not a claim that "
        "pre-3h is a universally better training target.",
        "- `pre_embedding_3h` has more dimensions (198 vs 128); a linear probe can benefit from the "
        "extra width independent of information content. Read AUPRC alongside AUROC and alert-budget.",
        "- `post_embedding_128` is reused from earlier extractions while `pre_embedding_3h` is a "
        "fresh forward pass; in the hetero loader the train split is sampled, so on high-degree "
        "nodes the two passes may sample slightly different neighborhoods. Pairing is by `edge_id` "
        "and coverage is reported above.",
        "- Single checkpoint per run and a single probe seed.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="+", required=True, help="Per-run comparison JSON paths.")
    p.add_argument("--output_json", default="results/diagnostics/pre3h_strong_run_comparison.json")
    p.add_argument("--output_md", default="notes/pre3h_strong_run_comparison.md")
    args = p.parse_args()

    summary = build_summary([Path(x) for x in args.inputs])

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_note(Path(args.output_md), summary)
    print(f"Wrote {out_json} and {args.output_md} ({summary['num_runs']} run(s))")


if __name__ == "__main__":
    main()
