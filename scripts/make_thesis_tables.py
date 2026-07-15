#!/usr/bin/env python3
"""Generate thesis-ready Markdown and LaTeX tables from the experiment registry."""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]

CANONICAL_PREPOST = "results/diagnostics/pre3h_strong_run_comparison.json"
MULTISEED_LI = "results/diagnostics/pre_embedding_3h_vs_post_embedding_small_li_multiseed.json"
LEGACY_EVAL = "results/diagnostics/eval_small_li_legacy_supervised_gin_emlps_tds_100ep_seed1.json"
HI_LEGACY_EVAL = "results/diagnostics/eval_small_hi_legacy_supervised_gin_emlps_tds_100ep_seed1.json"
FEATURE_ABLATION = "results/diagnostics/probe_feature_ablation_current_protocol_comparison.json"
ARCH_SWEEP = "results/diagnostics/architecture_sweep_shared_probe_weights.json"
PNA_WIDTH_ALIGNED = "results/diagnostics/pna_width_aligned_probe.json"
PNA_WIDTH_TF_PROBE = "results/diagnostics/pna_width65_temporal_flow_probe.json"
EMB198_SOURCE = "results/diagnostics/small_li_embedding_dim_128_vs_198.json"

REP_DISPLAY = {
    "post_embedding_128": "post-128",
    "pre_embedding_3h": "pre-3h",
}

REP_REGISTRY = {v: k for k, v in REP_DISPLAY.items()}
REP_REGISTRY.update(REP_DISPLAY)

ARCH_GEOMETRY = {
    "gin": {"hidden": 66, "pre": 198, "post": 128, "params": None},
    "gat": {"hidden": None, "pre": None, "post": 128, "params": None},
    "pna": {"hidden": 20, "pre": 60, "post": 128, "params": None},
    "rgcn": {"hidden": None, "pre": None, "post": 128, "params": None},
}

ARCH_ENCODER_ORDER = ("gin", "gat", "pna", "rgcn")

# Internal arm IDs -> reader-facing comparison labels (Table 5 and notes).
TF_ARM_COMPARISON = {
    "A": "pre-3h only",
    "B": "pre-3h + raw",
    "C": "pre-3h + temporal-flow",
    "D": "pre-3h + raw + temporal-flow",
}

TF_ARM_STACK = {
    "A": "embedding",
    "B": "embedding+raw",
    "C": "embedding+temporal_flow_causal",
    "D": "embedding+raw+temporal_flow_causal",
}

LIFT_AT_100_NOTE = (
    "Lift@100 = P@100 divided by the test-set positive rate for that dataset."
)

ALERT_BUDGET_NOTES = [
    "P@100 = precision among the top 100 scored test transactions.",
    "R@100 = fraction of all positive test transactions recovered in the top 100 scored test transactions.",
    LIFT_AT_100_NOTE,
]

ALERT_BUDGET_KS = (100, 500, 1000)

MISSING = "—"


def _load_registry(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    return payload.get("rows") or []


def _eligible(
    row: Optional[Dict[str, Any]],
    include_provisional: bool,
) -> bool:
    if not row:
        return False
    vs = row.get("validation_status")
    if vs in ("provisional", "pending_validation") and not include_provisional:
        return False
    if row.get("superseded"):
        return False
    return True


def _filter_rows(
    rows: Sequence[Dict[str, Any]],
    include_provisional: bool = False,
    **kw: Any,
) -> List[Dict[str, Any]]:
    out = list(rows)
    for k, v in kw.items():
        if v is None:
            out = [r for r in out if r.get(k) is None]
        else:
            out = [r for r in out if r.get(k) == v]
    out = [r for r in out if r.get("status") in (None, "evaluated")]
    if not include_provisional:
        out = [
            r for r in out
            if r.get("validation_status") not in ("provisional", "pending_validation", "superseded")
            or (include_provisional and r.get("validation_status") != "superseded")
        ]
    return out


def _pick(
    rows: Sequence[Dict[str, Any]],
    include_provisional: bool = False,
    prefer_source: Optional[str] = None,
    **kw: Any,
) -> Optional[Dict[str, Any]]:
    c = _filter_rows(rows, include_provisional=include_provisional, **kw)
    if prefer_source:
        pref = [r for r in c if r.get("source_json") == prefer_source]
        if pref:
            c = pref
    if not c:
        return None
    if len(c) > 1 and kw.get("run_id"):
        return c[0]
    return c[0]


def _pick_temporal_arm(
    rows: Sequence[Dict[str, Any]],
    dataset: str,
    arm: str,
    include_provisional: bool,
    prefer_validated: bool = True,
) -> Optional[Dict[str, Any]]:
    stack_map = {
        "A": "embedding",
        "B": "embedding+raw",
        "C": "embedding+temporal_flow_causal",
        "D": "embedding+raw+temporal_flow_causal",
    }
    stack = stack_map[arm]
    c = _filter_rows(
        rows,
        include_provisional=include_provisional,
        dataset=dataset,
        probe_feature_stack=stack,
        representation_source="pre_embedding_3h",
    )
    c = [r for r in c if "tf_arm{0}".format(arm) in (r.get("run_id") or "")]
    if prefer_validated:
        val = [r for r in c if r.get("validation_status") == "validated"]
        if val:
            c = val
        elif not include_provisional:
            c = [
                r for r in c
                if r.get("validation_status") not in ("provisional", "pending_validation")
            ]
    if dataset == "Small-HI":
        c = [r for r in c if "40ep_seed2" in (r.get("run_id") or "")]
    if not c:
        return None
    if dataset == "Small-LI" and len(c) > 1:
        # prefer seed1 canonical file unless multiseed aggregate requested elsewhere
        c = sorted(c, key=lambda r: (r.get("seed") or 99, r.get("source_json") or ""))
    return c[0]


def fmt_metric(v: Any, decimals: int = 3) -> str:
    if v is None:
        return MISSING
    try:
        return "{0:.{1}f}".format(float(v), decimals)
    except (TypeError, ValueError):
        return MISSING


def fmt_lift(v: Any) -> str:
    if v is None:
        return MISSING
    x = float(v)
    if abs(x) >= 10 or abs(x - round(x)) < 0.05:
        return str(int(round(x)))
    return "{0:.1f}".format(x)


def fmt_pm(mean: Any, std: Any, *, kind: str = "metric", decimals: int = 3) -> str:
    if mean is None:
        return MISSING
    fmt_fn = fmt_lift if kind == "lift" else (lambda v: fmt_metric(v, decimals))
    m_str = fmt_fn(mean)
    if std is None or float(std) == 0.0:
        return m_str
    return "{0} ± {1}".format(m_str, fmt_fn(std))


def _ms(
    payload: Dict[str, Any],
    stack: str,
    rep: str,
    metric: str,
) -> Tuple[Optional[float], Optional[float]]:
    agg = (payload.get("multiseed_aggregates") or {}).get("stacks", {}).get(stack, {})
    rep_block = (agg.get("representations") or {}).get(rep, {})
    m = rep_block.get(metric) or rep_block.get(
        metric.replace("f1", "f1_at_selected_threshold") if metric == "f1" else metric
    )
    if not m:
        key = "f1_at_selected_threshold" if metric == "f1" else metric
        m = rep_block.get(key, {})
    if isinstance(m, dict):
        return m.get("mean"), m.get("std")
    return None, None


def _tf_ms(
    payload: Dict[str, Any],
    arm: str,
    metric: str,
) -> Tuple[Optional[float], Optional[float]]:
    agg = (payload.get("temporal_flow_multiseed_aggregates") or {}).get("arms", {}).get(arm, {})
    m = (agg.get("metrics") or {}).get(metric, {})
    if isinstance(m, dict):
        return m.get("mean"), m.get("std")
    return None, None


def _tf_d_minus_b_ms(
    payload: Dict[str, Any],
    metric: str,
) -> Tuple[Optional[float], Optional[float]]:
    block = (payload.get("temporal_flow_multiseed_aggregates") or {}).get("D_minus_B", {})
    m = (block.get("metrics") or {}).get(metric, {})
    if isinstance(m, dict):
        std = m.get("sample_std_ddof1", m.get("std"))
        return m.get("mean"), std
    return None, None


def fmt_delta(mean: Any, std: Any = None, *, kind: str = "metric") -> str:
    if mean is None:
        return MISSING
    fmt_fn = fmt_lift if kind == "lift" else (lambda v: fmt_metric(v))
    if std is not None and float(std) != 0.0:
        m_str = fmt_fn(mean)
        if float(mean) >= 0 and not m_str.startswith("+"):
            m_str = "+{0}".format(m_str)
        return "{0} ± {1}".format(m_str, fmt_fn(std))
    m_str = fmt_fn(mean)
    if float(mean) >= 0 and not m_str.startswith("+"):
        return "+{0}".format(m_str)
    return m_str


def _li_tf_arm_d_aggregate(payload: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Validated Small-LI temporal-flow Arm D multiseed cells (shared by Tables 3, 5, 6)."""
    tf_val = payload.get("temporal_flow_validation") or {}
    if not tf_val.get("passed"):
        return None
    auprc = _tf_ms(payload, "D", "auprc")
    if auprc[0] is None:
        return None
    return {
        "auroc": fmt_pm(*_tf_ms(payload, "D", "auroc")),
        "auprc": fmt_pm(*_tf_ms(payload, "D", "auprc")),
        "f1": fmt_pm(*_tf_ms(payload, "D", "f1_at_selected_threshold")),
        "p100": fmt_pm(*_tf_ms(payload, "D", "precision_at_100")),
        "r100": fmt_pm(*_tf_ms(payload, "D", "recall_at_100")),
        "lift": fmt_pm(*_tf_ms(payload, "D", "lift_at_100"), kind="lift"),
        "caveat": "val-tuned F1; mean ± sample SD (n=3)",
    }


def _with_alert_budget_notes(footnotes: List[str]) -> List[str]:
    out = list(footnotes)
    for note in ALERT_BUDGET_NOTES:
        if note not in out:
            out.append(note)
    return out


def _with_lift_note(footnotes: List[str]) -> List[str]:
    return _with_alert_budget_notes(footnotes)


def _alert_metric(row: Optional[Dict[str, Any]], metric: str) -> Any:
    """Read alert-budget metric from a registry row with common alias fallbacks."""
    if not row:
        return None
    if row.get(metric) is not None:
        return row.get(metric)
    aliases = {
        "recall_at_100": ("r_at_100", "R_at_100", "alert_recall_at_100", "top100_recall"),
        "precision_at_100": ("p_at_100", "P_at_100", "alert_precision_at_100", "top100_precision"),
        "recall_at_500": ("r_at_500", "R_at_500"),
        "precision_at_500": ("p_at_500", "P_at_500"),
        "recall_at_1000": ("r_at_1000", "R_at_1000"),
        "precision_at_1000": ("p_at_1000", "P_at_1000"),
    }
    for key in aliases.get(metric, ()):
        if row.get(key) is not None:
            return row.get(key)
    return None


def _warn_missing_r100(
    label: str,
    row: Optional[Dict[str, Any]],
    missing_log: List[str],
) -> None:
    if not row:
        return
    if _alert_metric(row, "recall_at_100") is not None:
        return
    msg = "{0}: missing R@100 (source={1}, run_id={2})".format(
        label,
        row.get("source_json", "?"),
        row.get("run_id", "?"),
    )
    missing_log.append(msg)
    warnings.warn(msg)


def _metric_cells(
    row: Optional[Dict[str, Any]],
    *,
    label: str = "",
    missing_r100_log: Optional[List[str]] = None,
) -> List[str]:
    if not row:
        return [MISSING] * 6
    if missing_r100_log is not None and label:
        _warn_missing_r100(label, row, missing_r100_log)
    return [
        fmt_metric(row.get("AUROC")),
        fmt_metric(row.get("AUPRC")),
        fmt_metric(row.get("F1")),
        fmt_metric(_alert_metric(row, "precision_at_100")),
        fmt_metric(_alert_metric(row, "recall_at_100")),
        fmt_lift(_alert_metric(row, "lift_at_100")),
    ]


def _alert_budget_cells(
    row: Optional[Dict[str, Any]],
    ks: Sequence[int] = ALERT_BUDGET_KS,
) -> List[str]:
    if not row:
        return [MISSING] * (3 * len(ks))
    cells: List[str] = []
    for k in ks:
        cells.extend([
            fmt_metric(_alert_metric(row, "precision_at_{0}".format(k))),
            fmt_metric(_alert_metric(row, "recall_at_{0}".format(k))),
            fmt_lift(_alert_metric(row, "lift_at_{0}".format(k))),
        ])
    return cells


def _alert_budget_ms_cells(
    payload: Dict[str, Any],
    stack: str,
    rep: str,
    ks: Sequence[int] = ALERT_BUDGET_KS,
) -> List[str]:
    cells: List[str] = []
    for k in ks:
        cells.extend([
            fmt_pm(*_ms(payload, stack, rep, "precision_at_{0}".format(k))),
            fmt_pm(*_ms(payload, stack, rep, "recall_at_{0}".format(k))),
            fmt_pm(*_ms(payload, stack, rep, "lift_at_{0}".format(k)), kind="lift"),
        ])
    return cells


def _source_comment(row: Optional[Dict[str, Any]]) -> str:
    if not row:
        return ""
    return "<!-- source: {0} run_id={1} -->".format(
        row.get("source_json", "?"), row.get("run_id", "?")
    )


def _caveat_cell(row: Optional[Dict[str, Any]], extra: str = "") -> str:
    if not row:
        return extra or MISSING
    parts = []
    vs = row.get("validation_status")
    if vs in ("provisional", "pending_validation"):
        parts.append(vs.replace("_", " "))
    if row.get("threshold_rule") == "paper_argmax":
        parts.append("paper_argmax F1")
    elif row.get("threshold_rule") == "max_f1_on_val":
        parts.append("val-tuned F1")
    if extra:
        parts.append(extra)
    c = row.get("caveats")
    if c and "non-paired" in c.lower():
        parts.append("non-paired")
    return "; ".join(parts) if parts else MISSING


def _bold_best(values: List[str], higher_better: bool = True) -> List[str]:
    nums: List[Tuple[int, float]] = []
    for i, v in enumerate(values):
        if v == MISSING:
            continue
        try:
            nums.append((i, float(v.split("±")[0].strip())))
        except ValueError:
            continue
    if len(nums) < 2:
        return values
    best = max(nums, key=lambda x: x[1]) if higher_better else min(nums, key=lambda x: x[1])
    out = list(values)
    out[best[0]] = "**{0}**".format(out[best[0]])
    return out


def _ms_delta(
    payload: Dict[str, Any],
    stack: str,
    delta_key: str,
) -> Tuple[Optional[float], Optional[float]]:
    agg = (payload.get("multiseed_aggregates") or {}).get("stacks", {}).get(stack, {})
    d = (agg.get("deltas_pre_minus_post") or {}).get(delta_key, {})
    if isinstance(d, dict):
        return d.get("mean"), d.get("std")
    return None, None


def _registry_rep(rep: str) -> str:
    return REP_REGISTRY.get(rep, rep)


def _strip_source_comments(text: str) -> str:
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("<!--")]
    return "\n".join(lines).strip() + "\n"


def _md_table(headers: Sequence[str], body: Sequence[Sequence[str]], notes: Sequence[str]) -> str:
    if not body:
        lines = []
    else:
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in body:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
    if notes:
        lines.append("")
        lines.append("**Notes:**")
        for fn in notes:
            lines.append("- {0}".format(fn))
    return "\n".join(lines)


def _tex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def _tex_table(
    caption: str,
    headers: Sequence[str],
    body: Sequence[Sequence[str]],
    footnotes: Sequence[str],
) -> str:
    colspec = "l" + "r" * (len(headers) - 1)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{{{0}}}".format(_tex_escape(caption)),
        "\\begin{{tabular}}{{{0}}}".format(colspec),
        "\\toprule",
        " & ".join(_tex_escape(h) for h in headers) + " \\\\",
        "\\midrule",
    ]
    for row in body:
        cells = []
        for c in row:
            t = str(c).replace("**", "")
            cells.append(_tex_escape(t) if t != MISSING else "---")
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    for i, fn in enumerate(footnotes, 1):
        lines.append("\\footnotetext[{0}]{{{1}}}".format(i, _tex_escape(fn)))
    lines.append("\\end{table}")
    return "\n".join(lines)


def build_dataset_summary(payload: Dict[str, Any]) -> Tuple[List[str], List[List[str]], List[str]]:
    headers = ["Dataset", "Split", "# Transactions", "# Positives", "Positive rate", "Task"]
    body: List[List[str]] = []
    meta = payload.get("dataset_metadata") or {}
    for row in meta.get("rows") or []:
        pr = row.get("positive_rate")
        pr_s = "{0:.3f}%".format(float(pr) * 100) if pr is not None else MISSING
        body.append([
            row.get("dataset", MISSING),
            row.get("split", MISSING),
            str(row.get("n_transactions", MISSING)),
            str(row.get("n_positives", MISSING)),
            pr_s,
            row.get("task", MISSING),
        ])
    footnotes = [
        "Split counts from cited source JSON in registry dataset_metadata; node counts omitted when unavailable.",
    ]
    return headers, body, footnotes


def build_main_small_hi(
    payload: Dict[str, Any],
    include_provisional: bool,
    strict: bool,
) -> Tuple[List[str], List[List[str]], List[str], List[str]]:
    rows = _rows(payload)
    headers = [
        "Method", "Representation", "Features", "AUROC", "AUPRC", "F1",
        "P@100", "R@100", "Lift@100", "Caveat",
    ]
    body: List[List[str]] = []
    comments: List[str] = []
    missing: List[str] = []
    missing_r100: List[str] = []

    def add(label: str, rep: str, feats: str, row: Optional[Dict[str, Any]], extra_caveat: str = ""):
        if row is None:
            missing.append(label)
            body.append([label, rep, feats] + [MISSING] * 6 + [_caveat_cell(None, extra_caveat)])
            return
        comments.append(_source_comment(row))
        body.append([
            label, rep, feats,
            *_metric_cells(row, label=label, missing_r100_log=missing_r100),
            _caveat_cell(row, extra_caveat),
        ])

    raw = _pick(rows, probe_feature_stack="raw", dataset="Small-HI", training_epochs=20, seed=1)
    add("Raw features only", "—", "raw", raw, "no SSL")

    raw_morph = _pick(rows, probe_feature_stack="raw+morph", dataset="Small-HI", training_epochs=20, seed=1)
    add("Raw + morphology", "—", "raw+morph", raw_morph, "no SSL")

    add(
        "SSL post-128",
        "post-128",
        "embedding",
        _pick(rows, dataset="Small-HI", representation_source="post_embedding_128",
              probe_feature_stack="embedding", training_epochs=40, seed=2,
              source_json=CANONICAL_PREPOST),
    )
    add(
        "SSL pre-3h",
        "pre-3h",
        "embedding",
        _pick(rows, dataset="Small-HI", representation_source="pre_embedding_3h",
              probe_feature_stack="embedding", training_epochs=40, seed=2,
              source_json=CANONICAL_PREPOST),
    )
    add(
        "SSL post-128 + raw",
        "post-128",
        "embedding+raw",
        _pick(rows, dataset="Small-HI", representation_source="post_embedding_128",
              probe_feature_stack="embedding+raw", training_epochs=40, seed=2,
              source_json=CANONICAL_PREPOST),
    )
    add(
        "SSL pre-3h + raw",
        "pre-3h",
        "embedding+raw",
        _pick(rows, dataset="Small-HI", representation_source="pre_embedding_3h",
              probe_feature_stack="embedding+raw", training_epochs=40, seed=2,
              source_json=CANONICAL_PREPOST),
    )

    tf_row = _pick_temporal_arm(rows, "Small-HI", "D", include_provisional)
    if tf_row and _eligible(tf_row, include_provisional):
        add(
            "SSL pre-3h + raw + temporal-flow",
            "pre-3h",
            "embedding+raw+temporal_flow_causal",
            tf_row,
            "validated temporal-flow stack",
        )
    elif include_provisional:
        add(
            "SSL pre-3h + raw + temporal-flow",
            "pre-3h",
            "embedding+raw+temporal_flow_causal",
            None,
            "provisional; pending validation",
        )
    elif strict:
        missing.append("SSL pre-3h + raw + temporal-flow")

    hi_legacy = _pick(
        rows, dataset="Small-HI", objective="supervised",
        threshold_rule="paper_argmax", scout_or_formal="formal",
        source_json=HI_LEGACY_EVAL,
    )
    if hi_legacy:
        add(
            "Legacy supervised GIN (100ep seed1)",
            "logits",
            "in-GNN end-to-end",
            hi_legacy,
            "paper_argmax F1; supervised CE; not comparable to SSL val-tuned F1",
        )
    else:
        missing.append("Legacy supervised GIN (Small-HI)")
        body.append([
            "Legacy supervised GIN (100ep seed1)", "logits", "in-GNN end-to-end",
        ] + [MISSING] * 6 + ["pending Small-HI legacy supervised run"])

    if strict and missing:
        raise RuntimeError("Missing Small-HI main table rows: {0}".format(", ".join(missing)))

    for msg in missing_r100:
        warnings.warn(msg)

    # Bold best within comparable SSL paired rows only (pre/post same stack)
    auprc_col = 4
    for i in (3, 4, 5, 6):  # rows indices for SSL embedding and +raw (not raw-only)
        pass
    comparable_idx = [3, 4, 5, 6]
    vals = [body[i][auprc_col] for i in comparable_idx if i < len(body)]
    bolded = _bold_best(vals)
    for j, i in enumerate(comparable_idx):
        if i < len(body) and j < len(bolded):
            body[i][auprc_col] = bolded[j]

    footnotes = _with_lift_note([
        "Small-HI pre/post rows use paired strong-run protocol ({0}).".format(CANONICAL_PREPOST),
        "F1 for SSL rows is validation-tuned; raw-feature rows use val-tuned probe.",
        "Legacy supervised row uses end-to-end labeled training and paper_argmax F1 (not val-tuned).",
        "Temporal-flow stack included only when validated or with --include_provisional.",
        "Contrastive-method variants such as FNF are reported in the appendix.",
    ])
    return headers, body, footnotes, comments


def build_main_small_li(
    payload: Dict[str, Any],
    strict: bool,
) -> Tuple[List[str], List[List[str]], List[str], List[str]]:
    rows = _rows(payload)
    headers = [
        "Method", "Representation", "Features", "AUROC", "AUPRC", "F1",
        "P@100", "R@100", "Lift@100", "Caveat",
    ]
    body: List[List[str]] = []
    comments: List[str] = []
    missing: List[str] = []

    def add_ms(label: str, rep_display: str, stack: str, metrics: Sequence[Tuple[str, str]]):
        rep = _registry_rep(rep_display)
        m: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        for metric, kind in metrics:
            m[metric] = _ms(payload, stack, rep, metric)
        if all(v[0] is None for v in m.values()):
            missing.append(label)
            body.append([label, rep_display, stack] + [MISSING] * 6 + [MISSING])
            return
        comments.append("<!-- multiseed: {0} {1} -->".format(stack, rep))
        body.append([
            label, rep_display, stack,
            fmt_pm(*m["auroc"]),
            fmt_pm(*m["auprc"]),
            fmt_pm(*m["f1_at_selected_threshold"]),
            fmt_pm(*m["precision_at_100"]),
            fmt_pm(*m["recall_at_100"]),
            fmt_pm(*m["lift_at_100"], kind="lift"),
            "frozen probe; val-tuned F1; mean ± sample SD (n=3)",
        ])

    add_ms("SSL post-128", "post-128", "embedding", (
        ("auroc", "metric"), ("auprc", "metric"), ("f1_at_selected_threshold", "metric"),
        ("precision_at_100", "metric"), ("recall_at_100", "metric"), ("lift_at_100", "lift"),
    ))
    add_ms("SSL pre-3h", "pre-3h", "embedding", (
        ("auroc", "metric"), ("auprc", "metric"), ("f1_at_selected_threshold", "metric"),
        ("precision_at_100", "metric"), ("recall_at_100", "metric"), ("lift_at_100", "lift"),
    ))
    add_ms("SSL post-128 + raw", "post-128", "embedding+raw", (
        ("auroc", "metric"), ("auprc", "metric"), ("f1_at_selected_threshold", "metric"),
        ("precision_at_100", "metric"), ("recall_at_100", "metric"), ("lift_at_100", "lift"),
    ))
    add_ms("SSL pre-3h + raw", "pre-3h", "embedding+raw", (
        ("auroc", "metric"), ("auprc", "metric"), ("f1_at_selected_threshold", "metric"),
        ("precision_at_100", "metric"), ("recall_at_100", "metric"), ("lift_at_100", "lift"),
    ))

    tf_d = _li_tf_arm_d_aggregate(payload)
    if tf_d:
        comments.append("<!-- multiseed: temporal_flow Arm D Small-LI (Table 5 aggregate) -->")
        body.append([
            "SSL pre-3h + raw + temporal-flow",
            "pre-3h",
            "embedding+raw+temporal_flow_causal",
            tf_d["auroc"],
            tf_d["auprc"],
            tf_d["f1"],
            tf_d["p100"],
            tf_d["r100"],
            tf_d["lift"],
            tf_d["caveat"],
        ])

    legacy = _pick(
        rows,
        dataset="Small-LI",
        objective="supervised",
        threshold_rule="paper_argmax",
        scout_or_formal="formal",
        superseded=False,
    )
    if legacy:
        comments.append(_source_comment(legacy))
        body.append([
            "Legacy supervised GIN (100ep seed1)",
            "logits",
            "in-GNN end-to-end",
            *_metric_cells(legacy, label="Legacy supervised GIN (Small-LI)"),
            _caveat_cell(legacy),
        ])
    else:
        missing.append("Legacy supervised GIN")
        body.append(["Legacy supervised GIN (100ep seed1)", "logits", "in-GNN end-to-end"]
                    + [MISSING] * 6 + [MISSING])

    if strict and missing:
        raise RuntimeError("Missing Small-LI main table rows: {0}".format(", ".join(missing)))

    footnotes = _with_lift_note([
        "SSL multiseed rows: mean ± sample SD (ddof=1) over seeds 1–3; frozen linear probe with validation-tuned thresholds.",
        "SSL pre-3h + raw + temporal-flow uses validated temporal-flow multiseed aggregate (same as Table 5).",
        "Supervised row uses paper_argmax F1 from {0}; not directly comparable to SSL F1 without footnote.".format(LEGACY_EVAL),
    ])
    return headers, body, footnotes, comments


def build_representation_ablation(
    payload: Dict[str, Any],
    strict: bool,
) -> Tuple[List[str], List[List[str]], List[str], List[str]]:
    rows = _rows(payload)
    headers = [
        "Dataset / run", "Feature stack", "Post-128 AUPRC", "Pre-3h AUPRC", "Δ AUPRC",
        "Post-128 F1", "Pre-3h F1", "Caveat",
    ]
    body: List[List[str]] = []
    comments: List[str] = []
    missing: List[str] = []

    def add_pair(dataset: str, run_label: str, stack: str, seed: int = 2, ep: int = 40):
        post = _pick(
            rows, dataset=dataset, probe_feature_stack=stack,
            representation_source="post_embedding_128", seed=seed, training_epochs=ep,
            source_json=CANONICAL_PREPOST if dataset == "Small-HI" else MULTISEED_LI,
        )
        pre = _pick(
            rows, dataset=dataset, probe_feature_stack=stack,
            representation_source="pre_embedding_3h", seed=seed, training_epochs=ep,
            source_json=CANONICAL_PREPOST if dataset == "Small-HI" else MULTISEED_LI,
        )
        if dataset == "Small-LI":
            post_m, post_s = _ms(payload, stack, "post_embedding_128", "auprc")
            pre_m, pre_s = _ms(payload, stack, "pre_embedding_3h", "auprc")
            post_f, post_fs = _ms(payload, stack, "post_embedding_128", "f1_at_selected_threshold")
            pre_f, pre_fs = _ms(payload, stack, "pre_embedding_3h", "f1_at_selected_threshold")
            if post_m is None or pre_m is None:
                missing.append("{0} {1}".format(run_label, stack))
                body.append([run_label, stack] + [MISSING] * 6)
                return
            d_mean, d_std = _ms_delta(payload, stack, "delta_auprc_pre_minus_post")
            delta_cell = fmt_pm(d_mean, d_std) if d_mean is not None else fmt_metric(pre_m - post_m)
            body.append([
                run_label, stack,
                fmt_pm(post_m, post_s), fmt_pm(pre_m, pre_s), delta_cell,
                fmt_pm(post_f, post_fs), fmt_pm(pre_f, pre_fs),
                "paired; multiseed mean ± SD",
            ])
            return
        if not post or not pre:
            missing.append("{0} {1}".format(run_label, stack))
            body.append([run_label, stack] + [MISSING] * 6)
            return
        comments.extend([_source_comment(post), _source_comment(pre)])
        dp = float(pre.get("AUPRC")) - float(post.get("AUPRC"))
        body.append([
            run_label, stack,
            fmt_metric(post.get("AUPRC")), fmt_metric(pre.get("AUPRC")), fmt_metric(dp),
            fmt_metric(post.get("F1")), fmt_metric(pre.get("F1")),
            "paired strong-run",
        ])

    add_pair("Small-HI", "Small-HI 40ep seed2", "embedding")
    add_pair("Small-HI", "Small-HI 40ep seed2", "embedding+raw")
    add_pair("Small-LI", "Small-LI multiseed", "embedding", seed=1, ep=20)
    add_pair("Small-LI", "Small-LI multiseed", "embedding+raw", seed=1, ep=20)

    if strict and missing:
        raise RuntimeError("Missing representation ablation rows: {0}".format(", ".join(missing)))

    footnotes = [
        "Δ AUPRC = pre-3h minus post-128 on paired rows.",
        "Small-LI Δ AUPRC uses mean ± sample SD over per-seed paired deltas when available.",
        "emb198 scout omitted from main table (diagnostic-only; see contrastive appendix if included).",
    ]
    return headers, body, footnotes, comments


def build_temporal_flow_ablation(
    payload: Dict[str, Any],
    include_provisional: bool,
    strict: bool,
) -> Tuple[List[str], List[List[str]], List[str], List[str]]:
    tf_val = payload.get("temporal_flow_validation") or {}
    passed = bool(tf_val.get("passed"))
    rows = _rows(payload)
    comments: List[str] = []
    missing: List[str] = []

    if not include_provisional and not passed:
        headers = ["Status", "Message"]
        body = [[
            "Pending validation",
            "Temporal-flow causal ablation results exist but are not table-eligible until "
            "temporal_flow_causal_validation_summary.json indicates validation passed. "
            "Re-run with --include_provisional to preview provisional metrics.",
        ]]
        footnotes = [
            "Comparisons use pre-3h stacks with and without temporal_flow_causal features.",
            "After validation passes, max_iter=5000 JSONs become the default canonical source.",
        ]
        return headers, body, footnotes, comments

    headers = [
        "Dataset / run", "Comparison", "Feature stack", "AUPRC",
        "Δ AUPRC vs pre-3h + raw", "F1", "P@100", "R@100", "Lift@100", "Validation status",
    ]
    body: List[List[str]] = []
    missing_r100: List[str] = []

    def arm_row(dataset: str, run_label: str, arm: str, delta: Optional[str] = None,
                ms: bool = False):
        if ms:
            mean, std = _tf_ms(payload, arm, "auprc")
            f_mean, f_std = _tf_ms(payload, arm, "f1_at_selected_threshold")
            p_mean, p_std = _tf_ms(payload, arm, "precision_at_100")
            r_mean, r_std = _tf_ms(payload, arm, "recall_at_100")
            l_mean, l_std = _tf_ms(payload, arm, "lift_at_100")
            if arm == "D":
                d_mean, d_std = _tf_d_minus_b_ms(payload, "auprc")
                delta_s = fmt_delta(d_mean, d_std) if d_mean is not None else (delta or MISSING)
            else:
                delta_s = delta or MISSING
            vs = "validated" if passed else "provisional"
            if mean is None:
                missing.append("{0} arm {1}".format(run_label, arm))
            body.append([
                run_label,
                TF_ARM_COMPARISON[arm],
                TF_ARM_STACK[arm],
                fmt_pm(mean, std) if mean is not None else MISSING,
                delta_s,
                fmt_pm(f_mean, f_std) if f_mean is not None else MISSING,
                fmt_pm(p_mean, p_std) if p_mean is not None else MISSING,
                fmt_pm(r_mean, r_std) if r_mean is not None else MISSING,
                fmt_pm(l_mean, l_std, kind="lift") if l_mean is not None else MISSING,
                vs,
            ])
            return
        row = _pick_temporal_arm(rows, dataset, arm, include_provisional or passed)
        if not row:
            missing.append("{0} {1}".format(run_label, TF_ARM_COMPARISON[arm]))
            body.append([
                run_label, TF_ARM_COMPARISON[arm], MISSING, MISSING, delta or MISSING,
                MISSING, MISSING, MISSING, MISSING, "missing",
            ])
            return
        comments.append(_source_comment(row))
        _warn_missing_r100("{0} {1}".format(run_label, TF_ARM_COMPARISON[arm]), row, missing_r100)
        b_row = _pick_temporal_arm(rows, dataset, "B", include_provisional or passed)
        if delta is None and b_row and row.get("AUPRC") is not None and b_row.get("AUPRC") is not None:
            delta = fmt_delta(float(row["AUPRC"]) - float(b_row["AUPRC"]))
        vs = row.get("validation_status", MISSING)
        if include_provisional and vs in ("provisional", "pending_validation"):
            vs = "{0} (preview)".format(vs)
        body.append([
            run_label,
            TF_ARM_COMPARISON[arm],
            row.get("probe_feature_stack", TF_ARM_STACK[arm]),
            fmt_metric(row.get("AUPRC")),
            delta or MISSING,
            fmt_metric(row.get("F1")),
            fmt_metric(_alert_metric(row, "precision_at_100")),
            fmt_metric(_alert_metric(row, "recall_at_100")),
            fmt_lift(_alert_metric(row, "lift_at_100")),
            vs,
        ])

    for arm in ("A", "B", "C", "D"):
        arm_row("Small-HI", "Small-HI 40ep seed2", arm, delta=MISSING if arm != "D" else None)
    arm_row("Small-LI", "Small-LI multiseed", "B", ms=True)
    arm_row("Small-LI", "Small-LI multiseed", "D", ms=True)

    if strict and missing:
        raise RuntimeError("Missing temporal-flow rows: {0}".format(", ".join(missing)))

    for msg in missing_r100:
        warnings.warn(msg)

    footnotes = _with_alert_budget_notes([
        "Primary comparison: pre-3h + raw + temporal-flow versus pre-3h + raw.",
        "Provisional rows shown only with --include_provisional until validation summary passes.",
        "Validated max_iter=5000 JSONs preferred when validation summary passes.",
    ])
    return headers, body, footnotes, comments


def build_supervised_vs_ssl(
    payload: Dict[str, Any],
    include_provisional: bool,
    strict: bool,
) -> Tuple[List[str], List[List[str]], List[str], List[str]]:
    rows = _rows(payload)
    headers = [
        "Dataset", "Method", "Training signal", "Encoder updated with labels?",
        "AUPRC", "F1", "P@100", "R@100", "Lift@100", "Caveat",
    ]
    body: List[List[str]] = []
    comments: List[str] = []
    missing: List[str] = []
    missing_r100: List[str] = []

    def add(dataset: str, method: str, signal: str, enc_upd: str, row: Optional[Dict[str, Any]]):
        if not row:
            missing.append(method)
            body.append([dataset, method, signal, enc_upd] + [MISSING] * 6)
            return
        comments.append(_source_comment(row))
        _warn_missing_r100("{0} / {1}".format(dataset, method), row, missing_r100)
        body.append([
            dataset, method, signal, enc_upd,
            fmt_metric(row.get("AUPRC")),
            fmt_metric(row.get("F1")),
            fmt_metric(_alert_metric(row, "precision_at_100")),
            fmt_metric(_alert_metric(row, "recall_at_100")),
            fmt_lift(_alert_metric(row, "lift_at_100")),
            _caveat_cell(row),
        ])

    legacy_li = _pick(
        rows, dataset="Small-LI", objective="supervised",
        threshold_rule="paper_argmax", scout_or_formal="formal",
    )
    legacy_hi = _pick(
        rows, dataset="Small-HI", objective="supervised",
        threshold_rule="paper_argmax", scout_or_formal="formal",
        source_json=HI_LEGACY_EVAL,
    )

    hi_tf = _pick_temporal_arm(rows, "Small-HI", "D", include_provisional)
    if hi_tf and _eligible(hi_tf, include_provisional):
        add(
            "Small-HI",
            "SSL pre-3h + raw + temporal-flow",
            "contrastive + frozen probe",
            "no (frozen probe)",
            hi_tf,
        )
    elif include_provisional:
        add(
            "Small-HI",
            "SSL pre-3h + raw + temporal-flow (provisional)",
            "contrastive + frozen probe",
            "no (frozen probe)",
            None,
        )

    add("Small-HI", "Legacy supervised GIN", "supervised CE (end-to-end)", "yes", legacy_hi)

    ssl_raw = _pick(rows, dataset="Small-LI", probe_feature_stack="embedding+raw",
                    representation_source="pre_embedding_3h", source_json=MULTISEED_LI, seed=1)
    if ssl_raw:
        mean, std = _ms(payload, "embedding+raw", "pre_embedding_3h", "auprc")
        f_mean, f_std = _ms(payload, "embedding+raw", "pre_embedding_3h", "f1_at_selected_threshold")
        comments.append("<!-- multiseed SSL pre-3h +raw -->")
        body.append([
            "Small-LI", "SSL pre-3h + raw (multiseed mean)",
            "contrastive + frozen probe", "no (frozen probe)",
            fmt_pm(mean, std), fmt_pm(f_mean, f_std),
            fmt_pm(*_ms(payload, "embedding+raw", "pre_embedding_3h", "precision_at_100")),
            fmt_pm(*_ms(payload, "embedding+raw", "pre_embedding_3h", "recall_at_100")),
            fmt_pm(*_ms(payload, "embedding+raw", "pre_embedding_3h", "lift_at_100"), kind="lift"),
            "val-tuned F1; frozen linear probe",
        ])
    else:
        missing.append("SSL pre-3h + raw")

    tf_d = _li_tf_arm_d_aggregate(payload)
    if tf_d:
        comments.append("<!-- multiseed: temporal-flow pre-3h + raw + temporal-flow Small-LI -->")
        body.append([
            "Small-LI",
            "SSL pre-3h + raw + temporal-flow (multiseed mean)",
            "contrastive + frozen probe",
            "no (frozen probe)",
            tf_d["auprc"],
            tf_d["f1"],
            tf_d["p100"],
            tf_d["r100"],
            tf_d["lift"],
            tf_d["caveat"],
        ])
    elif _pick_temporal_arm(rows, "Small-LI", "D", include_provisional) and include_provisional:
        tf = _pick_temporal_arm(rows, "Small-LI", "D", include_provisional)
        add("Small-LI", "SSL pre-3h + raw + temporal-flow (provisional)", "contrastive + frozen probe", "no", tf)

    add("Small-LI", "Legacy supervised GIN", "supervised CE (end-to-end)", "yes", legacy_li)

    if strict and missing:
        raise RuntimeError("Missing supervised vs SSL rows: {0}".format(", ".join(missing)))

    for msg in missing_r100:
        warnings.warn(msg)

    footnotes = _with_alert_budget_notes([
        "SSL rows use frozen linear probe with validation-tuned threshold.",
        "Supervised rows use end-to-end labeled training and paper_argmax F1.",
        "SSL and supervised F1 values are not directly comparable without the protocol caveat above.",
        "Small-LI SSL pre-3h + raw + temporal-flow uses validated temporal-flow multiseed aggregate (same as Table 5).",
        "Small-HI SSL temporal-flow row uses validated single-seed strong-run protocol when available.",
    ])
    return headers, body, footnotes, comments


def _tf_alert_budget_ms_cells(
    payload: Dict[str, Any],
    arm: str,
    ks: Sequence[int] = ALERT_BUDGET_KS,
) -> List[str]:
    cells: List[str] = []
    for k in ks:
        cells.extend([
            fmt_pm(*_tf_ms(payload, arm, "precision_at_{0}".format(k))),
            fmt_pm(*_tf_ms(payload, arm, "recall_at_{0}".format(k))),
            fmt_pm(*_tf_ms(payload, arm, "lift_at_{0}".format(k)), kind="lift"),
        ])
    return cells


def build_alert_budget_appendix(
    payload: Dict[str, Any],
    include_provisional: bool,
    strict: bool,
) -> Tuple[List[str], List[List[str]], List[str], List[str]]:
    rows = _rows(payload)
    headers = [
        "Dataset", "Method",
        "P@100", "R@100", "Lift@100",
        "P@500", "R@500", "Lift@500",
        "P@1000", "R@1000", "Lift@1000",
        "Caveat",
    ]
    body: List[List[str]] = []
    comments: List[str] = []
    missing: List[str] = []

    def add_row(
        dataset: str,
        method: str,
        cells: Sequence[str],
        caveat: str = MISSING,
        comment: str = "",
    ) -> None:
        if comment:
            comments.append(comment)
        body.append([dataset, method, *cells, caveat])

    def add_registry_row(dataset: str, method: str, row: Optional[Dict[str, Any]], caveat: str = ""):
        if not row:
            missing.append("{0} {1}".format(dataset, method))
            add_row(dataset, method, [MISSING] * 9, caveat or MISSING)
            return
        add_row(
            dataset, method, _alert_budget_cells(row),
            _caveat_cell(row, caveat),
            _source_comment(row),
        )

    # Small-HI curated rows
    add_registry_row(
        "Small-HI", "Raw features only",
        _pick(rows, probe_feature_stack="raw", dataset="Small-HI", training_epochs=20, seed=1),
        "no SSL",
    )
    add_registry_row(
        "Small-HI", "Raw + morphology",
        _pick(rows, probe_feature_stack="raw+morph", dataset="Small-HI", training_epochs=20, seed=1),
        "no SSL",
    )
    for label, rep, stack in (
        ("SSL post-128", "post-128", "embedding"),
        ("SSL pre-3h", "pre-3h", "embedding"),
        ("SSL post-128 + raw", "post-128", "embedding+raw"),
        ("SSL pre-3h + raw", "pre-3h", "embedding+raw"),
    ):
        add_registry_row(
            "Small-HI", label,
            _pick(rows, dataset="Small-HI", representation_source=_registry_rep(rep),
                  probe_feature_stack=stack, training_epochs=40, seed=2,
                  source_json=CANONICAL_PREPOST),
        )
    hi_tf = _pick_temporal_arm(rows, "Small-HI", "D", include_provisional)
    if hi_tf and _eligible(hi_tf, include_provisional):
        add_registry_row("Small-HI", "SSL pre-3h + raw + temporal-flow", hi_tf, "validated temporal-flow stack")
    hi_legacy = _pick(
        rows, dataset="Small-HI", objective="supervised",
        threshold_rule="paper_argmax", scout_or_formal="formal",
        source_json=HI_LEGACY_EVAL,
    )
    if hi_legacy:
        add_registry_row("Small-HI", "Legacy supervised GIN (100ep seed1)", hi_legacy)

    # Small-LI multiseed SSL rows
    for label, rep_display, stack in (
        ("SSL post-128", "post-128", "embedding"),
        ("SSL pre-3h", "pre-3h", "embedding"),
        ("SSL post-128 + raw", "post-128", "embedding+raw"),
        ("SSL pre-3h + raw", "pre-3h", "embedding+raw"),
    ):
        rep = _registry_rep(rep_display)
        cells = _alert_budget_ms_cells(payload, stack, rep)
        if all(c == MISSING for c in cells):
            missing.append("Small-LI {0}".format(label))
            add_row("Small-LI", label, [MISSING] * 9)
        else:
            comments.append("<!-- multiseed alert-budget: {0} {1} -->".format(stack, rep))
            add_row(
                "Small-LI", label, cells,
                "frozen probe; mean ± sample SD (n=3)",
            )

    tf_d_cells = _tf_alert_budget_ms_cells(payload, "D")
    if any(c != MISSING for c in tf_d_cells):
        comments.append("<!-- multiseed alert-budget: temporal-flow Arm D -->")
        add_row(
            "Small-LI", "SSL pre-3h + raw + temporal-flow",
            tf_d_cells,
            "val-tuned F1; mean ± sample SD (n=3)",
        )

    li_legacy = _pick(
        rows, dataset="Small-LI", objective="supervised",
        threshold_rule="paper_argmax", scout_or_formal="formal",
    )
    if li_legacy:
        add_registry_row("Small-LI", "Legacy supervised GIN (100ep seed1)", li_legacy)

    if strict and missing:
        raise RuntimeError("Missing alert-budget appendix rows: {0}".format(", ".join(missing)))

    footnotes = _with_alert_budget_notes([
        "Fixed top-K alert-budget metrics on the test split; threshold-tuned precision/recall are omitted.",
        "Small-LI SSL rows use mean ± sample SD (ddof=1) over seeds 1–3 where available.",
        "K=500 and K=1000 may be unavailable (—) for some multiseed aggregates when not present in registry summaries.",
    ])
    return headers, body, footnotes, comments


def build_architecture_ablation(
    payload: Dict[str, Any],
) -> Tuple[List[str], List[List[str]], List[str], List[str]]:
    rows = _rows(payload)
    headers = [
        "Encoder", "Hidden dim", "Pre dim", "Post dim", "Params",
        "AUROC", "AUPRC", "F1", "Caveat",
    ]
    body: List[List[str]] = []
    comments: List[str] = []
    pending: List[str] = []

    arch_rows = _filter_rows(
        rows, include_provisional=True,
        source_json=ARCH_SWEEP,
        dataset="Small-HI",
        probe_feature_stack="embedding",
        representation_source="post_embedding_128",
    )
    by_encoder = {r.get("encoder"): r for r in arch_rows if r.get("AUROC") is not None}

    for enc in ARCH_ENCODER_ORDER:
        r = by_encoder.get(enc)
        if not r:
            pending.append("{0} (architecture sweep)".format(enc))
            continue
        comments.append(_source_comment(r))
        geom = ARCH_GEOMETRY.get(enc, {})
        caveat = "not capacity-matched to GIN" if enc == "pna" else MISSING
        params = geom.get("params")
        body.append([
            enc,
            str(geom.get("hidden")) if geom.get("hidden") is not None else MISSING,
            str(geom.get("pre")) if geom.get("pre") is not None else MISSING,
            str(geom.get("post")) if geom.get("post") is not None else MISSING,
            str(params) if params is not None else MISSING,
            fmt_metric(r.get("AUROC")),
            fmt_metric(r.get("AUPRC")),
            fmt_metric(r.get("F1")),
            caveat,
        ])

    width_rows = _filter_rows(
        rows, include_provisional=True,
        source_json=PNA_WIDTH_ALIGNED,
        dataset="Small-HI",
        probe_feature_stack="embedding",
        representation_source="post_embedding_128",
    )
    if width_rows:
        wr = width_rows[0]
        comments.append(_source_comment(wr))
        body.append([
            "pna (width-aligned)",
            "65",
            "195",
            "128",
            MISSING,
            fmt_metric(wr.get("AUROC")),
            fmt_metric(wr.get("AUPRC")),
            fmt_metric(wr.get("F1")),
            "GIN-matched LR/dropout; seed 1 scout",
        ])
    else:
        pending.append("PNA width-aligned (pending result JSON)")

    tf_rows = _filter_rows(
        rows, include_provisional=True,
        source_json=PNA_WIDTH_TF_PROBE,
        dataset="Small-HI",
        probe_feature_stack="embedding+raw+temporal_flow_causal",
        representation_source="pre_embedding_3h",
    )
    if tf_rows:
        tr = tf_rows[0]
        comments.append(_source_comment(tr))
        body.append([
            "pna (width-aligned, best stack)",
            "65",
            "195",
            "128",
            MISSING,
            fmt_metric(tr.get("AUROC")),
            fmt_metric(tr.get("AUPRC")),
            fmt_metric(tr.get("F1")),
            "pre-3h+raw+temporal-flow; one seed; downstream-only diagnostic",
        ])

    footnotes = [
        "Comparable rows only: embedding-only, post-128, shared probe settings, Small-HI architecture sweep ({0}).".format(ARCH_SWEEP),
        "Default PNA (hidden 20, pre dim 60) was not capacity/hyperparameter matched to GIN (hidden 66, pre dim 198).",
    ]
    if pending:
        footnotes.append("Pending/manual review: {0}.".format("; ".join(pending)))
    return headers, body, footnotes, comments, pending


def build_contrastive_ablations(
    payload: Dict[str, Any],
) -> Tuple[List[str], List[List[str]], List[str], List[str]]:
    rows = _rows(payload)
    headers = [
        "Variant", "Dataset", "Representation", "Feature stack",
        "AUROC", "AUPRC", "F1", "Takeaway",
    ]
    body: List[List[str]] = []
    comments: List[str] = []
    pending: List[str] = []

    curated = [
        (
            "GIN baseline (20ep)",
            {
                "source_json": ARCH_SWEEP,
                "encoder": "gin",
                "probe_feature_stack": "embedding",
                "representation_source": "post_embedding_128",
            },
            "embedding-only SSL baseline from architecture sweep",
        ),
        (
            "FNF full stack",
            {
                "run_id": "same_pair_fnf_emlps_tds_asym_proj_8192neg_queue0_20ep|embedding+raw+morph|post_embedding_128",
                "source_json": CANONICAL_PREPOST,
            },
            "FNF contrastive variant; +raw+morph (not comparable to embedding-only baseline)",
        ),
        (
            "degree-aware edge-drop",
            {
                "run_id": "degree_aware_edgedrop_emlps_tds_asym_proj_8192neg_queue0_20ep|embedding|post_embedding_128",
                "thesis_role": "negative_result",
            },
            "embedding-only negative result; no gain vs baseline",
        ),
    ]
    for label, kw, takeaway in curated:
        row = _pick(rows, include_provisional=True, **kw)
        if not row:
            pending.append(label)
            continue
        comments.append(_source_comment(row))
        body.append([
            label,
            row.get("dataset", MISSING),
            REP_DISPLAY.get(row.get("representation_source", ""), row.get("representation_source", MISSING)),
            row.get("probe_feature_stack", MISSING),
            fmt_metric(row.get("AUROC")),
            fmt_metric(row.get("AUPRC")),
            fmt_metric(row.get("F1")),
            takeaway,
        ])

    emb198 = _pick(
        rows, include_provisional=True,
        source_json=EMB198_SOURCE,
        probe_feature_stack="embedding+raw",
        representation_source="pre_embedding_3h_emb198_scout",
        thesis_role="diagnostic",
    )
    if emb198:
        comments.append(_source_comment(emb198))
        body.append([
            "emb198 scout (Small-LI)",
            emb198.get("dataset", MISSING),
            "pre-3h emb198 scout",
            "embedding+raw",
            fmt_metric(emb198.get("AUROC")),
            fmt_metric(emb198.get("AUPRC")),
            fmt_metric(emb198.get("F1")),
            "one-seed diagnostic scout; not multiseed canonical",
        ])
    else:
        pending.append("emb198 scout (accurate scout row not found)")

    for item in (
        "queue-size contrastive variants",
        "multi-positive contrastive variants",
        "KNN positive variants",
        "morphology auxiliary-loss variants",
    ):
        pending.append(item)

    footnotes = [
        "Appendix rows are curated for interpretability; raw-only rows are not compared directly to embedding-only SSL baselines.",
    ]
    if pending:
        footnotes.append("Pending/manual review: {0}.".format("; ".join(pending)))
    return headers, body, footnotes, comments


TABLE_BUILDERS: Dict[str, Callable[..., Tuple[List[str], List[List[str]], List[str], List[str]]]] = {
    "dataset_summary": lambda p, ip, st: build_dataset_summary(p) + ([],),
    "main_results_small_hi": build_main_small_hi,
    "main_results_small_li": lambda p, ip, st: build_main_small_li(p, st) ,
    "representation_readout_ablation": lambda p, ip, st: build_representation_ablation(p, st),
    "temporal_flow_ablation": build_temporal_flow_ablation,
    "supervised_vs_ssl": build_supervised_vs_ssl,
    "alert_budget_performance_appendix": build_alert_budget_appendix,
    "architecture_ablation_appendix": lambda p, ip, st: build_architecture_ablation(p) + ([],),
    "contrastive_ablations_appendix": lambda p, ip, st: build_contrastive_ablations(p) + ([],),
}


def write_table_files(
    name: str,
    caption: str,
    headers: List[str],
    body: List[List[str]],
    footnotes: List[str],
    comments: List[str],
    out_dir: Path,
    formats: Sequence[str],
    include_sources: bool = False,
) -> List[Path]:
    written: List[Path] = []
    md_lines = ["# {0}".format(caption), ""]
    if include_sources and comments:
        md_lines.extend([c for c in comments if c])
        md_lines.append("")
    md_lines.append(_md_table(headers, body, footnotes))
    if "md" in formats or "both" in formats:
        md_path = out_dir / "{0}.md".format(name)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        written.append(md_path)
    if "tex" in formats or "both" in formats:
        tex_path = out_dir / "{0}.tex".format(name)
        tex_path.write_text(_tex_table(caption, headers, body, footnotes) + "\n", encoding="utf-8")
        written.append(tex_path)
    return written


def build_preview(
    payload: Dict[str, Any],
    include_provisional: bool,
    strict: bool,
    formats: Sequence[str],
    out_dir: Path,
    include_sources: bool = False,
) -> Tuple[str, List[Path]]:
    sections = []
    all_written: List[Path] = []
    captions = {
        "dataset_summary": "Table 1 — Dataset summary",
        "main_results_small_hi": "Table 2 — Main Small-HI results",
        "main_results_small_li": "Table 3 — Main Small-LI results",
        "representation_readout_ablation": "Table 4 — Representation readout ablation",
        "temporal_flow_ablation": "Table 5 — Temporal-flow ablation",
        "supervised_vs_ssl": "Table 6 — Supervised versus frozen SSL",
        "alert_budget_performance_appendix": "Appendix — Alert-budget performance",
        "architecture_ablation_appendix": "Appendix — Architecture ablation",
        "contrastive_ablations_appendix": "Appendix — Contrastive and diagnostic ablations",
    }
    for name, builder in TABLE_BUILDERS.items():
        result = builder(payload, include_provisional, strict)
        headers, body, footnotes = result[:3]
        comments = result[3] if len(result) > 3 else []
        caption = captions[name]
        written = write_table_files(
            name, caption, headers, body, footnotes, comments, out_dir, formats, include_sources,
        )
        all_written.extend(written)
        sections.append("## {0}\n\n{{include}} tables/{0}.md\n".format(name))
    tf_val = payload.get("temporal_flow_validation") or {}
    header = [
        "# Thesis tables preview",
        "",
        "Auto-generated from `{0}`.".format("results/diagnostics/thesis_experiment_registry.json"),
        "",
        "- **Rows in registry:** {0}".format(payload.get("row_count")),
        "- **Include provisional:** {0}".format(include_provisional),
        "- **Temporal-flow validation passed:** {0}".format(tf_val.get("passed")),
        "",
    ]
    if payload.get("pending_sources"):
        header.append("### Pending optional sources")
        for p in payload["pending_sources"]:
            header.append("- `{0}`".format(p))
        header.append("")
    content_parts = []
    for name in TABLE_BUILDERS:
        md_path = out_dir / "{0}.md".format(name)
        if md_path.is_file():
            text = md_path.read_text(encoding="utf-8")
            if not include_sources:
                text = _strip_source_comments(text)
            lines = text.splitlines()
            if lines and lines[0].startswith("# "):
                text = "\n".join(lines[1:]).lstrip()
            content_parts.append("## {0}\n\n".format(captions[name]) + text)
    preview = "\n".join(header) + "\n".join(content_parts)
    return preview, all_written


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry", default="results/diagnostics/thesis_experiment_registry.json")
    p.add_argument("--out_dir", default="tables")
    p.add_argument("--preview", default="notes/thesis_tables_preview.md")
    p.add_argument("--include_provisional", action="store_true")
    p.add_argument("--include_sources", action="store_true",
                   help="Include HTML source comments in table markdown and preview")
    p.add_argument("--format", default="both", help="md,tex,both")
    p.add_argument("--strict", action="store_true")
    args = p.parse_args(argv)

    registry_path = ROOT / args.registry
    out_dir = ROOT / args.out_dir
    preview_path = ROOT / args.preview
    formats = [x.strip() for x in args.format.split(",")]

    if not registry_path.is_file():
        print("Registry not found: {0}".format(registry_path), file=sys.stderr)
        return 1

    payload = _load_registry(registry_path)
    try:
        preview, written = build_preview(
            payload, args.include_provisional, args.strict, formats, out_dir,
            include_sources=args.include_sources,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    preview_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.write_text(preview, encoding="utf-8")
    print("Wrote preview: {0}".format(preview_path))
    for w in written:
        try:
            print("  {0}".format(w.relative_to(ROOT)))
        except ValueError:
            print("  {0}".format(w))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
