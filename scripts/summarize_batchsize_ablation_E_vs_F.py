#!/usr/bin/env python3
"""Summarize E vs F batch-size ablation (all-neg, corrected-TDS+preserve, seed2).

Does not resume training. Reads train logs, resolved markers, probe/ablation artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from linear_probe import load_embedding_npz

RUNS = {
    "E": "gin_emlps_ports_tds_corrected_preserve_seed_asym_proj_allneg_bs8192_accum4_10ep_seed2",
    "F": "gin_emlps_ports_tds_corrected_preserve_seed_asym_proj_allneg_bs2048_accum16_10ep_seed2",
}


def _mean_std(xs: List[float]) -> Dict[str, Optional[float]]:
    arr = np.asarray([x for x in xs if x is not None and math.isfinite(x)], dtype=float)
    if arr.size == 0:
        return {"mean": None, "std": None, "n": 0, "values": []}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
        "values": arr.tolist(),
    }


def parse_train_log(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"missing": True, "path": str(path)}
    text = path.read_text(errors="replace")
    losses = [float(x) for x in re.findall(r"Train Loss:\s*([0-9.]+)", text)]
    diags = []
    pat = re.compile(
        r"Contrastive epoch diagnostics: epoch=(\d+) microbatches=(\d+) optimizer_steps=(\d+) "
        r"shared_anchors_total=(\d+) mean_requested_seeds=([0-9.]+) mean_shared_seeds=([0-9.]+) "
        r"shared_anchors_per_opt_update≈([0-9.]+) denom_mode=(\S+) unique_negs_per_anchor≈([0-9.]+) "
        r"duplicate_neg_count=(\S+)"
    )
    for m in pat.finditer(text):
        diags.append(
            {
                "epoch": int(m.group(1)),
                "microbatches": int(m.group(2)),
                "optimizer_steps": int(m.group(3)),
                "shared_anchors_total": int(m.group(4)),
                "mean_requested_seeds": float(m.group(5)),
                "mean_shared_seeds": float(m.group(6)),
                "shared_anchors_per_opt_update": float(m.group(7)),
                "denom_mode": m.group(8),
                "unique_negs_per_anchor": float(m.group(9)),
                "duplicate_neg_count": float(m.group(10)) if m.group(10) not in ("nan", "None") else None,
            }
        )
    first_shared = None
    m0 = re.search(
        r"Hetero contrastive seed-edge filtering: requested_seed_edges=(\d+) shared_seed_edges=(\d+)",
        text,
    )
    if m0:
        first_shared = {
            "requested_seed_edges": int(m0.group(1)),
            "shared_seed_edges": int(m0.group(2)),
        }
    denom = None
    md = re.search(r"Contrastive denom mode: (\S+)", text)
    if md:
        denom = md.group(1)
    return {
        "path": str(path),
        "train_loss_by_epoch": losses,
        "epoch_diagnostics": diags,
        "first_batch_seed_filter": first_shared,
        "denom_mode_logged": denom,
        "n_epochs_logged": len(losses),
    }


def embedding_variance(emb_dir: Path) -> Dict[str, Any]:
    out = {}
    for split in ("train", "val", "test"):
        p = emb_dir / f"{split}.npz"
        if not p.is_file():
            continue
        z, y, _ = load_embedding_npz(p)
        # L2-normalize like InfoNCE path for collapse diagnostics
        norms = np.linalg.norm(z, axis=1, keepdims=True)
        zn = z / np.clip(norms, 1e-12, None)
        # mean pairwise cosine via (mean embedding) proxy: std of dims + mean ||z||
        out[split] = {
            "n": int(z.shape[0]),
            "dim": int(z.shape[1]),
            "mean_l2_norm": float(norms.mean()),
            "std_l2_norm": float(norms.std()),
            "mean_feature_std": float(z.std(axis=0).mean()),
            "min_feature_std": float(z.std(axis=0).min()),
            "frac_near_zero_feature_std": float((z.std(axis=0) < 1e-4).mean()),
            "mean_abs_cosine_to_mean": float(
                np.abs((zn * zn.mean(axis=0, keepdims=True)).sum(axis=1)).mean()
            ),
            "positive_rate": float(y.mean()) if y.size else None,
        }
    return out


def load_probe(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def load_ablation(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def extract_test_metrics(probe: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not probe:
        return {}
    block = probe.get("splits_at_selected_threshold", {}).get("test", {})
    return {
        "auroc": block.get("auroc"),
        "auprc": block.get("auprc"),
        "f1": block.get("f1"),
        "precision": block.get("precision"),
        "recall": block.get("recall"),
        "threshold": probe.get("classification_threshold", {}).get("value"),
    }


def ablation_arm(abl: Optional[Dict[str, Any]], arm_key: str) -> Dict[str, Any]:
    if not abl:
        return {}
    # tolerate nested schemas
    arms = abl.get("arms") or abl.get("results") or abl
    if isinstance(arms, dict) and arm_key in arms:
        node = arms[arm_key]
    else:
        # try list of arms
        node = None
        for cand in (abl.get("modes"), abl.get("feature_modes"), []):
            if isinstance(cand, list):
                for row in cand:
                    if row.get("name") == arm_key or row.get("mode") == arm_key:
                        node = row
                        break
        if node is None:
            return {}
    test = node.get("test") or node.get("splits", {}).get("test") or node
    ranking = test.get("ranking") or test.get("ranking_metrics") or {}
    return {
        "auroc": test.get("auroc"),
        "auprc": test.get("auprc"),
        "f1": test.get("f1"),
        "precision": test.get("precision"),
        "recall": test.get("recall"),
        "P@100": ranking.get("precision_at_100") or test.get("precision_at_100"),
        "R@100": ranking.get("recall_at_100") or test.get("recall_at_100"),
    }


def decide(payload: Dict[str, Any]) -> Dict[str, Any]:
    e = payload["variants"]["E"]
    f = payload["variants"]["F"]
    e_emb = e.get("probe_post", {})
    f_emb = f.get("probe_post", {})
    primaries = ["auroc", "auprc", "f1"]
    deltas = {k: (None if e_emb.get(k) is None or f_emb.get(k) is None else f_emb[k] - e_emb[k]) for k in primaries}
    material = 0.01
    improved = [k for k, d in deltas.items() if d is not None and d >= material]
    regressed = [k for k, d in deltas.items() if d is not None and d <= -material]
    if not e_emb or not f_emb:
        verdict = "incomplete"
        rationale = "Missing probe metrics for E and/or F."
    elif improved and not regressed:
        verdict = "promising"
        rationale = (
            f"F improved {improved} by ≥{material} vs E with no material regression among {primaries}."
        )
    elif improved and regressed:
        verdict = "mixed"
        rationale = f"F improved {improved} but regressed {regressed}; not auto-promoted."
    else:
        verdict = "not_promising"
        rationale = (
            f"F did not materially improve primary embedding-only metrics vs E "
            f"(deltas={deltas}). Do not auto-resume to 40ep."
        )
    return {"verdict": verdict, "rationale": rationale, "deltas_F_minus_E": deltas}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_json", default="results/diagnostics/batchsize_ablation_E_vs_F_seed2.json")
    p.add_argument("--output_md", default="notes/batchsize_ablation_E_vs_F_seed2.md")
    args = p.parse_args()

    variants = {}
    for tag, run in RUNS.items():
        emb = Path("embeddings") / run
        variants[tag] = {
            "run_name": run,
            "resolved": load_probe(Path(f"results/diagnostics/batchsize_ablation_{tag}_resolved_run_seed2.json")),
            "train_log": parse_train_log(Path(f"logs/{run}_train.log")),
            "peak_gpu_mem_mib": None,
            "probe_post": extract_test_metrics(load_probe(emb / "probe_results.json")),
            "probe_pre3h": extract_test_metrics(
                load_probe(emb / "pre_embedding_3h" / "probe_results.json")
            ),
            "ablation": {},
            "embedding_variance_post": embedding_variance(emb),
            "embedding_variance_pre3h": embedding_variance(emb / "pre_embedding_3h"),
        }
        resolved = variants[tag]["resolved"] or {}
        variants[tag]["peak_gpu_mem_mib"] = resolved.get("peak_gpu_mem_mib")
        abl = load_ablation(Path(f"results/diagnostics/probe_feature_ablation_{run}.json"))
        if abl:
            # Discover arm names
            for key in (
                "embedding",
                "A_embedding",
                "embedding_only",
                "emb",
                "B_embedding_raw",
                "embedding_raw",
                "emb+raw",
            ):
                arm = ablation_arm(abl, key)
                if arm:
                    variants[tag]["ablation"][key] = arm

    payload = {
        "title": "Batch-size ablation E vs F (all-neg, corrected-TDS+preserve, seed2, 10ep)",
        "question": (
            "Does reducing seed batch from 8192 to 2048 improve representation quality when "
            "optimizer-update frequency and all-in-batch negative semantics are controlled?"
        ),
        "controls": {
            "E": {"batch_size": 8192, "accum": 4, "num_neg": 0},
            "F": {"batch_size": 2048, "accum": 16, "num_neg": 0},
            "requested_anchors_per_optimizer_update": 32768,
            "note": "Do not compare raw InfoNCE magnitudes across E/F without denom-size adjustment.",
        },
        "variants": variants,
    }
    payload["decision"] = decide(payload)

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    d = payload["decision"]
    lines = [
        "# Batch-size ablation E vs F (seed 2)",
        "",
        payload["question"],
        "",
        "## Controls",
        "",
        "- E: bs=8192, accum=4, `num_neg=0` (all aligned current-batch)",
        "- F: bs=2048, accum=16, `num_neg=0`",
        "- Requested anchors/optimizer update: 32768 for both",
        "- Corrected reverse TDS + preserve_seed; asymmetric; projection; queue=0; identity-only",
        "- No KNN / structural positives / morphology / third view",
        "- 10 epochs; do not auto-resume to 40",
        "",
        f"**Decision: {d['verdict']}** — {d['rationale']}",
        "",
        "### Embedding-only deltas (F − E)",
        "",
        f"```json\n{json.dumps(d['deltas_F_minus_E'], indent=2)}\n```",
        "",
    ]
    for tag in ("E", "F"):
        v = variants[tag]
        lines += [
            f"## Variant {tag}: `{v['run_name']}`",
            "",
            f"- peak GPU MiB: {v.get('peak_gpu_mem_mib')}",
            f"- probe post: {json.dumps(v.get('probe_post'))}",
            f"- probe pre3h: {json.dumps(v.get('probe_pre3h'))}",
            f"- ablation arms: {json.dumps(v.get('ablation'))}",
            "",
            "### Train diagnostics (parsed)",
            "",
            "```json",
            json.dumps(
                {
                    "n_epochs_logged": v["train_log"].get("n_epochs_logged"),
                    "train_loss_by_epoch": v["train_log"].get("train_loss_by_epoch"),
                    "first_batch_seed_filter": v["train_log"].get("first_batch_seed_filter"),
                    "epoch_diagnostics_head": (v["train_log"].get("epoch_diagnostics") or [])[:2],
                    "epoch_diagnostics_tail": (v["train_log"].get("epoch_diagnostics") or [])[-2:],
                },
                indent=2,
            ),
            "```",
            "",
            "### Representation variance",
            "",
            "```json",
            json.dumps(v.get("embedding_variance_post"), indent=2),
            "```",
            "",
        ]
    out_md.write_text("\n".join(lines))
    print(out_json)
    print(out_md)
    print("DECISION", d["verdict"])


if __name__ == "__main__":
    main()
