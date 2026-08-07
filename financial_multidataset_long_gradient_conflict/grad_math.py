"""Pure gradient algebra helpers for the no-update conflict diagnostic."""

from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


def classify_cosine(c: float, *, conflict: float = -0.10, align: float = 0.10) -> str:
    if c < conflict:
        return "conflicting"
    if c > align:
        return "aligned"
    return "approximately_orthogonal"


def cosine_from_norms_dot(dot: float, n1: float, n2: float, *, eps: float = 1e-12) -> float:
    if n1 < eps or n2 < eps:
        return float("nan")
    return float(dot / (n1 * n2))


def accumulate_grad_stats(
    grads: Sequence[Optional[torch.Tensor]],
) -> Tuple[float, float]:
    """Return (L2 norm, sum of squares) over a list of optional per-parameter grads."""
    ss = 0.0
    for g in grads:
        if g is None:
            continue
        ss += float(g.detach().float().pow(2).sum().item())
    return float(ss ** 0.5), ss


def accumulate_dot(
    a: Sequence[Optional[torch.Tensor]],
    b: Sequence[Optional[torch.Tensor]],
) -> float:
    dot = 0.0
    for ga, gb in zip(a, b):
        if ga is None or gb is None:
            continue
        dot += float((ga.detach().float() * gb.detach().float()).sum().item())
    return float(dot)


def scale_grads(
    grads: Sequence[Optional[torch.Tensor]], scale: float
) -> List[Optional[torch.Tensor]]:
    out: List[Optional[torch.Tensor]] = []
    for g in grads:
        if g is None:
            out.append(None)
        else:
            out.append(g.detach() * float(scale))
    return out


def add_grads(
    a: Sequence[Optional[torch.Tensor]],
    b: Sequence[Optional[torch.Tensor]],
) -> List[Optional[torch.Tensor]]:
    out: List[Optional[torch.Tensor]] = []
    for ga, gb in zip(a, b):
        if ga is None and gb is None:
            out.append(None)
        elif ga is None:
            out.append(gb.detach().clone() if gb is not None else None)
        elif gb is None:
            out.append(ga.detach().clone())
        else:
            out.append(ga.detach() + gb.detach())
    return out


def grad_diff_norm(
    a: Sequence[Optional[torch.Tensor]],
    b: Sequence[Optional[torch.Tensor]],
) -> float:
    ss = 0.0
    for ga, gb in zip(a, b):
        if ga is None and gb is None:
            continue
        if ga is None:
            ss += float(gb.detach().float().pow(2).sum().item())
        elif gb is None:
            ss += float(ga.detach().float().pow(2).sum().item())
        else:
            ss += float((ga.detach().float() - gb.detach().float()).pow(2).sum().item())
    return float(ss ** 0.5)


def reconstruction_ok(
    recon: Sequence[Optional[torch.Tensor]],
    direct: Sequence[Optional[torch.Tensor]],
    *,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> Dict[str, float | bool]:
    n_direct, _ = accumulate_grad_stats(direct)
    n_recon, _ = accumulate_grad_stats(recon)
    n_diff = grad_diff_norm(recon, direct)
    rel = n_diff / max(n_direct, 1e-12)
    ok = bool(n_diff <= atol + rtol * max(n_direct, n_recon))
    return {
        "ok": ok,
        "diff_l2": n_diff,
        "direct_l2": n_direct,
        "recon_l2": n_recon,
        "rel_error": rel,
        "rtol": rtol,
        "atol": atol,
    }


def encoder_parameters(model: nn.Module) -> List[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


@torch.no_grad()
def state_tensor_sha256(module: nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items(), key=lambda kv: kv[0]):
        h.update(k.encode("utf-8"))
        t = v.detach().cpu().contiguous()
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def bn_bundle_sha256(bn: Dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k in sorted(bn):
        h.update(k.encode("utf-8"))
        t = bn[k].detach().cpu().contiguous()
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha256(t: torch.Tensor) -> str:
    h = hashlib.sha256()
    x = t.detach().cpu().contiguous()
    h.update(str(tuple(x.shape)).encode("utf-8"))
    h.update(str(x.dtype).encode("utf-8"))
    h.update(x.numpy().tobytes())
    return h.hexdigest()


def refuse_optimizer_step(_optimizer=None) -> None:
    """Hard guard: any attempt to step must fail the diagnostic."""
    raise RuntimeError("REFUSED: optimizer.step is forbidden in no-update gradient diagnostic")


def refuse_test_split(split_name: str) -> None:
    if str(split_name).lower() in {"test", "testing", "holdout_test"}:
        raise RuntimeError(f"REFUSED: test split access forbidden ({split_name})")


def summarize_cosines(
    values: Iterable[float], *, conflict: float = -0.10, align: float = 0.10
) -> Dict[str, float]:
    xs = [float(v) for v in values if v == v]  # drop NaN
    n = len(xs)
    if n == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "frac_conflicting": float("nan"),
            "frac_orthogonal": float("nan"),
            "frac_aligned": float("nan"),
        }
    t = torch.tensor(xs, dtype=torch.float64)
    labels = [classify_cosine(v, conflict=conflict, align=align) for v in xs]
    return {
        "n": float(n),
        "mean": float(t.mean()),
        "median": float(t.median()),
        "std": float(t.std(unbiased=False)) if n > 1 else 0.0,
        "min": float(t.min()),
        "max": float(t.max()),
        "frac_conflicting": float(sum(1 for L in labels if L == "conflicting") / n),
        "frac_orthogonal": float(
            sum(1 for L in labels if L == "approximately_orthogonal") / n
        ),
        "frac_aligned": float(sum(1 for L in labels if L == "aligned") / n),
    }
