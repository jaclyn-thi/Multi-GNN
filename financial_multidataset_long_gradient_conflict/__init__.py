"""No-update gradient-conflict diagnostic for MIXED_3DOMAIN_LONG (constants)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONTRACT_ID = "financial_multidataset_shared_core_v1"
ARM = "MIXED_3DOMAIN_LONG"
SEED = 2
N_BATCHES_PER_DOMAIN = 8
BATCH_SIZE = 8192
NUM_NEIGHS = [100, 100]
TEMP = 0.5
N_NEG = 8192
DOMAINS = ("Small-HI", "SAML-D", "Small-LI")

CKPT_ROOT = (
    ROOT
    / "results/checkpoints/financial_multidataset_shared_core_phase4b_mixed_long_3000_seed2"
    / "mixed_3domain_long"
)
CHECKPOINTS = {
    1500: {
        "path": CKPT_ROOT / "checkpoint_step_1500.tar",
        "sha256": "85e71a42cbcbf22564b857e1119847ea9099dba04301543621481f05db7ec0aa",
        "step": 1500,
    },
    3000: {
        "path": CKPT_ROOT / "checkpoint_step_3000.tar",
        "sha256": "092a8c1159dc8b16786902c7e204861f6c08c7ee2ccfef1b8d2c878d4bd25fb1",
        "step": 3000,
    },
}

OUT_ROOT = ROOT / "results/diagnostics/financial_multidataset_long_gradient_conflict"

# Cosine interpretation bins (predeclared)
COS_CONFLICT = -0.10
COS_ALIGN = 0.10

# Reconstruction relative tolerance
RECON_RTOL = 1e-4
RECON_ATOL = 1e-6

TF_NAMES = (
    "log1p_sender_interarrival",
    "log1p_sender_past_7d_count",
    "log1p_amount_vs_sender_past_mean",
)
