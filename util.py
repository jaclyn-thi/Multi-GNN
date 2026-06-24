import argparse
import numpy as np
import torch
import random
import logging
import os
import sys

def logger_setup():
    # Setup logging
    log_directory = "logs"
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-5.5s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_directory, "logs.log")),     ## log to local log file
            logging.StreamHandler(sys.stdout)          ## log also to stdout (i.e., print to screen)
        ]
    )

def create_parser():
    parser = argparse.ArgumentParser()

    #Adaptations
    parser.add_argument("--emlps", action='store_true', help="Use emlps in GNN training")
    parser.add_argument("--reverse_mp", action='store_true', help="Use heterogeneous graph form with reverse message passing (graph structure only).")
    parser.add_argument(
        "--objective",
        type=str,
        choices=["contrastive", "supervised"],
        default="contrastive",
        help="Training objective: contrastive pretraining (default) or supervised AML edge classification.",
    )
    parser.add_argument("--ports", action='store_true', help="Use port numberings in GNN training")
    parser.add_argument("--tds", action='store_true', help="Use time deltas (i.e. the time between subsequent transactions) in GNN training")
    parser.add_argument("--ego", action='store_true', help="Use ego IDs in GNN training")

    #Model parameters
    parser.add_argument("--batch_size", default=8192, type=int, help="Select the batch size for GNN training")
    parser.add_argument("--n_epochs", default=100, type=int, help="Select the number of epochs for GNN training")
    parser.add_argument('--num_neighs', nargs='+', type=int, default=[100,100], help='Pass the number of neighors to be sampled in each hop (descending).')

    #Misc
    parser.add_argument("--seed", default=1, type=int, help="Select the random seed for reproducability")
    parser.add_argument("--tqdm", action='store_true', help="Use tqdm logging (when running interactively in terminal)")
    parser.add_argument(
        "--data",
        default=None,
        type=str,
        help="Dataset folder under aml-data/ (e.g. Small-HI, PaySim). "
        "Registered adapters: PaySim; others use AMLWorld loading conventions.",
        required=True,
    )
    parser.add_argument(
        "--load_pattern_metadata",
        action="store_true",
        help="Load laundering_attempt_metadata.csv for edge-level pattern metadata (auxiliary only). "
        "Also auto-loads when aml-data/{data}/laundering_attempt_metadata.csv exists.",
    )
    parser.add_argument(
        "--pattern_metadata",
        type=str,
        default=None,
        help="Override path to laundering_attempt_metadata.csv (implies load when set).",
    )
    parser.add_argument("--model", default=None, type=str, help="Select the model architecture. Needs to be one of [gin, gat, rgcn, pna]", required=True)
    parser.add_argument("--testing", action='store_true', help="Disable wandb logging while running the script in 'testing' mode.")
    parser.add_argument("--save_model", action='store_true', help="Save training checkpoints to saved-models/.")
    parser.add_argument(
        "--checkpoint_policy",
        type=str,
        default="last",
        choices=("last", "best"),
        help="Checkpoint selection for contrastive pretrain when --save_model: "
        "'last' overwrites each epoch (legacy); 'best' keeps the lowest SSL val score "
        "(morph/expert_val + morph/contrast_val when available, else loss/train) in "
        "checkpoint_{unique_name}.tar and writes the final epoch to checkpoint_{unique_name}_last.tar.",
    )
    # parser.add_argument("--unique_name", action='store_true', help="Unique name under which the model will be stored.")
    parser.add_argument("--unique_name", type=str, default=None,help="Unique name under which the model will be stored.")
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Initialize weights (and optimizer state) from checkpoint_{unique_name}.tar before training; works with either objective.",
    )
    parser.add_argument("--inference", action='store_true', help="Load a trained model and only do AML inference with it. args.unique name needs to point to the trained model.")
    parser.add_argument(
        "--loader_num_workers",
        type=int,
        default=10,
        help="CPU workers for LinkNeighborLoader prefetch/sampling (0 = single process). Often 8–12 is enough even on many-core nodes; use 0 to debug or on memory-tight login nodes.",
    )

    # Contrastive / memory (homogeneous training path)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA automatic mixed precision for contrastive training (homogeneous path).",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Use gradient checkpointing in GIN message-passing layers (homogeneous GIN only).",
    )
    parser.add_argument(
        "--contrastive_num_neg_samples",
        type=int,
        default=8192,
        help="Edge InfoNCE: negatives per anchor sampled uniformly (0 = all negatives, chunked).",
    )
    parser.add_argument(
        "--contrastive_asymmetric",
        action="store_true",
        help="Compute view-2 embeddings with no grad and use only the z1→z2 loss term (saves VRAM vs symmetric two-branch training).",
    )
    parser.add_argument(
        "--contrastive_accum_steps",
        type=int,
        default=1,
        help="Homogeneous contrastive: accumulate this many loader batches before optimizer.step (loss is scaled 1/N per batch). Use with a smaller --batch_size to cut peak VRAM while keeping similar step frequency.",
    )
    parser.add_argument(
        "--contrastive_memory_bank_size",
        type=int,
        default=0,
        help="Optional detached queue of prior seed-edge embeddings used as extra negatives (0 disables the queue).",
    )
    parser.add_argument(
        "--false_neg_filter_mode",
        type=str,
        default="none",
        choices=["none", "same_sender", "same_receiver", "same_endpoint", "same_pair"],
        help=(
            "Optional exclusion-only false-negative filtering for contrastive negatives. "
            "Filters candidates sharing sender, receiver, either endpoint, or ordered sender->receiver pair "
            "with the anchor transaction. Default keeps existing behavior."
        ),
    )
    parser.add_argument(
        "--false_neg_filter_min_negatives",
        type=int,
        default=1,
        help=(
            "Minimum filtered negatives required per anchor row before falling back to the unfiltered "
            "candidate set for that row (0 disables fallback)."
        ),
    )
    parser.add_argument(
        "--knn_cache_path",
        type=str,
        default=None,
        help="Optional sparse transaction KNN .npz cache for contrastive negative exclusion.",
    )
    parser.add_argument(
        "--enable_knn_negative_filter",
        action="store_true",
        help="Exclude candidates listed in --knn_cache_path from the contrastive negative pool.",
    )
    parser.add_argument(
        "--knn_filter_k",
        type=int,
        default=0,
        help="Use only the first K KNN neighbors from --knn_cache_path (0 = all cached neighbors).",
    )
    parser.add_argument(
        "--enable_knn_soft_positives",
        action="store_true",
        help="Add low-weight KNN feature neighbors as soft positives in edge InfoNCE.",
    )
    parser.add_argument(
        "--knn_pos_source_k",
        type=int,
        default=15,
        help="Number of cached KNN neighbors to consider as soft-positive candidates per anchor.",
    )
    parser.add_argument(
        "--knn_pos_m",
        type=int,
        default=1,
        help="Number of KNN soft positives sampled/used per anchor per step.",
    )
    parser.add_argument(
        "--knn_pos_weight",
        type=float,
        default=0.025,
        help="Total soft-positive mass from KNN neighbors per anchor (split across pos_m positives).",
    )
    parser.add_argument(
        "--knn_pos_weight_mode",
        type=str,
        default="uniform",
        choices=["uniform", "similarity"],
        help="How to split --knn_pos_weight across selected KNN positives.",
    )
    parser.add_argument(
        "--knn_pos_min_sim",
        type=float,
        default=None,
        help="Optional minimum cached cosine similarity required for a KNN soft positive.",
    )
    parser.add_argument(
        "--knn_pos_seed",
        type=int,
        default=0,
        help="Base seed for deterministic KNN soft-positive sampling.",
    )
    parser.add_argument(
        "--knn_pos_loader_batch_size",
        type=int,
        default=4096,
        help="Chunk size for auxiliary LinkNeighborLoader forwards that materialize KNN positives.",
    )
    parser.add_argument(
        "--multi_positive_mode",
        type=str,
        default="none",
        choices=["none", "same_sender", "same_receiver", "same_endpoint", "same_pair"],
        help=(
            "Optional endpoint-based weak positives for edge InfoNCE. Identity positives keep weight 1.0; "
            "weak positives use --multi_positive_weight. Default keeps existing single-positive behavior."
        ),
    )
    parser.add_argument(
        "--multi_positive_weight",
        type=float,
        default=0.1,
        help="Weight for weak positives when --multi_positive_mode is enabled (identity positives remain 1.0).",
    )
    parser.add_argument(
        "--contrastive_temperature",
        type=float,
        default=0.5,
        help="InfoNCE temperature for contrastive loss logits (default 0.5, matching prior behavior).",
    )
    parser.add_argument(
        "--contrast_projection_head",
        action="store_true",
        help="GraphCL-style MLP on seed embeddings before edge InfoNCE only; morphology expert and "
        "embedding extraction still use the encoder readout (128-d).",
    )
    parser.add_argument(
        "--contrast_projection_hidden",
        type=int,
        default=128,
        help="Hidden width for --contrast_projection_head (default 128).",
    )
    parser.add_argument(
        "--contrast_projection_dim",
        type=int,
        default=128,
        help="Output width for --contrast_projection_head (default 128).",
    )
    parser.add_argument(
        "--edge_drop_policy",
        type=str,
        default="random",
        choices=["random", "degree_aware", "degree_flow_aware"],
        help="Edge-drop augmentation policy for contrastive views (default random = legacy uniform drop).",
    )
    parser.add_argument(
        "--edge_drop_target_rate",
        type=float,
        default=0.1,
        help="Target mean edge-drop rate for contrastive views (default 0.1, matching legacy behavior).",
    )
    parser.add_argument(
        "--edge_drop_min_prob",
        type=float,
        default=0.01,
        help="Minimum per-edge drop probability when using degree-aware policies.",
    )
    parser.add_argument(
        "--edge_drop_max_prob",
        type=float,
        default=0.95,
        help="Maximum per-edge drop probability when using degree-aware policies.",
    )
    parser.add_argument(
        "--edge_drop_importance_alpha",
        type=float,
        default=2.0,
        help="Strength of nonuniform edge dropping (higher = stronger preference to keep important edges).",
    )
    parser.add_argument(
        "--edge_drop_score_cache_path",
        type=str,
        default=None,
        help="Optional .npz cache of precomputed train-split edge-drop probabilities.",
    )

    # Morphology expert head (contrastive pretrain, Phase M1)
    parser.add_argument(
        "--morph_expert",
        action="store_true",
        help="Add morphology expert head loss during contrastive pretraining (Tier 1 local targets).",
    )
    parser.add_argument(
        "--morph_targets",
        type=str,
        default="local",
        choices=["local", "local+global", "local+tier2", "local+global+tier2"],
        help="Morphology target set: local (Tier 1); local+global (M1b); "
        "local+tier2 (BC-only global lift ablation); local+global+tier2 (M1b + BC).",
    )
    parser.add_argument(
        "--morph_tier0_cache",
        type=str,
        default=None,
        help="Directory with {train,val,test}_node_morphology.csv from precompute_morphology_tier0.py. "
        "If omitted with local+global(+tier2), tables are computed from split graphs at startup.",
    )
    parser.add_argument(
        "--morph_flow_balance",
        action="store_true",
        help="Append Tier 0 flow-balance morphology targets (10 dims) to the expert head. "
        "Off by default; does not change existing runs unless explicitly enabled.",
    )
    parser.add_argument(
        "--morph_tier0_flow_cache",
        type=str,
        default=None,
        help="Directory with {train,val,test}_node_flow_balance.csv from precompute_morphology_tier0_flow.py. "
        "Falls back to --morph_tier0_cache, then on-the-fly split-graph computation.",
    )
    parser.add_argument(
        "--morph_tier2_cache",
        type=str,
        default=None,
        help="Directory with {train,val,test}_node_tier2.csv from precompute_morphology_tier2.py. "
        "Required for practical local+tier2 / local+global+tier2 runs; otherwise BC is computed at startup (slow).",
    )
    parser.add_argument(
        "--morph_tier2_lift",
        type=str,
        default="full",
        choices=["full", "max"],
        help="Tier 2 BC expert lift: full (4 endpoint cols) or max (bc_max_global only).",
    )
    parser.add_argument(
        "--morph_expert_loss",
        type=str,
        default="mse",
        choices=["mse", "mae"],
        help="Morphology expert regression loss: mse (default) or mae (Papagei-style L1).",
    )
    parser.add_argument(
        "--morph_expert_weight",
        type=float,
        default=1.0,
        help="Weight for morphology expert loss vs InfoNCE.",
    )
    parser.add_argument(
        "--morph_expert_hidden",
        type=int,
        default=64,
        help="Hidden size of morphology expert MLP.",
    )
    parser.add_argument(
        "--morph_expert_layout",
        type=str,
        default="shared",
        choices=["shared", "grouped"],
        help="Expert head layout: shared (single MLP) or grouped (one MLP per block; M5a).",
    )
    parser.add_argument(
        "--morph_expert_group_weight_tier2",
        type=float,
        default=1.0,
        help="Tier 2 block MSE weight when --morph_expert_layout grouped (0 disables BC gradients).",
    )
    parser.add_argument(
        "--morph_local_subset",
        type=str,
        default="all",
        choices=["all", "degree", "clustering", "triangles"],
        help="Tier-1 columns in the morphology expert: all (14), degree (8), "
        "clustering (11, no triangles), triangles (11, no clustering).",
    )
    parser.add_argument(
        "--morph_target_groups",
        type=str,
        default="all",
        help="Semantic morphology target groups to keep for the shared expert head: "
        "all (default) or comma-separated groups such as degree_fan,motif_participation "
        "(legacy alias local_motif expands to motif_participation,local_density,local_context_size). "
        "Uses the diagnostic target registry and leaves default training behavior unchanged.",
    )
    parser.add_argument(
        "--no_morph_edge_native",
        action="store_true",
        help="Do not append forward edge_attr values to morphology targets.",
    )

    # Morphology-aware contrast (Phase M2): soft positives merged into edge InfoNCE
    parser.add_argument(
        "--morph_contrast",
        action="store_true",
        help="Merge morphology-bin soft positives into edge InfoNCE (cross-view, same bin).",
    )
    parser.add_argument(
        "--morph_contrast_features",
        type=str,
        default="local_ego,local_degree",
        help="Comma-separated feature groups for binning: local_ego, local_degree, local_clustering, global_degree, edge_native.",
    )
    parser.add_argument(
        "--morph_contrast_scope",
        type=str,
        default="local",
        choices=["local", "local+global"],
        help="Whether global_degree lift is available for contrast binning (train/val split-safe).",
    )
    parser.add_argument(
        "--morph_contrast_bins",
        type=int,
        default=5,
        help="Quantile buckets per morphology contrast feature dimension (train-split edges).",
    )
    parser.add_argument(
        "--morph_contrast_calib_batches",
        type=int,
        default=32,
        help="Train loader batches used to estimate morphology contrast bin edges at startup.",
    )
    parser.add_argument(
        "--morph_contrast_max_soft_positives",
        type=int,
        default=256,
        help="Max same-bin soft positives per anchor in InfoNCE numerator (0 = no cap). "
        "Limits work when many seeds share one bin; does not reduce batch_size.",
    )
    parser.add_argument(
        "--morph_val_every",
        type=int,
        default=1,
        help="Run morph/expert_val and morph/contrast_val every N epochs (1 = every epoch). "
        "Always runs on the final epoch. Skips the expensive full val-loader forward passes otherwise.",
    )
    parser.add_argument(
        "--morph_val_max_batches",
        type=int,
        default=0,
        help="Cap val-loader batches for morphology val metrics (0 = full val pass). "
        "Each batch still runs two augmented views + GNN forward(s).",
    )

    return parser

def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    logging.info(f"Random seed set as {seed}")
