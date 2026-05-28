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
    parser.add_argument("--data", default=None, type=str, help="Select the AML dataset. Needs to be either small or medium.", required=True)
    parser.add_argument("--model", default=None, type=str, help="Select the model architecture. Needs to be one of [gin, gat, rgcn, pna]", required=True)
    parser.add_argument("--testing", action='store_true', help="Disable wandb logging while running the script in 'testing' mode.")
    parser.add_argument("--save_model", action='store_true', help="Save the best model.")
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
