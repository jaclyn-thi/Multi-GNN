"""Documented assumptions for the standalone GCPAL-style txn-node baseline.

Not an exact paper reproduction. See notes/gcpal_txn_node_ambiguity_report.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

PACKAGE_NAME = "gcpal_txn_node"
NOT_EXACT_REPRODUCTION = True

# Paper-stated defaults we adopt when specified.
EMBEDDING_DIM = 128
N_GIN_LAYERS = 2
TEMPERATURE = 0.5
LAMBDA_MIX = 0.3
KNN_K = 15
EDGE_DROP_RATE = 0.1
FEATURE_DROP_RATE = 0.1
DEFAULT_BATCH_SIZE = 2048
DEFAULT_ADJACENCY_POLICY = "immediate_next"
DEFAULT_INCLUDE_IDENTITY = True

# KNN cache (train-split, sparse). Degree features are a recorded deviation.
DEFAULT_KNN_CACHE = (
    "morphology_cache/Small-HI/transaction_knn_edge_native_degree_fan_k15.npz"
)


@dataclass(frozen=True)
class DocumentedAssumptions:
    adjacency_policy: str = DEFAULT_ADJACENCY_POLICY
    knn_cache_feature_set: str = "edge_native+degree_fan"
    knn_deviation: str = (
        "Existing global train KNN cache includes degree_fan features beyond raw AML columns."
    )
    knn_scope: str = "sparse_global_train_cache"
    identity_in_positives: bool = DEFAULT_INCLUDE_IDENTITY
    identity_note: str = (
        "Eq.9 writes A+A_knn; ablation implies identity. Primary smoke uses I∪A∪A_knn."
    )
    feature_fit: str = "train_split_only_temporal"
    batching: str = "positive_aware_capped"
    not_exact_reproduction: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


EXPLICIT_PAPER_ITEMS = [
    "transactions_as_nodes",
    "financial_flow_edges",
    "two_edge_feature_drop_views",
    "knn_graph_from_transaction_features",
    "shared_vanilla_gin",
    "projection_mlp",
    "positive_matrix_A_plus_Aknn",
    "symmetric_contrast",
    "lambda_approx_0.3",
    "temperature_approx_0.5",
    "k_approx_15",
    "embedding_dim_128",
    "downstream_mlp_on_H_concat_X",
]

UNRESOLVED_ITEMS = [
    "exact_amlworld_transaction_node_adjacency",
    "global_vs_approximate_knn_implementation",
    "feature_preprocessing_details",
    "identity_inclusion_in_full_positive_matrix",
    "graph_batching_sampling",
    "optimizer_and_training_duration",
    "split_and_transductive_behavior",
]
