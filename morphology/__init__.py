"""
Morphology metrics for transaction-graph SSL (Tier 0 global + Tier 1 local).

See ``morphology/IDS.md`` and ``notes/morphology-metrics-plan.md``.
"""

from morphology.graph_access import (
    get_edge_ids_for_positions,
    get_forward_edge_attr,
    get_forward_edge_index,
    get_forward_timestamps,
    get_num_nodes,
    seed_endpoints_from_edge_ids,
)
from morphology.tier0_global import (
    DEFAULT_LIFT_FEATURE_NAMES,
    GLOBAL_LIFT_FEATURE_NAMES,
    MorphTier0Context,
    TIER0_NODE_COLUMNS,
    compute_tier0_node_stats,
    get_default_lift_feature_names,
    lift_global_to_seed_edges_torch,
    lift_node_to_seed_edges,
    load_node_table,
    save_node_table,
    setup_morph_tier0_contexts,
    tier0_context_from_cache,
    tier0_context_from_graph,
)
from morphology.expert import (
    MorphExpertConfig,
    MorphologyExpertHead,
    build_morph_targets,
    create_morph_expert_bundle,
    morph_expert_mse_loss,
    morphology_expert_step,
    setup_morphology_expert,
    target_dim_for_config,
)
from morphology.tier1_local import (
    LOCAL_FEATURE_NAMES,
    compute_local_morphology,
    compute_local_morphology_torch,
    resolve_seed_positions_in_subgraph,
)

__all__ = [
    "DEFAULT_LIFT_FEATURE_NAMES",
    "GLOBAL_LIFT_FEATURE_NAMES",
    "LOCAL_FEATURE_NAMES",
    "MorphExpertConfig",
    "MorphTier0Context",
    "MorphologyExpertHead",
    "TIER0_NODE_COLUMNS",
    "build_morph_targets",
    "compute_local_morphology",
    "compute_local_morphology_torch",
    "compute_tier0_node_stats",
    "create_morph_expert_bundle",
    "get_default_lift_feature_names",
    "get_edge_ids_for_positions",
    "get_forward_edge_attr",
    "get_forward_edge_index",
    "get_forward_timestamps",
    "get_num_nodes",
    "lift_global_to_seed_edges_torch",
    "lift_node_to_seed_edges",
    "load_node_table",
    "morph_expert_mse_loss",
    "morphology_expert_step",
    "resolve_seed_positions_in_subgraph",
    "save_node_table",
    "seed_endpoints_from_edge_ids",
    "setup_morph_tier0_contexts",
    "setup_morphology_expert",
    "target_dim_for_config",
    "tier0_context_from_cache",
    "tier0_context_from_graph",
]
