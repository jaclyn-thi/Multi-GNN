"""Standalone GCPAL-style transaction-node baseline (not an exact paper reproduction)."""

from gcpal_txn_node.spec import (
    DEFAULT_KNN_CACHE,
    DocumentedAssumptions,
    LAMBDA_MIX,
    NOT_EXACT_REPRODUCTION,
    TEMPERATURE,
)
from gcpal_txn_node.extraction import (
    CANONICAL_EXTRACTION_MODE,
    LEGACY_CHUNKED_EXTRACTION_MODE,
    canonical_extraction_config,
    extract_split_embeddings,
)

__all__ = [
    "DEFAULT_KNN_CACHE",
    "DocumentedAssumptions",
    "LAMBDA_MIX",
    "NOT_EXACT_REPRODUCTION",
    "TEMPERATURE",
    "CANONICAL_EXTRACTION_MODE",
    "LEGACY_CHUNKED_EXTRACTION_MODE",
    "canonical_extraction_config",
    "extract_split_embeddings",
]
