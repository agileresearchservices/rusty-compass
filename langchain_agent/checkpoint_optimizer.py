"""
Custom checkpoint serialization for selective state persistence.

Excludes large fields (retrieved_documents, document_grades) from checkpoint storage
while preserving essential state for conversation recovery. This reduces checkpoint
size by ~10x and improves write performance.

Usage:
    from checkpoint_optimizer import SelectiveJsonPlusSerializer
    checkpointer = PostgresSaver(pool, serde=SelectiveJsonPlusSerializer())
"""

from typing import Any, Tuple
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


class SelectiveJsonPlusSerializer(JsonPlusSerializer):
    """
    Custom serializer that excludes large state fields from checkpointing.

    Fields excluded:
    - retrieved_documents: Large Document objects (regenerated on retrieval)
    - document_grades: Can be regenerated from documents

    Fields preserved:
    - messages: Essential for conversation continuity
    - lambda_mult, query_analysis, optimized_query: Query context
    - iteration_count, response_retry_count: Reflection state
    - original_query, transformed_query: Query history
    - document_grade_summary, response_grade: Summary results (small)
    - force_retrieval_retry: Control flag

    Note: The excluded fields are large but transient - they're only needed
    during a single query execution and are regenerated on each retrieval.
    """

    EXCLUDED_FIELDS = {
        "retrieved_documents",  # Large Document objects - regenerated on retrieval
        "document_grades",      # Large list of grade dicts - regenerated from docs
    }

    def dumps_typed(self, value: Any) -> Tuple[str, bytes]:
        """
        Serialize state, excluding large fields.

        Args:
            value: The state value to serialize

        Returns:
            Tuple of (type_string, serialized_bytes)
        """
        if isinstance(value, dict):
            # Filter out excluded fields before serialization
            filtered = {
                k: v for k, v in value.items()
                if k not in self.EXCLUDED_FIELDS
            }
            return super().dumps_typed(filtered)
        return super().dumps_typed(value)
