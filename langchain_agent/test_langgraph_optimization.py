#!/usr/bin/env python3
"""
Tests for Phase 2: LangGraph Optimization features.

Tests:
1. SelectiveJsonPlusSerializer - Verifies large fields are excluded
2. Checkpoint maintenance utilities - Verifies compaction logic
3. Config variables - Verifies new config options are accessible
"""

import unittest
from unittest.mock import MagicMock, patch
from typing import Dict, Any


class TestSelectiveSerializer(unittest.TestCase):
    """Test checkpoint selective serialization."""

    def test_excludes_retrieved_documents(self):
        """Verify retrieved_documents field is excluded from serialization."""
        from checkpoint_optimizer import SelectiveJsonPlusSerializer
        from langchain_core.documents import Document

        serializer = SelectiveJsonPlusSerializer()

        # Create state with large retrieved_documents
        state = {
            "messages": [{"role": "user", "content": "test query"}],
            "retrieved_documents": [
                Document(page_content="large content " * 100, metadata={"source": "doc1.txt"})
            ],
            "lambda_mult": 0.5,
            "iteration_count": 1,
        }

        # Serialize
        type_str, serialized = serializer.dumps_typed(state)

        # Verify excluded field not in serialized data
        self.assertNotIn(b"large content", serialized)
        # Verify preserved fields are present
        self.assertIn(b"messages", serialized)
        self.assertIn(b"lambda_mult", serialized)
        self.assertIn(b"iteration_count", serialized)

    def test_excludes_document_grades(self):
        """Verify document_grades field is excluded from serialization."""
        from checkpoint_optimizer import SelectiveJsonPlusSerializer

        serializer = SelectiveJsonPlusSerializer()

        # Create state with document_grades
        state = {
            "messages": [],
            "document_grades": [
                {"source": "doc1.txt", "score": 0.9, "reasoning": "Very relevant content"},
                {"source": "doc2.txt", "score": 0.3, "reasoning": "Less relevant"},
            ],
            "document_grade_summary": {"grade": "pass", "score": 0.6},
        }

        # Serialize
        type_str, serialized = serializer.dumps_typed(state)

        # Verify excluded field not in serialized data
        self.assertNotIn(b"Very relevant content", serialized)
        # Verify summary (small, preserved) is present
        self.assertIn(b"document_grade_summary", serialized)

    def test_preserves_essential_fields(self):
        """Verify essential fields are preserved in serialization."""
        from checkpoint_optimizer import SelectiveJsonPlusSerializer

        serializer = SelectiveJsonPlusSerializer()

        # Create state with all essential fields
        state = {
            "messages": [{"role": "user", "content": "hello"}],
            "lambda_mult": 0.25,
            "query_analysis": "keyword query",
            "optimized_query": "optimized version",
            "iteration_count": 2,
            "response_retry_count": 1,
            "original_query": "original question",
            "transformed_query": "transformed question",
            "document_grade_summary": {"grade": "pass"},
            "response_grade": {"grade": "pass", "score": 0.9},
            "force_retrieval_retry": False,
        }

        # Serialize and deserialize
        type_str, serialized = serializer.dumps_typed(state)
        restored = serializer.loads_typed((type_str, serialized))

        # Verify all essential fields are preserved
        self.assertIn("messages", restored)
        self.assertIn("lambda_mult", restored)
        self.assertIn("iteration_count", restored)
        self.assertIn("original_query", restored)
        self.assertIn("response_grade", restored)

    def test_non_dict_passthrough(self):
        """Verify non-dict values are passed through unchanged."""
        from checkpoint_optimizer import SelectiveJsonPlusSerializer

        serializer = SelectiveJsonPlusSerializer()

        # Test with a simple list
        value = ["item1", "item2", "item3"]
        type_str, serialized = serializer.dumps_typed(value)
        restored = serializer.loads_typed((type_str, serialized))

        self.assertEqual(restored, value)


class TestConfigVariables(unittest.TestCase):
    """Test new configuration variables are accessible."""

    def test_langsmith_config_exists(self):
        """Verify LangSmith config variables are defined."""
        from config import (
            LANGSMITH_API_KEY,
            LANGSMITH_PROJECT,
            LANGSMITH_TRACING_ENABLED,
        )

        # LANGSMITH_API_KEY will be None if not set
        self.assertIsNotNone(LANGSMITH_PROJECT)
        self.assertEqual(LANGSMITH_PROJECT, "rusty-compass")  # Default value
        # Tracing enabled only if API key is set
        self.assertEqual(LANGSMITH_TRACING_ENABLED, LANGSMITH_API_KEY is not None)

    def test_astream_events_config_exists(self):
        """Verify astream_events config variable is defined."""
        from config import ENABLE_ASTREAM_EVENTS

        # Default is false
        self.assertIsInstance(ENABLE_ASTREAM_EVENTS, bool)

    def test_checkpoint_config_exists(self):
        """Verify checkpoint optimization config variables are defined."""
        from config import (
            CHECKPOINT_SELECTIVE_SERIALIZATION,
            CHECKPOINT_KEEP_VERSIONS,
            CHECKPOINT_COMPACTION_DAYS,
        )

        self.assertIsInstance(CHECKPOINT_SELECTIVE_SERIALIZATION, bool)
        self.assertIsInstance(CHECKPOINT_KEEP_VERSIONS, int)
        self.assertIsInstance(CHECKPOINT_COMPACTION_DAYS, int)
        self.assertGreater(CHECKPOINT_KEEP_VERSIONS, 0)
        self.assertGreater(CHECKPOINT_COMPACTION_DAYS, 0)


class TestCheckpointMaintenance(unittest.TestCase):
    """Test checkpoint maintenance utility functions."""

    def test_get_checkpoint_stats_structure(self):
        """Verify checkpoint stats returns expected structure."""
        # This test requires database connection, so we mock it
        with patch('checkpoint_maintenance.psycopg.connect') as mock_connect:
            mock_cursor = MagicMock()
            mock_cursor.fetchone.side_effect = [
                (5,),   # thread_count
                (25,),  # checkpoint_count
                (100,), # blob_count
                ("1 MB",),  # estimated_size
            ]
            mock_connect.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cursor

            from checkpoint_maintenance import get_checkpoint_stats

            stats = get_checkpoint_stats()

            self.assertIn("thread_count", stats)
            self.assertIn("checkpoint_count", stats)
            self.assertIn("blob_count", stats)
            self.assertIn("estimated_size", stats)
            self.assertEqual(stats["thread_count"], 5)
            self.assertEqual(stats["checkpoint_count"], 25)


class TestSelectiveSerializerIntegration(unittest.TestCase):
    """Integration test for selective serializer with PostgresSaver."""

    @unittest.skip("Requires database connection - run manually")
    def test_checkpoint_with_selective_serialization(self):
        """Test that checkpointing works with selective serialization."""
        from langgraph.checkpoint.postgres import PostgresSaver
        from checkpoint_optimizer import SelectiveJsonPlusSerializer
        from config import DATABASE_URL

        # This is a manual integration test
        # Run with: python -m pytest test_langgraph_optimization.py::TestSelectiveSerializerIntegration -v
        pass


def run_tests():
    """Run all tests and print summary."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSelectiveSerializer))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigVariables))
    suite.addTests(loader.loadTestsFromTestCase(TestCheckpointMaintenance))

    # Run with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.wasSuccessful():
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1


if __name__ == "__main__":
    exit(run_tests())
