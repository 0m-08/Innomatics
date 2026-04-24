"""Tests for the Retriever module."""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retriever import check_retrieval_confidence, format_retrieved_context


class TestRetriever:
    """Test suite for retrieval and confidence scoring."""

    def test_confidence_high(self):
        """Test high confidence detection."""
        mock_doc = type('Doc', (), {'page_content': 'test', 'metadata': {'source_file': 'test.pdf', 'page': 1}})()
        results = [(mock_doc, 0.85), (mock_doc, 0.75), (mock_doc, 0.60)]
        confidence = check_retrieval_confidence(results)
        assert confidence["confidence_level"] == "high"
        assert confidence["is_sufficient"] is True
        assert confidence["needs_escalation"] is False

    def test_confidence_medium(self):
        """Test medium confidence detection."""
        mock_doc = type('Doc', (), {'page_content': 'test', 'metadata': {'source_file': 'test.pdf', 'page': 1}})()
        results = [(mock_doc, 0.45), (mock_doc, 0.35), (mock_doc, 0.32)]
        confidence = check_retrieval_confidence(results)
        assert confidence["confidence_level"] == "medium"
        assert confidence["is_sufficient"] is True

    def test_confidence_low(self):
        """Test low confidence triggers escalation."""
        mock_doc = type('Doc', (), {'page_content': 'test', 'metadata': {'source_file': 'test.pdf', 'page': 1}})()
        results = [(mock_doc, 0.2), (mock_doc, 0.15), (mock_doc, 0.1)]
        confidence = check_retrieval_confidence(results)
        assert confidence["confidence_level"] == "low"
        assert confidence["needs_escalation"] is True

    def test_confidence_empty_results(self):
        """Test empty results trigger escalation."""
        confidence = check_retrieval_confidence([])
        assert confidence["confidence_level"] == "low"
        assert confidence["needs_escalation"] is True

    def test_format_context_with_docs(self):
        """Test context formatting."""
        mock_doc = type('Doc', (), {
            'page_content': 'This is test content.',
            'metadata': {'source_file': 'guide.pdf', 'page': 3}
        })()
        results = [(mock_doc, 0.9)]
        context = format_retrieved_context(results)
        assert "test content" in context
        assert "guide.pdf" in context

    def test_format_context_empty(self):
        """Test empty results formatting."""
        context = format_retrieved_context([])
        assert "No relevant" in context


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
