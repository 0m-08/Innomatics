"""Tests for the Graph Workflow module."""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph_workflow import route_after_intent, route_after_confidence, GraphState


class TestGraphWorkflow:
    """Test suite for graph routing logic."""

    def test_route_after_intent_escalation(self):
        """Test routing to escalation when needs_escalation is True."""
        state = {"needs_escalation": True, "intent": "COMPLAINT"}
        assert route_after_intent(state) == "escalate_to_human"

    def test_route_after_intent_normal(self):
        """Test routing to retrieval for normal queries."""
        state = {"needs_escalation": False, "intent": "GENERAL_QUERY"}
        assert route_after_intent(state) == "retrieve_documents"

    def test_route_after_confidence_low(self):
        """Test routing to escalation on low confidence."""
        state = {"needs_escalation": True, "retrieval_confidence": 0.2}
        assert route_after_confidence(state) == "escalate_to_human"

    def test_route_after_confidence_high(self):
        """Test routing to generation on sufficient confidence."""
        state = {"needs_escalation": False, "retrieval_confidence": 0.85}
        assert route_after_confidence(state) == "generate_response"

    def test_graph_state_schema(self):
        """Test that GraphState has all required fields."""
        required_fields = [
            'query', 'intent', 'retrieved_docs', 'retrieval_confidence',
            'response', 'needs_escalation', 'escalation_reason',
            'human_response', 'chat_history', 'sources'
        ]
        annotations = GraphState.__annotations__
        for field in required_fields:
            assert field in annotations, f"GraphState missing field: {field}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
