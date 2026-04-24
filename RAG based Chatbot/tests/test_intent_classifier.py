"""Tests for the Intent Classifier module."""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.intent_classifier import _fuzzy_match_intent, get_escalation_reason


class TestIntentClassifier:
    """Test suite for intent classification."""

    def test_fuzzy_match_billing(self):
        """Test fuzzy matching for billing intents."""
        assert _fuzzy_match_intent("BILLING_QUERY") == "BILLING"
        assert _fuzzy_match_intent("payment refund") == "BILLING"

    def test_fuzzy_match_technical(self):
        """Test fuzzy matching for technical intents."""
        assert _fuzzy_match_intent("TECH_SUPPORT") == "TECHNICAL_ISSUE"
        assert _fuzzy_match_intent("error encountered") == "TECHNICAL_ISSUE"

    def test_fuzzy_match_escalation(self):
        """Test fuzzy matching for escalation intents."""
        assert _fuzzy_match_intent("talk to human") == "ESCALATE"
        assert _fuzzy_match_intent("need a manager") == "ESCALATE"

    def test_fuzzy_match_complaint(self):
        """Test fuzzy matching for complaint intents."""
        assert _fuzzy_match_intent("frustrated customer") == "COMPLAINT"
        assert _fuzzy_match_intent("very angry") == "COMPLAINT"

    def test_fuzzy_match_default(self):
        """Test default fallback to GENERAL_QUERY."""
        assert _fuzzy_match_intent("xyzabc123") == "GENERAL_QUERY"

    def test_escalation_reason_complaint(self):
        """Test escalation reason for complaints."""
        reason = get_escalation_reason("COMPLAINT", "I'm very upset")
        assert "dissatisfaction" in reason.lower() or "complaint" in reason.lower()

    def test_escalation_reason_escalate(self):
        """Test escalation reason for explicit escalation."""
        reason = get_escalation_reason("ESCALATE", "Let me talk to a person")
        assert "human" in reason.lower()

    def test_escalation_reason_out_of_scope(self):
        """Test escalation reason for out-of-scope."""
        reason = get_escalation_reason("OUT_OF_SCOPE", "What is quantum physics?")
        assert "scope" in reason.lower() or "outside" in reason.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
