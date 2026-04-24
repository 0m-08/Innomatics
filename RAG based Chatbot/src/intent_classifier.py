"""
Intent Classification Module for the RAG Customer Support Assistant.

Uses the LLM to classify customer queries into predefined intent categories
for routing within the LangGraph workflow.
"""

from src.config import INTENT_CATEGORIES, ESCALATION_INTENTS
from src.llm import INTENT_CLASSIFICATION_PROMPT


def classify_intent(query: str, llm) -> dict:
    """
    Classify the intent of a customer query using the LLM.

    Args:
        query: The customer's input query.
        llm: LLM instance for classification.

    Returns:
        Dictionary with classification results:
        {
            "intent": str,           # One of INTENT_CATEGORIES
            "needs_escalation": bool, # Whether this intent triggers escalation
            "confidence": str        # "high" or "inferred"
        }
    """
    try:
        # Use LLM to classify intent
        prompt = INTENT_CLASSIFICATION_PROMPT.format(query=query)
        response = llm.invoke(prompt)
        raw_intent = response.content.strip().upper().replace(" ", "_")

        # Validate the classification
        if raw_intent in INTENT_CATEGORIES:
            intent = raw_intent
            confidence = "high"
        else:
            # Fallback: try to match partial intent
            intent = _fuzzy_match_intent(raw_intent)
            confidence = "inferred"

        needs_escalation = intent in ESCALATION_INTENTS

        result = {
            "intent": intent,
            "needs_escalation": needs_escalation,
            "confidence": confidence,
        }

        print(f"🏷️  Intent: {intent} (confidence: {confidence}, escalate: {needs_escalation})")
        return result

    except Exception as e:
        print(f"⚠️ Intent classification error: {e}")
        # Default to general query on error
        return {
            "intent": "GENERAL_QUERY",
            "needs_escalation": False,
            "confidence": "fallback",
        }


def _fuzzy_match_intent(raw_intent: str) -> str:
    """
    Attempt to match an unrecognized intent to the closest valid category.

    Args:
        raw_intent: Raw LLM output that didn't match exactly.

    Returns:
        Best matching intent category, or GENERAL_QUERY as default.
    """
    raw_lower = raw_intent.lower()

    # Keyword-based fallback matching
    keyword_map = {
        "GENERAL_QUERY": ["general", "faq", "question", "info", "information", "help"],
        "TECHNICAL_ISSUE": ["technical", "tech", "error", "bug", "issue", "problem", "troubleshoot", "fix"],
        "BILLING": ["billing", "bill", "payment", "invoice", "refund", "charge", "price", "plan", "subscription"],
        "COMPLAINT": ["complaint", "complain", "angry", "frustrated", "upset", "dissatisfied", "unhappy"],
        "ESCALATE": ["escalate", "human", "agent", "manager", "supervisor", "person", "real", "speak", "talk"],
        "OUT_OF_SCOPE": ["out_of_scope", "unrelated", "irrelevant", "off_topic"],
    }

    for intent, keywords in keyword_map.items():
        if any(kw in raw_lower for kw in keywords):
            return intent

    return "GENERAL_QUERY"


def get_escalation_reason(intent: str, query: str) -> str:
    """
    Generate a human-readable escalation reason based on intent.

    Args:
        intent: Classified intent category.
        query: Original customer query.

    Returns:
        Descriptive reason for escalation.
    """
    reasons = {
        "COMPLAINT": (
            "Customer has expressed dissatisfaction or filed a complaint. "
            "Human intervention recommended for empathetic resolution."
        ),
        "ESCALATE": (
            "Customer has explicitly requested to speak with a human agent. "
            "Transferring to human support."
        ),
        "OUT_OF_SCOPE": (
            "The customer's query falls outside the scope of the available knowledge base. "
            "A human agent may be able to provide specialized assistance."
        ),
    }

    return reasons.get(intent, f"Escalation triggered for intent: {intent}")
