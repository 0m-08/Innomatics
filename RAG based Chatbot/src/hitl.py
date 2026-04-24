"""
Human-in-the-Loop (HITL) Module for the RAG Customer Support Assistant.

Manages escalation triggers, ticket creation, and integration of human
responses back into the automated workflow.
"""

from src.config import ESCALATION_INTENTS, CONFIDENCE_THRESHOLD
from src.utils import get_timestamp, create_escalation_summary


def check_escalation_needed(state: dict) -> bool:
    """
    Evaluate whether the current state requires HITL escalation.

    Escalation is triggered when:
    1. Intent is classified as COMPLAINT, ESCALATE, or OUT_OF_SCOPE
    2. Retrieval confidence is below the threshold
    3. The needs_escalation flag is already set

    Args:
        state: Current graph state dictionary.

    Returns:
        True if escalation is needed, False otherwise.
    """
    # Check 1: Intent-based escalation
    intent = state.get("intent", "")
    if intent in ESCALATION_INTENTS:
        return True

    # Check 2: Confidence-based escalation
    confidence = state.get("retrieval_confidence", 1.0)
    if confidence < CONFIDENCE_THRESHOLD:
        return True

    # Check 3: Explicit flag
    if state.get("needs_escalation", False):
        return True

    return False


def create_escalation_ticket(state: dict) -> dict:
    """
    Package the current context into an escalation ticket for a human agent.

    Args:
        state: Current graph state dictionary.

    Returns:
        Escalation ticket dictionary with all relevant context.
    """
    ticket = {
        "ticket_id": f"ESC-{hash(state.get('query', '')) % 10000:04d}",
        "timestamp": get_timestamp(),
        "customer_query": state.get("query", ""),
        "detected_intent": state.get("intent", "UNKNOWN"),
        "retrieval_confidence": state.get("retrieval_confidence", 0.0),
        "escalation_reason": state.get("escalation_reason", "Manual escalation"),
        "retrieved_context": _summarize_retrieved_docs(state.get("retrieved_docs", [])),
        "chat_history_length": len(state.get("chat_history", [])),
        "status": "PENDING_HUMAN_REVIEW",
    }

    print(f"\n🎫 Escalation Ticket Created: {ticket['ticket_id']}")
    print(create_escalation_summary(state))

    return ticket


def integrate_human_response(state: dict, human_input: str) -> dict:
    """
    Merge the human agent's response back into the workflow state.

    After a human agent provides a response to an escalated query,
    this function integrates that response into the state so the
    workflow can continue.

    Args:
        state: Current graph state dictionary.
        human_input: The human agent's response text.

    Returns:
        Updated state dictionary with human response integrated.
    """
    updated_state = {
        "human_response": human_input,
        "response": (
            f"*[This response was provided by a human support agent]*\n\n"
            f"{human_input}"
        ),
        "needs_escalation": False,
        "escalation_reason": "",
    }

    print(f"✅ Human response integrated ({len(human_input)} chars)")
    return updated_state


def get_escalation_message(state: dict) -> str:
    """
    Generate the message shown to the user when escalation occurs.

    Args:
        state: Current graph state dictionary.

    Returns:
        User-facing escalation message.
    """
    intent = state.get("intent", "")
    reason = state.get("escalation_reason", "")

    if intent == "ESCALATE":
        return (
            "I understand you'd like to speak with a human agent. "
            "I'm transferring your query to our support team now. "
            "A human agent will review your request and respond shortly."
        )
    elif intent == "COMPLAINT":
        return (
            "I'm sorry to hear about your experience. I want to make sure "
            "your concern receives the attention it deserves. I'm escalating "
            "this to a human support agent who can better assist you."
        )
    elif intent == "OUT_OF_SCOPE":
        return (
            "I apologize, but this question falls outside the scope of what "
            "I can help with based on my current knowledge base. Let me "
            "connect you with a human agent who may be able to assist you."
        )
    else:
        return (
            f"I'm not fully confident in my ability to answer this accurately. "
            f"Reason: {reason}\n\n"
            "I'm escalating this to a human agent for a more reliable response."
        )


def _summarize_retrieved_docs(docs: list) -> str:
    """Summarize retrieved documents for the escalation ticket."""
    if not docs:
        return "No documents were retrieved."

    summaries = []
    for i, doc in enumerate(docs[:3]):
        if hasattr(doc, 'page_content'):
            content = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
        elif isinstance(doc, tuple):
            content = doc[0].page_content[:150] + "..." if len(doc[0].page_content) > 150 else doc[0].page_content
        else:
            content = str(doc)[:150]
        summaries.append(f"  [{i+1}] {content}")

    return "\n".join(summaries)
