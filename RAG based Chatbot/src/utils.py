"""
Utility functions for the RAG Customer Support Assistant.

Shared helper functions used across multiple modules.
"""

import datetime


def get_timestamp() -> str:
    """Get current timestamp in a readable format."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def truncate_text(text: str, max_length: int = 200) -> str:
    """
    Truncate text to a maximum length with ellipsis.

    Args:
        text: Input text to truncate.
        max_length: Maximum character length.

    Returns:
        Truncated text with '...' if it exceeded the limit.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def format_chat_history(messages: list) -> list:
    """
    Format chat history for the LLM prompt.

    Converts a list of message dicts to LangChain message tuples.

    Args:
        messages: List of {"role": str, "content": str} dicts.

    Returns:
        List of (role, content) tuples for MessagesPlaceholder.
    """
    formatted = []
    for msg in messages:
        role = msg.get("role", "human")
        content = msg.get("content", "")
        if role == "user":
            formatted.append(("human", content))
        elif role == "assistant":
            formatted.append(("ai", content))
    return formatted


def create_escalation_summary(state: dict) -> str:
    """
    Create a summary of the escalation context for human agents.

    Args:
        state: Current graph state dictionary.

    Returns:
        Formatted escalation summary string.
    """
    summary = f"""
╔══════════════════════════════════════════════════╗
║           ESCALATION TICKET                      ║
╠══════════════════════════════════════════════════╣
║ Timestamp: {get_timestamp()}
║ Customer Query: {truncate_text(state.get('query', 'N/A'), 100)}
║ Detected Intent: {state.get('intent', 'N/A')}
║ Retrieval Confidence: {state.get('retrieval_confidence', 'N/A')}
║ Escalation Reason: {state.get('escalation_reason', 'N/A')}
╚══════════════════════════════════════════════════╝
    """
    return summary.strip()
