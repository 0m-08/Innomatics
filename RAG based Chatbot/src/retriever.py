"""
Retrieval Module for the RAG Customer Support Assistant.

Handles document retrieval from ChromaDB with relevance scoring
and confidence evaluation for HITL escalation decisions.
"""

from src.config import RETRIEVAL_TOP_K, CONFIDENCE_THRESHOLD, HIGH_CONFIDENCE_THRESHOLD


def create_retriever(vectorstore, k: int = RETRIEVAL_TOP_K):
    """
    Create a retriever from the vector store.

    Args:
        vectorstore: Chroma vector store instance.
        k: Number of top documents to retrieve (default: 3).

    Returns:
        LangChain retriever object configured for similarity search.
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    return retriever


def retrieve_with_scores(vectorstore, query: str, k: int = RETRIEVAL_TOP_K) -> list:
    """
    Retrieve documents with similarity scores.

    Uses ChromaDB's similarity_search_with_score which returns L2 distance.
    Converts distance to a 0-1 similarity score: score = 1 / (1 + distance).

    Args:
        vectorstore: Chroma vector store instance.
        query: User's search query.
        k: Number of top results to return.

    Returns:
        List of tuples: [(Document, score), ...] sorted by relevance.
        Score is 0.0 (irrelevant) to 1.0 (perfect match).
    """
    # similarity_search_with_score returns (doc, distance) where lower distance = better match
    raw_results = vectorstore.similarity_search_with_score(query, k=k)

    # Convert L2 distance to a 0-1 similarity score
    results = []
    for doc, distance in raw_results:
        # score = 1 / (1 + distance) → maps [0, inf) to (0, 1]
        score = 1.0 / (1.0 + distance)
        results.append((doc, score))

    # Log retrieval results
    print(f"\n🔍 Retrieved {len(results)} documents for query: '{query[:50]}...'")
    for i, (doc, score) in enumerate(results):
        print(f"   [{i+1}] Score: {score:.3f} | Source: {doc.metadata.get('source_file', 'N/A')} | "
              f"Chunk: {doc.metadata.get('chunk_index', 'N/A')}")

    return results


def check_retrieval_confidence(results: list) -> dict:
    """
    Evaluate the confidence of retrieved results.

    Determines whether the retrieved context is sufficient to generate
    a reliable answer or if HITL escalation is needed.

    Args:
        results: List of (Document, score) tuples from retrieve_with_scores.

    Returns:
        Dictionary with confidence assessment:
        {
            "confidence_level": "high" | "medium" | "low",
            "average_score": float,
            "top_score": float,
            "is_sufficient": bool,
            "needs_escalation": bool,
            "reason": str
        }
    """
    if not results:
        return {
            "confidence_level": "low",
            "average_score": 0.0,
            "top_score": 0.0,
            "is_sufficient": False,
            "needs_escalation": True,
            "reason": "No relevant documents found in the knowledge base.",
        }

    scores = [score for _, score in results]
    avg_score = sum(scores) / len(scores)
    top_score = max(scores)

    # Determine confidence level
    if top_score >= HIGH_CONFIDENCE_THRESHOLD:
        confidence_level = "high"
        is_sufficient = True
        needs_escalation = False
        reason = "High-confidence match found in the knowledge base."
    elif top_score >= CONFIDENCE_THRESHOLD:
        confidence_level = "medium"
        is_sufficient = True
        needs_escalation = False
        reason = "Moderate match found. Answer may require verification."
    else:
        confidence_level = "low"
        is_sufficient = False
        needs_escalation = True
        reason = (
            f"Low retrieval confidence (top score: {top_score:.3f}). "
            "The query may be outside the knowledge base scope."
        )

    result = {
        "confidence_level": confidence_level,
        "average_score": round(avg_score, 3),
        "top_score": round(top_score, 3),
        "is_sufficient": is_sufficient,
        "needs_escalation": needs_escalation,
        "reason": reason,
    }

    print(f"📊 Confidence: {confidence_level} (top={top_score:.3f}, avg={avg_score:.3f})")
    return result


def format_retrieved_context(results: list) -> str:
    """
    Format retrieved documents into a context string for the LLM prompt.

    Args:
        results: List of (Document, score) tuples.

    Returns:
        Formatted context string with source attribution.
    """
    if not results:
        return "No relevant information found in the knowledge base."

    context_parts = []
    for i, (doc, score) in enumerate(results):
        source = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page", "N/A")
        context_parts.append(
            f"[Source {i+1}: {source}, Page {page}]\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(context_parts)
