"""
Graph Workflow Module for the RAG Customer Support Assistant.

Implements the LangGraph StateGraph with conditional routing,
intent-based branching, and HITL escalation support.

Workflow:
    START → classify_intent
        → [ESCALATE/COMPLAINT/OUT_OF_SCOPE] → escalate_to_human → END
        → [GENERAL/TECHNICAL/BILLING] → retrieve_documents → evaluate_confidence
            → [Low Confidence] → escalate_to_human → END
            → [Sufficient Confidence] → generate_response → END
"""

import time
from typing import TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

from src.llm import get_llm, get_embeddings, RAG_SYSTEM_PROMPT
from src.intent_classifier import classify_intent, get_escalation_reason
from src.retriever import retrieve_with_scores, check_retrieval_confidence, format_retrieved_context
from src.vector_store import get_vectorstore
from src.hitl import check_escalation_needed, create_escalation_ticket, get_escalation_message
from src.config import RETRIEVAL_TOP_K, ESCALATION_INTENTS

# Retry settings for rate-limited API calls
MAX_RETRIES = 3
BASE_RETRY_DELAY = 15  # seconds


# ─────────────────────────────────────────────
# State Definition
# ─────────────────────────────────────────────

class GraphState(TypedDict):
    """State object that flows through the LangGraph workflow.

    Each node reads from and writes to this shared state.
    """
    query: str                              # User's input query
    intent: str                             # Classified intent category
    intent_confidence: str                  # Intent classification confidence
    retrieved_docs: list                    # Retrieved document chunks
    retrieval_confidence: float             # Similarity score of top result
    confidence_details: dict                # Full confidence assessment
    context: str                            # Formatted context for LLM
    response: str                           # Generated response
    needs_escalation: bool                  # HITL escalation flag
    escalation_reason: str                  # Why escalation was triggered
    escalation_ticket: Optional[dict]       # Escalation ticket data
    human_response: str                     # Human agent's response
    chat_history: list                      # Conversation history
    sources: list                           # Source documents used


# ─────────────────────────────────────────────
# Graph Nodes
# ─────────────────────────────────────────────

def classify_intent_node(state: GraphState) -> dict:
    """
    Node 1: Classify the user's query intent.

    Analyzes the query using the LLM and determines the intent category.
    Sets the escalation flag if the intent requires human intervention.
    Retries automatically on rate-limit (429) errors.
    """
    print("\n═══ Node: CLASSIFY INTENT ═══")
    query = state["query"]

    result = None
    for attempt in range(MAX_RETRIES):
        try:
            llm = get_llm()
            result = classify_intent(query, llm)
            break  # Success
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = BASE_RETRY_DELAY * (attempt + 1)
                print(f"⏳ Rate limited. Retrying in {wait}s... (attempt {attempt+1}/{MAX_RETRIES})")
                time.sleep(wait)
            else:
                raise  # Non-rate-limit error, don't retry

    # If all retries failed, use fallback
    if result is None:
        print("⚠️ All retries exhausted for intent classification, using fallback.")
        result = {"intent": "GENERAL_QUERY", "needs_escalation": False, "confidence": "fallback"}

    # Determine if escalation is needed based on intent
    needs_escalation = result["needs_escalation"]
    escalation_reason = ""

    if needs_escalation:
        escalation_reason = get_escalation_reason(result["intent"], query)

    return {
        "intent": result["intent"],
        "intent_confidence": result["confidence"],
        "needs_escalation": needs_escalation,
        "escalation_reason": escalation_reason,
    }


def retrieve_documents_node(state: GraphState) -> dict:
    """
    Node 2: Retrieve relevant documents from ChromaDB.

    Performs similarity search and returns documents with scores.
    """
    print("\n═══ Node: RETRIEVE DOCUMENTS ═══")
    query = state["query"]

    try:
        embeddings = get_embeddings()
        vectorstore = get_vectorstore(embeddings)

        # Retrieve documents with relevance scores
        results = retrieve_with_scores(vectorstore, query, k=RETRIEVAL_TOP_K)

        # Extract documents and format context
        docs = [doc for doc, score in results]
        context = format_retrieved_context(results)

        # Prepare source information
        sources = []
        for doc, score in results:
            sources.append({
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "source": doc.metadata.get("source_file", "N/A"),
                "page": doc.metadata.get("page", "N/A"),
                "score": round(score, 3),
            })

        return {
            "retrieved_docs": results,
            "context": context,
            "sources": sources,
        }

    except FileNotFoundError as e:
        print(f"⚠️ Vector store not found: {e}")
        return {
            "retrieved_docs": [],
            "context": "No knowledge base has been loaded. Please upload a PDF first.",
            "sources": [],
            "needs_escalation": True,
            "escalation_reason": "Knowledge base not initialized. No documents available for retrieval.",
        }


def evaluate_confidence_node(state: GraphState) -> dict:
    """
    Node 3: Evaluate the confidence of retrieved results.

    Scores the retrieval quality and determines if the context
    is sufficient for answer generation or if escalation is needed.
    """
    print("\n═══ Node: EVALUATE CONFIDENCE ═══")
    results = state.get("retrieved_docs", [])

    # Check retrieval confidence
    confidence = check_retrieval_confidence(results)

    update = {
        "retrieval_confidence": confidence["top_score"],
        "confidence_details": confidence,
    }

    # If confidence is too low, flag for escalation
    if confidence["needs_escalation"]:
        update["needs_escalation"] = True
        update["escalation_reason"] = confidence["reason"]

    return update


def generate_response_node(state: GraphState) -> dict:
    """
    Node 4: Generate the final response using the LLM and retrieved context.

    Combines the user's query with retrieved context to generate
    a contextually relevant answer.
    """
    print("\n═══ Node: GENERATE RESPONSE ═══")
    query = state["query"]
    context = state.get("context", "No context available.")
    chat_history = state.get("chat_history", [])

    llm = get_llm()

    # Build the prompt with context
    system_prompt = RAG_SYSTEM_PROMPT.format(context=context)

    messages = [{"role": "system", "content": system_prompt}]

    # Add chat history (last 6 messages for context window management)
    for msg in chat_history[-6:]:
        messages.append(msg)

    messages.append({"role": "user", "content": query})

    # Convert to LangChain messages
    from langchain_core.messages import SystemMessage
    lc_messages = []
    for msg in messages:
        if msg["role"] == "system":
            lc_messages.append(SystemMessage(content=msg["content"]))
        elif msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = llm.invoke(lc_messages)
            answer = response.content

            print(f"✅ Response generated ({len(answer)} chars)")
            return {"response": answer}

        except Exception as e:
            last_error = e
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = BASE_RETRY_DELAY * (attempt + 1)
                print(f"⏳ Rate limited. Retrying in {wait}s... (attempt {attempt+1}/{MAX_RETRIES})")
                time.sleep(wait)
                llm = get_llm()  # Refresh the LLM instance
            else:
                print(f"❌ LLM error: {e}")
                break  # Non-rate-limit error, don't retry

    # All retries failed
    print(f"❌ LLM error after retries: {last_error}")
    return {
        "response": "I apologize, but I encountered an error generating a response. "
                   "Please try again in a moment or contact our support team for assistance.",
        "needs_escalation": True,
        "escalation_reason": f"LLM generation error: {str(last_error)}",
    }


def escalate_to_human_node(state: GraphState) -> dict:
    """
    Node 5: Handle HITL escalation.

    Creates an escalation ticket and generates a user-facing message
    explaining that the query is being transferred to a human agent.
    """
    print("\n═══ Node: ESCALATE TO HUMAN ═══")

    # Create escalation ticket
    ticket = create_escalation_ticket(state)

    # Generate user-facing escalation message
    escalation_message = get_escalation_message(state)

    return {
        "response": escalation_message,
        "escalation_ticket": ticket,
        "needs_escalation": True,
    }


# ─────────────────────────────────────────────
# Routing Functions (Conditional Edges)
# ─────────────────────────────────────────────

def route_after_intent(state: GraphState) -> str:
    """
    Route after intent classification.

    If the intent requires immediate escalation (COMPLAINT, ESCALATE,
    OUT_OF_SCOPE), skip retrieval and go directly to escalation.
    Otherwise, proceed to document retrieval.
    """
    if state.get("needs_escalation", False):
        print("🔀 Routing → ESCALATION (intent-based)")
        return "escalate_to_human"
    else:
        print("🔀 Routing → RETRIEVE DOCUMENTS")
        return "retrieve_documents"


def route_after_confidence(state: GraphState) -> str:
    """
    Route after confidence evaluation.

    If retrieval confidence is too low, escalate to human.
    Otherwise, proceed to response generation.
    """
    if state.get("needs_escalation", False):
        print("🔀 Routing → ESCALATION (low confidence)")
        return "escalate_to_human"
    else:
        print("🔀 Routing → GENERATE RESPONSE")
        return "generate_response"


# ─────────────────────────────────────────────
# Graph Construction
# ─────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Build and compile the LangGraph StateGraph workflow.

    Graph Structure:
        START → classify_intent
            ─→ [escalation needed] → escalate_to_human → END
            ─→ [normal flow] → retrieve_documents → evaluate_confidence
                ─→ [low confidence] → escalate_to_human → END
                ─→ [high confidence] → generate_response → END

    Returns:
        Compiled StateGraph ready for execution.
    """
    # Initialize the graph with state schema
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("retrieve_documents", retrieve_documents_node)
    workflow.add_node("evaluate_confidence", evaluate_confidence_node)
    workflow.add_node("generate_response", generate_response_node)
    workflow.add_node("escalate_to_human", escalate_to_human_node)

    # Set entry point
    workflow.set_entry_point("classify_intent")

    # Add conditional edge: after intent classification
    workflow.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {
            "escalate_to_human": "escalate_to_human",
            "retrieve_documents": "retrieve_documents",
        }
    )

    # Add edge: retrieve → evaluate
    workflow.add_edge("retrieve_documents", "evaluate_confidence")

    # Add conditional edge: after confidence evaluation
    workflow.add_conditional_edges(
        "evaluate_confidence",
        route_after_confidence,
        {
            "escalate_to_human": "escalate_to_human",
            "generate_response": "generate_response",
        }
    )

    # Add terminal edges
    workflow.add_edge("generate_response", END)
    workflow.add_edge("escalate_to_human", END)

    # Compile the graph
    graph = workflow.compile()

    print("✅ LangGraph workflow compiled successfully")
    return graph


def run_graph(query: str, chat_history: list = None) -> dict:
    """
    Execute the graph workflow with a user query.

    Args:
        query: The user's input query.
        chat_history: Optional list of previous messages.

    Returns:
        Final state dictionary containing the response and metadata.
    """
    graph = build_graph()

    # Initialize the input state
    initial_state = {
        "query": query,
        "intent": "",
        "intent_confidence": "",
        "retrieved_docs": [],
        "retrieval_confidence": 0.0,
        "confidence_details": {},
        "context": "",
        "response": "",
        "needs_escalation": False,
        "escalation_reason": "",
        "escalation_ticket": None,
        "human_response": "",
        "chat_history": chat_history or [],
        "sources": [],
    }

    print(f"\n{'='*60}")
    print(f"🚀 Processing query: '{query[:80]}...' " if len(query) > 80 else f"🚀 Processing query: '{query}'")
    print(f"{'='*60}")

    # Execute the graph
    final_state = graph.invoke(initial_state)

    print(f"\n{'='*60}")
    print(f"✅ Workflow complete | Intent: {final_state.get('intent')} | "
          f"Escalated: {final_state.get('needs_escalation')}")
    print(f"{'='*60}\n")

    return final_state
