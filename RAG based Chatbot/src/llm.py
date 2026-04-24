"""
LLM Module for the RAG Customer Support Assistant.

Handles LLM initialization, embedding model setup, and prompt template
construction for RAG-based response generation.
"""

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from src.config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE


def get_llm() -> ChatGroq:
    """
    Initialize and return the Groq LLM (Llama 3.3 70B).

    Groq provides fast inference with generous free-tier limits:
    - 14,400 requests/day, 30 requests/minute
    - No credit card required

    Returns:
        Configured ChatGroq instance.
    """
    llm = ChatGroq(
        model=LLM_MODEL,
        api_key=GROQ_API_KEY,
        temperature=LLM_TEMPERATURE,
    )
    return llm


# Cache the embedding model so it's only loaded once
_embedding_model = None

def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Initialize and return a local HuggingFace embedding model.

    Uses 'all-MiniLM-L6-v2' which runs entirely on your machine:
    - No API calls, no rate limits, no cost
    - ~80MB download on first run (cached afterwards)
    - 384-dimensional vectors

    Returns:
        Configured HuggingFaceEmbeddings instance.
    """
    global _embedding_model
    if _embedding_model is None:
        print("📦 Loading local embedding model (first time may take a moment)...")
        _embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("✅ Local embedding model loaded successfully")
    return _embedding_model


# ─────────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """You are a helpful and professional AI Customer Support Assistant.
Your role is to assist customers by answering their questions accurately using ONLY
the information provided in the context below.

IMPORTANT RULES:
1. Answer ONLY based on the provided context. Do not make up information.
2. If the context does not contain enough information to answer the question,
   clearly state that you don't have enough information and suggest the customer
   contact a human agent for further assistance.
3. Be polite, concise, and professional in your responses.
4. If the customer seems frustrated or upset, acknowledge their feelings empathetically.
5. Provide specific details (e.g., steps, policies, contact info) when available in the context.
6. Format your response clearly with bullet points or numbered steps when appropriate.

CONTEXT FROM KNOWLEDGE BASE:
{context}
"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{query}"),
])


INTENT_CLASSIFICATION_PROMPT = """Analyze the following customer query and classify its intent
into exactly ONE of these categories:

- GENERAL_QUERY: Standard FAQ questions about products, services, features, or general information
- TECHNICAL_ISSUE: Technical problems, error messages, troubleshooting requests, connectivity issues
- BILLING: Questions about payments, invoices, refunds, pricing, subscription plans, charges
- COMPLAINT: Customer expressing dissatisfaction, frustration, anger, or filing a complaint
- ESCALATE: Customer explicitly requesting to speak with a human agent, manager, or supervisor
- OUT_OF_SCOPE: Query completely unrelated to customer support (e.g., general knowledge, personal questions)

Customer Query: {query}

Respond with ONLY the category name (e.g., "GENERAL_QUERY"). Nothing else."""


def create_rag_chain(retriever, llm):
    """
    Build the RAG chain: Retriever → Prompt → LLM → Output Parser.

    This chain retrieves relevant documents, formats them into a prompt,
    sends it to the LLM, and parses the output.

    Args:
        retriever: LangChain retriever for document retrieval.
        llm: LLM instance for response generation.

    Returns:
        Configured RAG chain.
    """
    from langchain_core.runnables import RunnablePassthrough

    def format_docs(docs):
        """Format retrieved documents into a single context string."""
        return "\n\n---\n\n".join(
            f"[Source: {doc.metadata.get('source_file', 'N/A')}, "
            f"Page {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
            for doc in docs
        )

    chain = (
        {
            "context": retriever | format_docs,
            "query": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain
