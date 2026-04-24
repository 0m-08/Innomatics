"""
Configuration module for the RAG Customer Support Assistant.
Centralizes all settings, thresholds, and environment variable loading.
"""

import os
import sys

# Fix Windows console encoding for emoji support
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ─────────────────────────────────────────────
# API Keys
# ─────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Please set it in your .env file.\n"
        "Get a free key at: https://console.groq.com"
    )

# ─────────────────────────────────────────────
# LLM Configuration (Groq + Llama 3.3 70B)
# ─────────────────────────────────────────────
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.3          # Lower for factual responses

# ─────────────────────────────────────────────
# Document Processing
# ─────────────────────────────────────────────
CHUNK_SIZE = 500               # Characters per chunk
CHUNK_OVERLAP = 100            # Overlap between consecutive chunks
PDF_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_base")

# ─────────────────────────────────────────────
# Vector Store (ChromaDB)
# ─────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
CHROMA_COLLECTION_NAME = "customer_support_docs"

# ─────────────────────────────────────────────
# Retrieval Settings
# ─────────────────────────────────────────────
RETRIEVAL_TOP_K = 3                    # Number of chunks to retrieve
CONFIDENCE_THRESHOLD = 0.3             # Below this → escalate to human
HIGH_CONFIDENCE_THRESHOLD = 0.5        # Above this → high confidence answer

# ─────────────────────────────────────────────
# Intent Categories
# ─────────────────────────────────────────────
INTENT_CATEGORIES = [
    "GENERAL_QUERY",       # Standard FAQ questions
    "TECHNICAL_ISSUE",     # Technical support requests
    "BILLING",             # Billing / account inquiries
    "COMPLAINT",           # Complaints / negative sentiment
    "ESCALATE",            # Explicit request for human agent
    "OUT_OF_SCOPE",        # Outside knowledge base
]

# ─────────────────────────────────────────────
# HITL (Human-in-the-Loop) Settings
# ─────────────────────────────────────────────
ESCALATION_INTENTS = ["COMPLAINT", "ESCALATE", "OUT_OF_SCOPE"]
MAX_RETRIES_BEFORE_ESCALATION = 2      # Retry count before auto-escalation

# ─────────────────────────────────────────────
# UI Settings
# ─────────────────────────────────────────────
APP_TITLE = "🤖 AI Customer Support Assistant"
APP_ICON = "🤖"
WELCOME_MESSAGE = (
    "Hello! I'm your AI Customer Support Assistant. "
    "I can help you with account management, billing, technical issues, and more. "
    "How can I assist you today?"
)
