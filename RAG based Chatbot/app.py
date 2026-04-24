"""
Streamlit Web Application for the RAG Customer Support Assistant.

Provides a rich chat interface with:
- PDF upload and knowledge base management
- Real-time chat with the RAG pipeline
- HITL escalation handling with human response input
- Source document display
- System status monitoring
"""

import sys
import os

# Suppress noisy transformers/tokenizers warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Fix Windows console encoding for emoji support in print statements
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

import streamlit as st
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import APP_TITLE, APP_ICON, WELCOME_MESSAGE
from src.document_processor import process_knowledge_base
from src.vector_store import initialize_vectorstore, add_to_vectorstore, get_collection_stats, clear_vectorstore
from src.llm import get_embeddings
from src.graph_workflow import run_graph
from src.hitl import integrate_human_response


# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Customer Support Assistant",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS for Premium Look
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 0.95rem;
    }

    /* Chat message styling */
    .stChatMessage {
        border-radius: 12px !important;
        margin-bottom: 0.5rem !important;
    }

    /* Status cards */
    .status-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem 1.2rem;
        border-radius: 10px;
        margin-bottom: 0.8rem;
        border-left: 4px solid #667eea;
    }
    .status-card.success {
        border-left-color: #27ae60;
        background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%);
    }
    .status-card.warning {
        border-left-color: #f39c12;
        background: linear-gradient(135deg, #fffbea 0%, #fef3c7 100%);
    }
    .status-card.error {
        border-left-color: #e74c3c;
        background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%);
    }

    /* Escalation alert */
    .escalation-alert {
        background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%);
        border: 2px solid #fc8181;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
    }
    .escalation-alert h4 {
        color: #c53030;
        margin: 0 0 0.5rem 0;
    }

    /* Source cards */
    .source-card {
        background: #f8f9fa;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.3rem 0;
        font-size: 0.85rem;
    }
    .source-card .score {
        color: #667eea;
        font-weight: 600;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #fff !important;
    }

    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    /* Metric styling */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 0.8rem;
        border: 1px solid rgba(255,255,255,0.1);
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "kb_initialized" not in st.session_state:
    st.session_state.kb_initialized = False
if "pending_escalation" not in st.session_state:
    st.session_state.pending_escalation = None
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📚 Knowledge Base")
    st.markdown("---")

    # PDF Upload
    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        help="Upload a customer support guide or FAQ document",
    )

    if uploaded_file is not None:
        if st.button("📥 Ingest Document", use_container_width=True, type="primary"):
            with st.spinner("Processing document..."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    # Process and add to existing knowledge base
                    chunks = process_knowledge_base(tmp_path)
                    embeddings = get_embeddings()
                    add_to_vectorstore(chunks, embeddings)

                    # Cleanup temp file
                    os.unlink(tmp_path)

                    st.session_state.kb_initialized = True
                    st.success(f"✅ Added {len(chunks)} chunks from '{uploaded_file.name}' to the knowledge base")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    # Or use default knowledge base
    default_kb_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "knowledge_base"
    )
    pdf_files = []
    if os.path.exists(default_kb_path):
        pdf_files = [f for f in os.listdir(default_kb_path) if f.endswith(".pdf")]

    if pdf_files:
        if st.button("📂 Load Default Knowledge Base", use_container_width=True):
            with st.spinner("Processing default knowledge base..."):
                try:
                    chunks = process_knowledge_base()
                    embeddings = get_embeddings()
                    initialize_vectorstore(chunks, embeddings)
                    st.session_state.kb_initialized = True
                    st.success(f"✅ Loaded {len(chunks)} chunks from {len(pdf_files)} file(s)")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    # Knowledge Base Status
    st.markdown("---")
    st.markdown("### 📊 System Status")

    try:
        embeddings = get_embeddings()
        stats = get_collection_stats(embeddings)
        if stats["status"] == "active":
            st.session_state.kb_initialized = True
            source_files = stats.get("source_files", [])
            sources_text = ", ".join(source_files) if source_files else "Unknown"
            st.markdown(
                f'<div class="status-card success">'
                f'<strong>✅ Knowledge Base Active</strong><br>'
                f'Documents: {stats["document_count"]}<br>'
                f'<small>Sources: {sources_text}</small>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="status-card warning">'
                '<strong>⚠️ No Knowledge Base</strong><br>'
                'Please upload a PDF to get started.'
                '</div>',
                unsafe_allow_html=True
            )
    except Exception:
        st.markdown(
            '<div class="status-card warning">'
            '<strong>⚠️ No Knowledge Base</strong><br>'
            'Please upload a PDF to get started.'
            '</div>',
            unsafe_allow_html=True
        )

    # Clear options
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear KB", use_container_width=True):
            clear_vectorstore()
            st.session_state.kb_initialized = False
            st.rerun()
    with col2:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.pending_escalation = None
            st.session_state.last_sources = []
            st.rerun()

    # Architecture info
    st.markdown("---")
    st.markdown("### 🏗️ Architecture")
    st.markdown("""
    - **LLM**: Groq (Llama 3.3 70B)
    - **Embeddings**: HuggingFace (Local)
    - **Vector DB**: ChromaDB
    - **Workflow**: LangGraph
    - **HITL**: Escalation System
    """)


# ─────────────────────────────────────────────
# Main Chat Area
# ─────────────────────────────────────────────

# Header
st.markdown(
    f"""
    <div class="main-header">
        <h1>{APP_TITLE}</h1>
        <p>Powered by RAG + LangGraph + Human-in-the-Loop Escalation</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Display welcome message if no messages
if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(WELCOME_MESSAGE)

# Display chat history
for message in st.session_state.messages:
    role = message["role"]
    avatar = "👤" if role == "user" else "🤖"
    if role == "escalation":
        avatar = "🧑‍💼"
    with st.chat_message(role if role != "escalation" else "assistant", avatar=avatar):
        st.markdown(message["content"])

        # Show sources if available
        if message.get("sources"):
            with st.expander("📚 View Sources", expanded=False):
                for src in message["sources"]:
                    st.markdown(
                        f'<div class="source-card">'
                        f'<span class="score">Score: {src["score"]}</span> | '
                        f'{src["source"]} (Page {src["page"]})<br>'
                        f'<small>{src["content"][:150]}...</small>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        # Show escalation info
        if message.get("escalated"):
            st.markdown(
                '<div class="escalation-alert">'
                '<h4>⚠️ Escalated to Human Agent</h4>'
                f'<strong>Reason:</strong> {message.get("escalation_reason", "N/A")}<br>'
                f'<strong>Intent:</strong> {message.get("intent", "N/A")}'
                '</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────
# Handle Pending Escalation (HITL Input)
# ─────────────────────────────────────────────
if st.session_state.pending_escalation is not None:
    st.markdown("---")
    st.markdown(
        '<div class="escalation-alert">'
        '<h4>🧑‍💼 Human Agent Response Required</h4>'
        '<p>The AI has escalated this query. Please provide a response as a human agent, '
        'or click "Skip" to let the AI handle it.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.form("hitl_form"):
        human_response = st.text_area(
            "Enter human agent response:",
            placeholder="Type your response to the customer here...",
            height=100,
        )
        col1, col2 = st.columns([3, 1])
        with col1:
            submitted = st.form_submit_button("📤 Send Human Response", type="primary", use_container_width=True)
        with col2:
            skipped = st.form_submit_button("⏭️ Skip", use_container_width=True)

    if submitted and human_response:
        # Integrate human response
        updated = integrate_human_response(st.session_state.pending_escalation, human_response)
        st.session_state.messages.append({
            "role": "escalation",
            "content": updated["response"],
        })
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": updated["response"],
        })
        st.session_state.pending_escalation = None
        st.rerun()

    if skipped:
        st.session_state.pending_escalation = None
        st.rerun()


# ─────────────────────────────────────────────
# Chat Input
# ─────────────────────────────────────────────
if prompt := st.chat_input("Ask a question...", key="chat_input"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Check if knowledge base is loaded
    if not st.session_state.kb_initialized:
        error_msg = (
            "⚠️ **Knowledge base not loaded.** Please upload a PDF document "
            "using the sidebar before asking questions."
        )
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(error_msg)
    else:
        # Process with LangGraph workflow
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🔍 Searching knowledge base & generating response..."):
                try:
                    result = run_graph(prompt, st.session_state.chat_history)

                    response = result.get("response", "I couldn't generate a response.")
                    is_escalated = result.get("needs_escalation", False)
                    sources = result.get("sources", [])

                    # Display response
                    st.markdown(response)

                    # Build message record
                    message_record = {
                        "role": "assistant",
                        "content": response,
                        "sources": sources,
                        "escalated": is_escalated,
                        "intent": result.get("intent", ""),
                        "escalation_reason": result.get("escalation_reason", ""),
                    }

                    # Show sources
                    if sources:
                        with st.expander("📚 View Sources", expanded=False):
                            for src in sources:
                                st.markdown(
                                    f'<div class="source-card">'
                                    f'<span class="score">Score: {src["score"]}</span> | '
                                    f'{src["source"]} (Page {src["page"]})<br>'
                                    f'<small>{src["content"][:150]}...</small>'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )

                    # Show escalation info
                    if is_escalated:
                        st.markdown(
                            '<div class="escalation-alert">'
                            '<h4>⚠️ Escalated to Human Agent</h4>'
                            f'<strong>Reason:</strong> {result.get("escalation_reason", "N/A")}<br>'
                            f'<strong>Intent:</strong> {result.get("intent", "N/A")}'
                            '</div>',
                            unsafe_allow_html=True,
                        )
                        # Set pending escalation for HITL input
                        st.session_state.pending_escalation = result

                    st.session_state.messages.append(message_record)

                    # Update chat history
                    st.session_state.chat_history.append({"role": "user", "content": prompt})
                    st.session_state.chat_history.append({"role": "assistant", "content": response})

                except Exception as e:
                    error_msg = f"❌ An error occurred: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

        # Rerun to show HITL form if escalated
        if result.get("needs_escalation", False):
            st.rerun()
