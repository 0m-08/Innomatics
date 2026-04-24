"""
CLI Interface for the RAG Customer Support Assistant.

Provides a simple command-line interface for testing the system
without the Streamlit web UI.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.graph_workflow import run_graph
from src.document_processor import process_knowledge_base
from src.vector_store import initialize_vectorstore, get_collection_stats
from src.llm import get_embeddings
from src.hitl import integrate_human_response
from src.config import APP_TITLE, WELCOME_MESSAGE


def ingest_documents(pdf_path: str = None):
    """Process and ingest PDF documents into ChromaDB."""
    print("\n📄 Document Ingestion Pipeline")
    print("─" * 40)

    chunks = process_knowledge_base(pdf_path)
    embeddings = get_embeddings()
    vectorstore = initialize_vectorstore(chunks, embeddings)

    print("\n✅ Knowledge base ready!")
    return vectorstore


def main():
    """Main CLI interaction loop."""
    print(f"\n{'='*60}")
    print(f"  {APP_TITLE}")
    print(f"{'='*60}")
    print(f"\n{WELCOME_MESSAGE}\n")
    print("Commands:")
    print("  /ingest <path>  - Ingest a PDF file")
    print("  /status         - Check knowledge base status")
    print("  /quit           - Exit the application")
    print("─" * 60)

    chat_history = []

    while True:
        try:
            query = input("\n👤 You: ").strip()

            if not query:
                continue

            # Handle special commands
            if query.startswith("/"):
                if query.lower() == "/quit":
                    print("\n👋 Goodbye!")
                    break
                elif query.lower() == "/status":
                    embeddings = get_embeddings()
                    stats = get_collection_stats(embeddings)
                    print(f"\n📊 Knowledge Base Status: {stats}")
                    continue
                elif query.lower().startswith("/ingest"):
                    parts = query.split(maxsplit=1)
                    path = parts[1] if len(parts) > 1 else None
                    ingest_documents(path)
                    continue
                else:
                    print("❓ Unknown command. Try /ingest, /status, or /quit")
                    continue

            # Run the graph workflow
            result = run_graph(query, chat_history)

            # Display response
            response = result.get("response", "No response generated.")
            print(f"\n🤖 Assistant: {response}")

            # Handle HITL escalation
            if result.get("needs_escalation", False):
                print("\n⚠️  This query has been escalated to a human agent.")
                human_input = input("🧑 Human Agent Response (or press Enter to skip): ").strip()

                if human_input:
                    updated = integrate_human_response(result, human_input)
                    response = updated["response"]
                    print(f"\n🤖 Assistant (via Human): {response}")

            # Show sources if available
            sources = result.get("sources", [])
            if sources:
                print("\n📚 Sources used:")
                for s in sources:
                    print(f"   • {s['source']} (Page {s['page']}, Score: {s['score']})")

            # Update chat history
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
