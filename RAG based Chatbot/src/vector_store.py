"""
Vector Store Module for the RAG Customer Support Assistant.

Manages ChromaDB initialization, document storage, and collection management.
Provides persistent vector storage for document embeddings.
"""

import os
import shutil
import gc
import time
from langchain_chroma import Chroma
from src.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME

# Module-level reference to the active vectorstore so we can close it before clearing
_active_vectorstore = None


def initialize_vectorstore(chunks: list, embeddings) -> Chroma:
    """
    Create a new ChromaDB vector store from document chunks.

    Embeds all chunks and stores them persistently in ChromaDB.
    If a collection already exists, it is cleared first.

    Args:
        chunks: List of LangChain Document objects (chunked).
        embeddings: Embedding model instance (e.g., GoogleGenerativeAIEmbeddings).

    Returns:
        Initialized Chroma vector store instance.
    """
    global _active_vectorstore

    # Close any existing connection first
    _close_active_store()

    # Clear existing data to avoid duplicates
    _force_remove_dir(CHROMA_PERSIST_DIR)

    # Create vector store from chunks (with retry for rate limits)
    vectorstore = None
    for attempt in range(3):
        try:
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=CHROMA_PERSIST_DIR,
                collection_name=CHROMA_COLLECTION_NAME,
            )
            break
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = 20 * (attempt + 1)
                print(f"⏳ Embedding rate limited. Retrying in {wait}s... (attempt {attempt+1}/3)")
                time.sleep(wait)
            else:
                raise

    if vectorstore is None:
        raise RuntimeError("Failed to create vector store after 3 retries. Please wait a minute and try again.")

    _active_vectorstore = vectorstore
    print(f"✅ Vector store created with {len(chunks)} documents")
    print(f"📁 Persisted at: {CHROMA_PERSIST_DIR}")
    return vectorstore


def add_to_vectorstore(chunks: list, embeddings) -> Chroma:
    """
    Add new document chunks to an existing ChromaDB vector store.

    If no vector store exists yet, creates a new one.
    Does NOT clear existing data — new chunks are appended.

    Args:
        chunks: List of LangChain Document objects (chunked).
        embeddings: Embedding model instance.

    Returns:
        Updated Chroma vector store instance.
    """
    global _active_vectorstore

    # If no existing store, just create a fresh one
    if not os.path.exists(CHROMA_PERSIST_DIR):
        return initialize_vectorstore(chunks, embeddings)

    # Open the existing store
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
    )

    existing_count = vectorstore._collection.count()
    print(f"📂 Existing vector store has {existing_count} documents")

    # Add new chunks (with retry for rate limits)
    for attempt in range(3):
        try:
            vectorstore.add_documents(chunks)
            break
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = 20 * (attempt + 1)
                print(f"⏳ Embedding rate limited. Retrying in {wait}s... (attempt {attempt+1}/3)")
                time.sleep(wait)
            else:
                raise

    new_count = vectorstore._collection.count()
    _active_vectorstore = vectorstore
    print(f"✅ Added {len(chunks)} chunks → total now {new_count} documents")
    return vectorstore


def get_vectorstore(embeddings) -> Chroma:
    """
    Load an existing ChromaDB vector store.

    Args:
        embeddings: Embedding model instance (must match the one used during creation).

    Returns:
        Loaded Chroma vector store instance.

    Raises:
        FileNotFoundError: If no persisted vector store exists.
    """
    global _active_vectorstore

    if not os.path.exists(CHROMA_PERSIST_DIR):
        raise FileNotFoundError(
            f"No vector store found at: {CHROMA_PERSIST_DIR}\n"
            "Please ingest documents first using the sidebar in the app."
        )

    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
    )

    # Verify the collection has documents
    count = vectorstore._collection.count()
    if count == 0:
        raise ValueError("Vector store exists but contains no documents. Please re-ingest.")

    _active_vectorstore = vectorstore
    print(f"✅ Loaded vector store with {count} documents")
    return vectorstore


def clear_vectorstore():
    """
    Delete the entire vector store directory.

    Properly closes any active ChromaDB connection before deleting
    to avoid WinError 32 file-lock issues on Windows.
    """
    _close_active_store()
    _force_remove_dir(CHROMA_PERSIST_DIR)


def _close_active_store():
    """Close the active vectorstore connection and release file handles."""
    global _active_vectorstore
    if _active_vectorstore is not None:
        try:
            # Try to delete the collection via the client API
            client = _active_vectorstore._client
            try:
                client.delete_collection(CHROMA_COLLECTION_NAME)
            except Exception:
                pass
        except Exception:
            pass
        _active_vectorstore = None

    # Force garbage collection to release any lingering file handles
    gc.collect()
    time.sleep(0.3)


def _force_remove_dir(path: str):
    """Remove a directory with retries for Windows file-lock issues."""
    if not os.path.exists(path):
        print("ℹ️  No vector store to clear")
        return

    for attempt in range(3):
        try:
            shutil.rmtree(path)
            print("🗑️  Vector store cleared successfully")
            return
        except PermissionError:
            gc.collect()
            time.sleep(0.5 * (attempt + 1))

    # Last resort: try to remove individual files
    print("⚠️  Could not fully remove vector store (files may be locked). "
          "It will be overwritten on next ingestion.")


def get_collection_stats(embeddings) -> dict:
    """
    Get statistics about the current vector store collection.

    Args:
        embeddings: Embedding model instance.

    Returns:
        Dictionary with collection statistics including source files.
    """
    try:
        vectorstore = get_vectorstore(embeddings)
        count = vectorstore._collection.count()

        # Get unique source files in the collection
        source_files = set()
        try:
            all_meta = vectorstore._collection.get(include=["metadatas"])
            for meta in all_meta.get("metadatas", []):
                if meta and meta.get("source_file"):
                    source_files.add(meta["source_file"])
        except Exception:
            pass

        return {
            "status": "active",
            "document_count": count,
            "persist_directory": CHROMA_PERSIST_DIR,
            "collection_name": CHROMA_COLLECTION_NAME,
            "source_files": sorted(source_files),
        }
    except (FileNotFoundError, ValueError):
        return {
            "status": "empty",
            "document_count": 0,
            "persist_directory": CHROMA_PERSIST_DIR,
            "collection_name": CHROMA_COLLECTION_NAME,
            "source_files": [],
        }
