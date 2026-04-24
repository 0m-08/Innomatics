"""
Document Processing Module for the RAG Customer Support Assistant.

Handles PDF loading, text chunking, and document preparation
for embedding and storage in the vector database.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, PDF_DIRECTORY
import os


def load_pdf(pdf_path: str) -> list:
    """
    Load a PDF file and extract text content as LangChain Document objects.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        List of Document objects, one per page.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If the PDF contains no extractable text.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    if not documents:
        raise ValueError(f"No text content could be extracted from: {pdf_path}")

    # Add source metadata
    for doc in documents:
        doc.metadata["source_file"] = os.path.basename(pdf_path)

    print(f"✅ Loaded {len(documents)} pages from '{os.path.basename(pdf_path)}'")
    return documents


def chunk_documents(documents: list, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list:
    """
    Split documents into smaller chunks for embedding.

    Uses RecursiveCharacterTextSplitter which tries to split on natural
    boundaries (paragraphs → sentences → words) to preserve context.

    Args:
        documents: List of LangChain Document objects.
        chunk_size: Maximum characters per chunk (default: 500).
        chunk_overlap: Character overlap between chunks (default: 100).

    Returns:
        List of chunked Document objects with preserved metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],  # Priority order for splitting
        add_start_index=True,  # Track chunk position in original document
    )

    chunks = text_splitter.split_documents(documents)

    # Add chunk index metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["total_chunks"] = len(chunks)

    print(f"✅ Created {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def process_knowledge_base(pdf_path: str = None) -> list:
    """
    End-to-end pipeline: Load PDF → Chunk into segments.

    If no path is provided, processes all PDFs in the knowledge_base directory.

    Args:
        pdf_path: Optional specific PDF path. If None, processes all PDFs
                  in the knowledge_base directory.

    Returns:
        List of all chunked Document objects ready for embedding.
    """
    all_chunks = []

    if pdf_path:
        # Process a single PDF
        documents = load_pdf(pdf_path)
        chunks = chunk_documents(documents)
        all_chunks.extend(chunks)
    else:
        # Process all PDFs in the knowledge base directory
        if not os.path.exists(PDF_DIRECTORY):
            os.makedirs(PDF_DIRECTORY, exist_ok=True)
            raise FileNotFoundError(
                f"No PDFs found. Place PDF files in: {PDF_DIRECTORY}"
            )

        pdf_files = [f for f in os.listdir(PDF_DIRECTORY) if f.lower().endswith(".pdf")]

        if not pdf_files:
            raise FileNotFoundError(
                f"No PDF files found in: {PDF_DIRECTORY}"
            )

        for pdf_file in pdf_files:
            full_path = os.path.join(PDF_DIRECTORY, pdf_file)
            try:
                documents = load_pdf(full_path)
                chunks = chunk_documents(documents)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"⚠️ Error processing {pdf_file}: {e}")

    print(f"\n📊 Total chunks prepared for embedding: {len(all_chunks)}")
    return all_chunks
