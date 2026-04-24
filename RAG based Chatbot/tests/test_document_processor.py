"""Tests for the Document Processing module."""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.document_processor import load_pdf, chunk_documents, process_knowledge_base


class TestDocumentProcessor:
    """Test suite for document processing pipeline."""

    PDF_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "knowledge_base", "customer_support_guide.pdf"
    )

    def test_load_pdf_returns_documents(self):
        """Test that PDF loading returns non-empty list of documents."""
        docs = load_pdf(self.PDF_PATH)
        assert len(docs) > 0, "PDF should return at least one document"

    def test_load_pdf_has_content(self):
        """Test that loaded documents contain text content."""
        docs = load_pdf(self.PDF_PATH)
        for doc in docs:
            assert hasattr(doc, 'page_content'), "Document should have page_content"

    def test_load_pdf_has_metadata(self):
        """Test that loaded documents contain metadata."""
        docs = load_pdf(self.PDF_PATH)
        for doc in docs:
            assert 'source_file' in doc.metadata, "Document should have source_file metadata"

    def test_load_pdf_invalid_path(self):
        """Test that loading non-existent PDF raises error."""
        with pytest.raises(FileNotFoundError):
            load_pdf("nonexistent.pdf")

    def test_chunk_documents_creates_chunks(self):
        """Test that chunking produces multiple chunks."""
        docs = load_pdf(self.PDF_PATH)
        chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=100)
        assert len(chunks) > len(docs), "Chunking should produce more chunks than pages"

    def test_chunk_size_respected(self):
        """Test that chunks respect the size limit (approximately)."""
        docs = load_pdf(self.PDF_PATH)
        chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=100)
        for chunk in chunks:
            assert len(chunk.page_content) <= 600, "Chunks should not greatly exceed chunk_size"

    def test_chunk_metadata_preserved(self):
        """Test that chunk metadata includes chunk index."""
        docs = load_pdf(self.PDF_PATH)
        chunks = chunk_documents(docs)
        for chunk in chunks:
            assert 'chunk_index' in chunk.metadata
            assert 'total_chunks' in chunk.metadata

    def test_process_knowledge_base(self):
        """Test end-to-end knowledge base processing."""
        chunks = process_knowledge_base(self.PDF_PATH)
        assert len(chunks) > 0, "Processing should return chunks"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
