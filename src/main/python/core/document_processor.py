"""
Document processing pipeline for the RAG system.

This module handles the complete document processing workflow:
PDF → Text Extraction → Chunking → Chunks ready for embedding
"""

from typing import List
from pathlib import Path
from src.main.python.models.schemas import Document, Chunk
from src.main.python.services.pdf_service import PDFService
from src.main.python.utils.chunker import BaseChunker, get_chunker
from src.main.python.utils.config import Config
from src.main.python.utils.logger import get_logger


logger = get_logger(__name__)


class DocumentProcessor:
    """Processes documents through extraction and chunking pipeline."""

    def __init__(self, config: Config, pdf_service: PDFService = None, chunker: BaseChunker = None):
        """
        Initialize document processor.

        Args:
            config: Configuration object
            pdf_service: Optional PDF service instance (creates new if None)
            chunker: Optional chunker instance (creates from config if None)
        """
        self.config = config
        self.pdf_service = pdf_service or PDFService()
        self.chunker = chunker or get_chunker(config)

        logger.info(
            f"Initialized DocumentProcessor with "
            f"chunking strategy: {config.chunking_strategy}"
        )

    def process_pdf(self, pdf_path: str) -> List[Chunk]:
        """
        Process a PDF file through the complete pipeline.

        Workflow:
        1. Validate PDF
        2. Extract text by page
        3. Chunk extracted text
        4. Return list of chunks ready for embedding

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of Chunk objects

        Raises:
            FileNotFoundError: If PDF file not found
            Exception: If processing fails
        """
        pdf_path = Path(pdf_path).resolve()
        logger.info(f"Processing PDF: {pdf_path}")

        try:
            # Step 1: Validate PDF
            logger.debug("Validating PDF...")
            self.pdf_service.validate_pdf(str(pdf_path))

            # Step 2: Extract text with metadata
            logger.debug("Extracting text from PDF...")
            documents = self.pdf_service.extract_text_with_metadata(str(pdf_path))
            logger.info(f"Extracted {len(documents)} pages from PDF")

            # Step 3: Chunk all documents
            logger.debug("Chunking documents...")
            all_chunks = []

            for doc in documents:
                chunks = self.chunker.chunk(doc.content, doc.metadata)
                all_chunks.extend(chunks)

            logger.info(
                f"Successfully processed PDF into {len(all_chunks)} chunks "
                f"from {len(documents)} pages"
            )

            return all_chunks

        except Exception as e:
            error_msg = f"Failed to process PDF {pdf_path}: {e}"
            logger.error(error_msg)
            raise Exception(error_msg) from e

    def process_text(self, text: str, metadata: dict = None) -> List[Chunk]:
        """
        Process raw text through chunking pipeline.

        Useful for processing text from sources other than PDF.

        Args:
            text: Raw text to process
            metadata: Optional metadata to attach to chunks

        Returns:
            List of Chunk objects

        Raises:
            ValueError: If text is empty
            Exception: If chunking fails
        """
        if not text or not text.strip():
            raise ValueError("Cannot process empty text")

        metadata = metadata or {}
        logger.info(f"Processing {len(text)} characters of text")

        try:
            chunks = self.chunker.chunk(text, metadata)
            logger.info(f"Successfully created {len(chunks)} chunks")
            return chunks

        except Exception as e:
            error_msg = f"Failed to process text: {e}"
            logger.error(error_msg)
            raise Exception(error_msg) from e

    def get_chunks_summary(self, chunks: List[Chunk]) -> dict:
        """
        Get summary statistics about chunks.

        Args:
            chunks: List of chunks to analyze

        Returns:
            Dictionary with summary statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_chars": 0,
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0,
            }

        chunk_sizes = [len(chunk.content) for chunk in chunks]

        summary = {
            "total_chunks": len(chunks),
            "total_chars": sum(chunk_sizes),
            "avg_chunk_size": sum(chunk_sizes) // len(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
        }

        logger.info(f"Chunks summary: {summary}")
        return summary
