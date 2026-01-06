"""
PDF text extraction service using pdfplumber.

This module provides functionality to extract text from PDF files
with metadata preservation (page numbers, etc.).
"""

from typing import List, Dict, Any
from pathlib import Path
import pdfplumber
from src.main.python.utils.logger import get_logger
from src.main.python.models.schemas import Document


logger = get_logger(__name__)


class PDFService:
    """Service for extracting text from PDF files."""

    def __init__(self):
        """Initialize PDF service."""
        logger.info("Initialized PDFService")

    def extract_text(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from PDF file by page.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of dictionaries with page number and text

        Raises:
            FileNotFoundError: If PDF file not found
            Exception: If PDF parsing fails
        """
        pdf_path = Path(pdf_path).resolve()

        if not pdf_path.exists():
            error_msg = f"PDF file not found: {pdf_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info(f"Extracting text from PDF: {pdf_path}")
        pages_data = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"PDF has {total_pages} pages")

                for page_num, page in enumerate(pdf.pages, start=1):
                    # Use layout-aware extraction to preserve word spacing
                    text = page.extract_text(layout=True, x_tolerance=2, y_tolerance=3)

                    # Normalize excessive whitespace while preserving structure
                    if text:
                        # Replace multiple spaces with single space
                        import re
                        text = re.sub(r' +', ' ', text)
                        # Normalize newlines (keep paragraph breaks)
                        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

                    if text:
                        pages_data.append(
                            {
                                "page": page_num,
                                "text": text,
                                "total_pages": total_pages,
                            }
                        )
                        logger.debug(
                            f"Extracted {len(text)} characters from page {page_num}"
                        )
                    else:
                        logger.warning(f"No text found on page {page_num}")

            logger.info(
                f"Successfully extracted text from {len(pages_data)} pages"
            )
            return pages_data

        except Exception as e:
            error_msg = f"Failed to extract text from PDF {pdf_path}: {e}"
            logger.error(error_msg)
            raise Exception(error_msg) from e

    def extract_text_with_metadata(self, pdf_path: str) -> List[Document]:
        """
        Extract text from PDF and return as Document objects.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of Document objects, one per page

        Raises:
            FileNotFoundError: If PDF file not found
            Exception: If PDF parsing fails
        """
        pages_data = self.extract_text(pdf_path)
        documents = []

        pdf_filename = Path(pdf_path).name

        for page_data in pages_data:
            doc = Document(
                content=page_data["text"],
                metadata={
                    "source": pdf_filename,
                    "page": page_data["page"],
                    "total_pages": page_data["total_pages"],
                    "source_type": "pdf",
                },
            )
            documents.append(doc)

        logger.info(f"Created {len(documents)} Document objects from PDF")
        return documents

    def validate_pdf(self, pdf_path: str) -> bool:
        """
        Validate that the file is a readable PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            True if PDF is valid and readable

        Raises:
            FileNotFoundError: If file doesn't exist
            Exception: If PDF is invalid or unreadable
        """
        pdf_path = Path(pdf_path).resolve()

        if not pdf_path.exists():
            raise FileNotFoundError(f"File not found: {pdf_path}")

        if not pdf_path.suffix.lower() == ".pdf":
            raise ValueError(f"File is not a PDF: {pdf_path}")

        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Try to access first page
                if len(pdf.pages) == 0:
                    raise ValueError("PDF has no pages")

            logger.info(f"PDF validation successful: {pdf_path}")
            return True

        except Exception as e:
            error_msg = f"PDF validation failed for {pdf_path}: {e}"
            logger.error(error_msg)
            raise Exception(error_msg) from e
