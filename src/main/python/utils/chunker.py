"""
Text chunking utilities for the RAG system.

This module provides different chunking strategies for splitting text into
manageable pieces for embedding and retrieval.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import re
import tiktoken
from src.main.python.models.schemas import Chunk, Document
from src.main.python.utils.logger import get_logger
from src.main.python.utils.config import Config


logger = get_logger(__name__)


class BaseChunker(ABC):
    """Abstract base class for text chunking strategies."""

    def __init__(self, config: Config):
        """
        Initialize chunker with configuration.

        Args:
            config: Configuration object
        """
        self.config = config
        self.max_tokens = config.chunk_size
        self.overlap_tokens = config.chunk_overlap

        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Could not load tiktoken encoding: {e}. Using character-based estimation.")
            self.tokenizer = None

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback: estimate as chars / 4
            return len(text) // 4

    @abstractmethod
    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        """
        Chunk text into smaller pieces.

        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to chunks

        Returns:
            List of Chunk objects
        """
        pass


class ParagraphChunker(BaseChunker):
    """
    Paragraph-based chunking strategy.

    Splits text by paragraphs (double newlines) and combines paragraphs
    until reaching max_tokens limit. Ideal for CV documents which have
    natural section boundaries.
    """

    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        """
        Chunk text by paragraphs.

        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to chunks

        Returns:
            List of Chunk objects
        """
        metadata = metadata or {}

        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text.strip())
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        logger.debug(f"Split text into {len(paragraphs)} paragraphs")

        chunks = []
        current_chunk = []
        current_tokens = 0

        for i, paragraph in enumerate(paragraphs):
            para_tokens = self.count_tokens(paragraph)

            # If single paragraph exceeds max_tokens, split it by sentences
            if para_tokens > self.max_tokens:
                logger.warning(
                    f"Paragraph {i} has {para_tokens} tokens, exceeds limit. "
                    "Splitting by sentences."
                )
                # Save current chunk if not empty
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    chunks.append(self._create_chunk(chunk_text, metadata, len(chunks)))
                    current_chunk = []
                    current_tokens = 0

                # Split long paragraph by sentences
                sentence_chunks = self._split_by_sentences(paragraph, metadata)
                chunks.extend(sentence_chunks)
                continue

            # Check if adding this paragraph would exceed limit
            if current_tokens + para_tokens > self.max_tokens and current_chunk:
                # Save current chunk
                chunk_text = "\n\n".join(current_chunk)
                chunks.append(self._create_chunk(chunk_text, metadata, len(chunks)))

                # Start new chunk with overlap
                if self.overlap_tokens > 0 and current_chunk:
                    # Keep last paragraph for overlap
                    current_chunk = [current_chunk[-1], paragraph]
                    current_tokens = self.count_tokens("\n\n".join(current_chunk))
                else:
                    current_chunk = [paragraph]
                    current_tokens = para_tokens
            else:
                # Add to current chunk
                current_chunk.append(paragraph)
                current_tokens += para_tokens

        # Add final chunk if not empty
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append(self._create_chunk(chunk_text, metadata, len(chunks)))

        logger.info(
            f"Created {len(chunks)} chunks from {len(paragraphs)} paragraphs "
            f"(max_tokens={self.max_tokens}, overlap={self.overlap_tokens})"
        )

        return chunks

    def _split_by_sentences(
        self, text: str, metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """
        Split long text by sentences when paragraph is too long.

        Args:
            text: Text to split
            metadata: Metadata to attach

        Returns:
            List of chunks
        """
        # Simple sentence splitting (can be improved with NLTK)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = self.count_tokens(sentence)

            if current_tokens + sent_tokens > self.max_tokens and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(
                    self._create_chunk(chunk_text, metadata, len(chunks))
                )
                current_chunk = [sentence]
                current_tokens = sent_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sent_tokens

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(self._create_chunk(chunk_text, metadata, len(chunks)))

        return chunks

    def _create_chunk(
        self, text: str, metadata: Dict[str, Any], chunk_index: int
    ) -> Chunk:
        """
        Create a Chunk object with metadata.

        Args:
            text: Chunk text
            metadata: Base metadata
            chunk_index: Index of this chunk

        Returns:
            Chunk object
        """
        chunk_metadata = {
            **metadata,
            "chunk_index": chunk_index,
            "chunk_tokens": self.count_tokens(text),
            "chunk_chars": len(text),
        }

        return Chunk(content=text, metadata=chunk_metadata)


class SentenceChunker(BaseChunker):
    """
    Sentence-based chunking strategy.

    Splits text by sentences and groups them up to max_tokens.
    More granular than paragraph chunking.
    """

    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        """
        Chunk text by sentences.

        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to chunks

        Returns:
            List of Chunk objects
        """
        metadata = metadata or {}

        # Split into sentences (simple approach)
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        logger.debug(f"Split text into {len(sentences)} sentences")

        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = self.count_tokens(sentence)

            if current_tokens + sent_tokens > self.max_tokens and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunk_metadata = {
                    **metadata,
                    "chunk_index": len(chunks),
                    "chunk_tokens": current_tokens,
                }
                chunks.append(Chunk(content=chunk_text, metadata=chunk_metadata))

                # Add overlap
                if self.overlap_tokens > 0 and current_chunk:
                    current_chunk = [current_chunk[-1], sentence]
                    current_tokens = self.count_tokens(" ".join(current_chunk))
                else:
                    current_chunk = [sentence]
                    current_tokens = sent_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sent_tokens

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk_metadata = {
                **metadata,
                "chunk_index": len(chunks),
                "chunk_tokens": current_tokens,
            }
            chunks.append(Chunk(content=chunk_text, metadata=chunk_metadata))

        logger.info(f"Created {len(chunks)} chunks from {len(sentences)} sentences")

        return chunks


def get_chunker(config: Config) -> BaseChunker:
    """
    Factory function to get appropriate chunker based on configuration.

    Args:
        config: Configuration object

    Returns:
        Chunker instance

    Raises:
        ValueError: If chunking strategy is invalid
    """
    strategy = config.chunking_strategy.lower()

    if strategy == "paragraph":
        return ParagraphChunker(config)
    elif strategy == "sentence":
        return SentenceChunker(config)
    else:
        raise ValueError(
            f"Unknown chunking strategy: {strategy}. "
            "Choose from: paragraph, sentence"
        )
