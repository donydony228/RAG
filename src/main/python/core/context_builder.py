"""
Context building for the RAG system.

This module assembles retrieved chunks and conversation history
into properly formatted context for Claude API.
"""

from typing import List, Dict, Any
from src.main.python.models.schemas import Chunk
from src.main.python.utils.config import Config
from src.main.python.utils.logger import get_logger


logger = get_logger(__name__)


class ContextBuilder:
    """Builds context from retrieved chunks and conversation history."""

    def __init__(self, config: Config):
        """
        Initialize context builder.

        Args:
            config: Configuration object
        """
        self.config = config
        self.system_prompt = config.system_prompt
        self.max_tokens = config.max_tokens

        logger.info("Initialized ContextBuilder")

    def build_context(
        self,
        query: str,
        chunks: List[Chunk],
        conversation_history: List[Dict[str, str]] = None,
    ) -> str:
        """
        Build context string from retrieved chunks.

        Args:
            query: User's query
            chunks: Retrieved chunks to use as context
            conversation_history: Previous conversation (not used here, managed separately)

        Returns:
            Formatted context string
        """
        if not chunks:
            return "No relevant information found in the CV."

        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            # Extract metadata for context
            page_info = ""
            if "page" in chunk.metadata:
                page_info = f" (Page {chunk.metadata['page']})"

            score_info = ""
            if "similarity_score" in chunk.metadata:
                score = chunk.metadata["similarity_score"]
                score_info = f" [Relevance: {score:.2f}]"

            # Format chunk with metadata
            context_parts.append(
                f"[Source {i}]{page_info}{score_info}:\n{chunk.content}"
            )

        # Join all chunks
        context = "\n\n".join(context_parts)

        logger.debug(
            f"Built context from {len(chunks)} chunks "
            f"({len(context)} chars total)"
        )

        return context

    def format_prompt(
        self,
        query: str,
        context: str,
        conversation_history: List[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Format complete prompt for Claude API.

        Args:
            query: User's query
            context: Context string from retrieved chunks
            conversation_history: Previous conversation turns

        Returns:
            Dictionary with 'messages' and 'system' for Claude API
        """
        conversation_history = conversation_history or []

        # Build messages array
        messages = []

        # Add conversation history
        for turn in conversation_history:
            messages.append({"role": turn["role"], "content": turn["content"]})

        # Add current query with context
        user_message = self._build_user_message(query, context)
        messages.append({"role": "user", "content": user_message})

        logger.debug(
            f"Formatted prompt with {len(conversation_history)} history turns, "
            f"context: {len(context)} chars, query: {len(query)} chars"
        )

        return {"messages": messages, "system": self.system_prompt}

    def _build_user_message(self, query: str, context: str) -> str:
        """
        Build user message with context and query.

        Args:
            query: User's query
            context: Retrieved context

        Returns:
            Formatted user message
        """
        message = f"""Here is relevant information from the CV:

{context}

Based on the information above, please answer the following question:
{query}

Please provide a clear and concise answer based only on the information provided in the context."""

        return message

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        This is a rough approximation (chars / 4).
        For exact counting, use tiktoken library.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        return len(text) // 4

    def truncate_context(
        self, chunks: List[Chunk], max_chunks: int = None
    ) -> List[Chunk]:
        """
        Truncate chunks to fit within token limit.

        Args:
            chunks: List of chunks to potentially truncate
            max_chunks: Maximum number of chunks to keep

        Returns:
            Truncated list of chunks
        """
        if max_chunks is None:
            max_chunks = self.config.retrieval_top_k

        if len(chunks) <= max_chunks:
            return chunks

        logger.warning(
            f"Truncating context from {len(chunks)} to {max_chunks} chunks"
        )

        # Keep highest-scoring chunks
        sorted_chunks = sorted(
            chunks,
            key=lambda c: c.metadata.get("similarity_score", 0.0),
            reverse=True,
        )

        return sorted_chunks[:max_chunks]

    def add_source_citations(self, answer: str, chunks: List[Chunk]) -> str:
        """
        Add source citations to answer (optional enhancement).

        Args:
            answer: Generated answer
            chunks: Chunks used for context

        Returns:
            Answer with citations appended
        """
        if not chunks:
            return answer

        citations = []
        for i, chunk in enumerate(chunks, 1):
            page = chunk.metadata.get("page", "N/A")
            score = chunk.metadata.get("similarity_score", 0.0)
            citations.append(f"[{i}] Page {page} (relevance: {score:.2f})")

        citations_text = "\n\nSources:\n" + "\n".join(citations)

        return answer + citations_text
