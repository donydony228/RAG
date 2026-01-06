"""
Retrieval logic for the RAG system.

This module handles retrieving relevant chunks from the vector store
based on semantic similarity to a query.
"""

from typing import List, Tuple, Optional
import time
from src.main.python.models.schemas import Chunk
from src.main.python.services.vector_store_service import VectorStoreService
from src.main.python.services.embedding_service import EmbeddingService
from src.main.python.utils.config import Config
from src.main.python.utils.logger import get_logger


logger = get_logger(__name__)


class Retriever:
    """Retrieves relevant chunks from vector store based on query."""

    def __init__(
        self,
        config: Config,
        vector_store_service: VectorStoreService,
        embedding_service: EmbeddingService,
    ):
        """
        Initialize retriever.

        Args:
            config: Configuration object
            vector_store_service: Vector store service instance
            embedding_service: Embedding service instance
        """
        self.config = config
        self.vector_store = vector_store_service
        self.embedding_service = embedding_service
        self.top_k = config.retrieval_top_k
        self.similarity_threshold = config.similarity_threshold

        logger.info(
            f"Initialized Retriever with top_k={self.top_k}, "
            f"threshold={self.similarity_threshold}"
        )

    def retrieve(
        self, query: str, top_k: Optional[int] = None, filter_dict: Optional[dict] = None
    ) -> List[Chunk]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: User query text
            top_k: Number of chunks to retrieve (uses config default if None)
            filter_dict: Optional metadata filter for vector search

        Returns:
            List of Chunk objects, ordered by relevance

        Raises:
            ValueError: If query is empty
            Exception: If retrieval fails
        """
        if not query or not query.strip():
            raise ValueError("Cannot retrieve with empty query")

        top_k = top_k or self.top_k
        logger.info(f"Retrieving top {top_k} chunks for query: {query[:100]}...")

        start_time = time.time()

        try:
            # Step 1: Generate query embedding
            logger.debug("Generating query embedding...")
            query_embedding = self.embedding_service.embed_text(query)

            # Step 2: Search vector store
            logger.debug(f"Searching vector store (top_k={top_k})...")
            results = self.vector_store.search(
                query_embedding=query_embedding, top_k=top_k, filter_dict=filter_dict
            )

            # Step 3: Convert results to Chunk objects
            chunks = []
            for result in results:
                # Log actual scores for debugging
                logger.info(f"Result score: {result['score']:.4f} (threshold: {self.similarity_threshold})")

                # Filter by similarity threshold
                if result["score"] < self.similarity_threshold:
                    logger.debug(
                        f"Skipping result with score {result['score']:.3f} "
                        f"(below threshold {self.similarity_threshold})"
                    )
                    continue

                # Create Chunk from metadata
                chunk = Chunk(
                    content=result["metadata"].get(
                        "full_content", result["metadata"].get("content", "")
                    ),
                    metadata={
                        **result["metadata"],
                        "similarity_score": result["score"],
                        "chunk_id": result["id"],
                    },
                    chunk_id=result["id"],
                )
                chunks.append(chunk)

            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(
                f"Retrieved {len(chunks)} chunks in {elapsed_ms:.2f}ms "
                f"(filtered from {len(results)} results)"
            )

            return chunks

        except Exception as e:
            error_msg = f"Failed to retrieve chunks: {e}"
            logger.error(error_msg)
            raise Exception(error_msg) from e

    def retrieve_with_scores(
        self, query: str, top_k: Optional[int] = None, filter_dict: Optional[dict] = None
    ) -> List[Tuple[Chunk, float]]:
        """
        Retrieve relevant chunks with their similarity scores.

        Args:
            query: User query text
            top_k: Number of chunks to retrieve
            filter_dict: Optional metadata filter

        Returns:
            List of tuples (Chunk, similarity_score)

        Raises:
            ValueError: If query is empty
            Exception: If retrieval fails
        """
        chunks = self.retrieve(query=query, top_k=top_k, filter_dict=filter_dict)

        # Extract scores from metadata
        chunks_with_scores = [
            (chunk, chunk.metadata.get("similarity_score", 0.0)) for chunk in chunks
        ]

        return chunks_with_scores

    def rerank(self, chunks: List[Chunk], query: str) -> List[Chunk]:
        """
        Rerank retrieved chunks (placeholder for future enhancement).

        This is a placeholder for more sophisticated reranking logic
        (e.g., using a cross-encoder model).

        Args:
            chunks: List of chunks to rerank
            query: Original query

        Returns:
            Reranked list of chunks
        """
        # For now, just return chunks as-is (already sorted by similarity)
        # In future, could implement cross-encoder reranking
        logger.debug("Reranking not implemented, returning original order")
        return chunks

    def format_context(self, chunks: List[Chunk]) -> str:
        """
        Format retrieved chunks into a context string.

        Args:
            chunks: List of chunks to format

        Returns:
            Formatted context string
        """
        if not chunks:
            return "No relevant information found in CV."

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            # Include metadata about source
            source_info = ""
            if "page" in chunk.metadata:
                source_info = f" (Page {chunk.metadata['page']})"
            if "similarity_score" in chunk.metadata:
                source_info += f" [Relevance: {chunk.metadata['similarity_score']:.2f}]"

            context_parts.append(f"[Source {i}]{source_info}:\n{chunk.content}")

        formatted_context = "\n\n".join(context_parts)
        logger.debug(f"Formatted context: {len(formatted_context)} chars")

        return formatted_context
