"""
Pinecone vector store service for storing and retrieving embeddings.

This module provides integration with Pinecone for vector storage and similarity search.
"""

from typing import List, Dict, Any, Optional, Tuple
from pinecone import Pinecone, ServerlessSpec
from tenacity import retry, stop_after_attempt, wait_exponential
from src.main.python.utils.logger import get_logger
from src.main.python.utils.config import Config
from src.main.python.models.schemas import Chunk


logger = get_logger(__name__)


class VectorStoreService:
    """Service for managing Pinecone vector database operations."""

    def __init__(self, config: Config):
        """
        Initialize Pinecone vector store service.

        Args:
            config: Configuration object containing Pinecone credentials

        Raises:
            Exception: If Pinecone initialization fails
        """
        self.config = config
        self.index_name = config.pinecone_index_name
        self.dimension = config.vector_dimension
        self.metric = config.vector_metric
        self.namespace = config.vector_namespace

        logger.info(f"Initializing Pinecone with index: {self.index_name}")

        try:
            # Initialize Pinecone client
            self.pc = Pinecone(api_key=config.pinecone_api_key)

            # Check if index exists, create if not
            existing_indexes = [index.name for index in self.pc.list_indexes()]

            if self.index_name not in existing_indexes:
                logger.info(f"Index {self.index_name} not found, creating...")
                self._create_index()
            else:
                logger.info(f"Index {self.index_name} already exists")

            # Connect to index
            self.index = self.pc.Index(self.index_name)
            logger.info("Successfully connected to Pinecone index")

        except Exception as e:
            error_msg = f"Failed to initialize Pinecone: {e}"
            logger.error(error_msg)
            raise Exception(error_msg) from e

    def _create_index(self) -> None:
        """
        Create a new Pinecone index.

        Raises:
            Exception: If index creation fails
        """
        try:
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(
                    cloud="aws", region=self.config.pinecone_environment
                ),
            )
            logger.info(
                f"Created index {self.index_name} with dimension={self.dimension}, "
                f"metric={self.metric}"
            )
        except Exception as e:
            error_msg = f"Failed to create index: {e}"
            logger.error(error_msg)
            raise Exception(error_msg) from e

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def upsert_chunks(
        self, chunks: List[Chunk], embeddings: List[List[float]]
    ) -> Dict[str, Any]:
        """
        Upsert chunks with their embeddings to Pinecone.

        Args:
            chunks: List of Chunk objects to store
            embeddings: Corresponding embedding vectors

        Returns:
            Upsert response from Pinecone

        Raises:
            ValueError: If chunks and embeddings lengths don't match
            Exception: If upsert fails after retries
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "must have same length"
            )

        logger.info(f"Upserting {len(chunks)} chunks to Pinecone")

        try:
            # Prepare vectors for upsert
            vectors = []
            for chunk, embedding in zip(chunks, embeddings):
                vector = {
                    "id": chunk.chunk_id,
                    "values": embedding,
                    "metadata": {
                        "content": chunk.content[:1000],  # Limit metadata size
                        "full_content": chunk.content,  # Store full content if needed
                        **chunk.metadata,  # Include original metadata
                    },
                }
                vectors.append(vector)

            # Upsert in batches of 100 (Pinecone limit)
            batch_size = 100
            total_upserted = 0

            for i in range(0, len(vectors), batch_size):
                batch = vectors[i : i + batch_size]
                response = self.index.upsert(vectors=batch, namespace=self.namespace)
                total_upserted += response.upserted_count
                logger.debug(
                    f"Upserted batch {i // batch_size + 1}: "
                    f"{response.upserted_count} vectors"
                )

            logger.info(f"Successfully upserted {total_upserted} vectors")
            return {"upserted_count": total_upserted}

        except Exception as e:
            error_msg = f"Failed to upsert chunks to Pinecone: {e}"
            logger.error(error_msg)
            raise Exception(error_msg) from e

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def search(
        self, query_embedding: List[float], top_k: int = 5, filter_dict: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in Pinecone.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            filter_dict: Optional metadata filter

        Returns:
            List of matching results with scores and metadata

        Raises:
            Exception: If search fails after retries
        """
        try:
            logger.debug(f"Searching Pinecone for top {top_k} similar vectors")

            response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                namespace=self.namespace,
                filter=filter_dict,
            )

            results = []
            for match in response.matches:
                results.append(
                    {
                        "id": match.id,
                        "score": match.score,
                        "metadata": match.metadata,
                    }
                )

            logger.info(f"Found {len(results)} matching vectors")
            return results

        except Exception as e:
            error_msg = f"Failed to search Pinecone: {e}"
            logger.error(error_msg)
            raise Exception(error_msg) from e

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the Pinecone index.

        Returns:
            Dictionary with index statistics

        Raises:
            Exception: If stats retrieval fails
        """
        try:
            stats = self.index.describe_index_stats()
            logger.info(f"Index stats: {stats}")
            return stats
        except Exception as e:
            error_msg = f"Failed to get index stats: {e}"
            logger.error(error_msg)
            raise Exception(error_msg) from e

    def clear_namespace(self, namespace: Optional[str] = None) -> None:
        """
        Clear all vectors in a namespace.

        Args:
            namespace: Namespace to clear (uses default if None)

        Raises:
            Exception: If clearing fails
        """
        namespace = namespace or self.namespace

        try:
            logger.warning(f"Clearing namespace: {namespace}")
            self.index.delete(delete_all=True, namespace=namespace)
            logger.info(f"Successfully cleared namespace: {namespace}")
        except Exception as e:
            error_msg = f"Failed to clear namespace {namespace}: {e}"
            logger.error(error_msg)
            raise Exception(error_msg) from e

    def delete_index(self) -> None:
        """
        Delete the entire Pinecone index.

        WARNING: This is destructive and cannot be undone!

        Raises:
            Exception: If deletion fails
        """
        try:
            logger.warning(f"Deleting index: {self.index_name}")
            self.pc.delete_index(self.index_name)
            logger.info(f"Successfully deleted index: {self.index_name}")
        except Exception as e:
            error_msg = f"Failed to delete index {self.index_name}: {e}"
            logger.error(error_msg)
            raise Exception(error_msg) from e
