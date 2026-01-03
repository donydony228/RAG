"""
Embedding generation service using Sentence Transformers.

This module provides local embedding generation without API calls.
Uses the all-MiniLM-L6-v2 model by default (384-dimensional vectors).
"""

from typing import List, Union
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
from src.main.python.utils.logger import get_logger
from src.main.python.utils.config import Config


logger = get_logger(__name__)


class EmbeddingService:
    """Service for generating embeddings using Sentence Transformers."""

    def __init__(self, config: Config):
        """
        Initialize embedding service.

        Args:
            config: Configuration object

        The model will be downloaded on first use and cached locally.
        """
        self.config = config
        self.model_name = config.embedding_model
        self.device = config.embedding_device
        self.batch_size = config.embedding_batch_size
        self.normalize = config.normalize_embeddings

        logger.info(
            f"Initializing EmbeddingService with model: {self.model_name}, "
            f"device: {self.device}"
        )

        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(
                f"Successfully loaded model. Embedding dimension: {self.dimension}"
            )
        except Exception as e:
            error_msg = f"Failed to load embedding model {self.model_name}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding vector

        Raises:
            ValueError: If text is empty
            Exception: If embedding generation fails
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )

            # Convert numpy array to list
            embedding_list = embedding.tolist()

            logger.debug(f"Generated embedding of dimension {len(embedding_list)}")
            return embedding_list

        except Exception as e:
            error_msg = f"Failed to generate embedding: {e}"
            logger.error(error_msg)
            raise Exception(error_msg) from e

    def embed_batch(
        self, texts: List[str], show_progress: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch processing).

        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar

        Returns:
            List of embedding vectors

        Raises:
            ValueError: If texts list is empty
            Exception: If embedding generation fails
        """
        if not texts:
            raise ValueError("Cannot embed empty list of texts")

        # Filter out empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        if len(valid_texts) < len(texts):
            logger.warning(
                f"Filtered out {len(texts) - len(valid_texts)} empty texts"
            )

        if not valid_texts:
            raise ValueError("All texts are empty after filtering")

        try:
            logger.info(f"Generating embeddings for {len(valid_texts)} texts")

            embeddings = self.model.encode(
                valid_texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
                show_progress_bar=show_progress,
            )

            # Convert numpy array to list of lists
            embeddings_list = embeddings.tolist()

            logger.info(
                f"Successfully generated {len(embeddings_list)} embeddings "
                f"of dimension {len(embeddings_list[0])}"
            )

            return embeddings_list

        except Exception as e:
            error_msg = f"Failed to generate batch embeddings: {e}"
            logger.error(error_msg)
            raise Exception(error_msg) from e

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.

        Returns:
            Embedding dimension (384 for all-MiniLM-L6-v2)
        """
        return self.dimension

    def compute_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0 to 1)

        Raises:
            ValueError: If embeddings have different dimensions
        """
        if len(embedding1) != len(embedding2):
            raise ValueError(
                f"Embeddings must have same dimension. "
                f"Got {len(embedding1)} and {len(embedding2)}"
            )

        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Compute cosine similarity
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

        return float(similarity)
