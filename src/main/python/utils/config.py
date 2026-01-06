"""
Configuration management for the RAG system.

This module provides centralized configuration loading from:
1. YAML configuration files
2. Environment variables (.env)
3. Code defaults

Configuration precedence: Environment Variables > YAML > Code Defaults
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
from dotenv import load_dotenv


class Config:
    """
    Central configuration class for the RAG system.

    Loads configuration from:
    - config.yaml in src/main/resources/config/
    - .env file in project root
    - Code defaults as fallback
    """

    _instance: Optional["Config"] = None

    def __init__(self, config_path: Optional[str] = None, env_file: str = ".env"):
        """
        Initialize configuration.

        Args:
            config_path: Path to YAML config file. If None, uses default path.
            env_file: Path to .env file. Defaults to ".env" in project root.
        """
        # Load environment variables
        load_dotenv(dotenv_path=env_file)

        # Load YAML configuration
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "resources",
                "config",
                "config.yaml",
            )

        self._yaml_config: Dict[str, Any] = {}
        config_path = Path(config_path).resolve()

        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    self._yaml_config = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")
                self._yaml_config = {}
        else:
            print(f"Warning: Config file not found at {config_path}, using defaults")

    @classmethod
    def load(cls, config_path: Optional[str] = None, env_file: str = ".env") -> "Config":
        """
        Load configuration (singleton pattern).

        Args:
            config_path: Path to YAML config file
            env_file: Path to .env file

        Returns:
            Config instance
        """
        if cls._instance is None:
            cls._instance = cls(config_path=config_path, env_file=env_file)
        return cls._instance

    def _get_nested(self, keys: str, default: Any = None) -> Any:
        """
        Get nested value from YAML config using dot notation.

        Args:
            keys: Dot-separated key path (e.g., "rag.chunking.max_tokens")
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        value = self._yaml_config
        for key in keys.split("."):
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
            if value is None:
                return default
        return value

    # API Keys (always from environment)
    @property
    def anthropic_api_key(self) -> str:
        """Get Anthropic Claude API key from environment."""
        key = os.getenv("ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found in environment. "
                "Please set it in your .env file."
            )
        return key

    @property
    def pinecone_api_key(self) -> str:
        """Get Pinecone API key from environment."""
        key = os.getenv("PINECONE_API_KEY", "")
        if not key:
            raise ValueError(
                "PINECONE_API_KEY not found in environment. "
                "Please set it in your .env file."
            )
        return key

    @property
    def pinecone_environment(self) -> str:
        """Get Pinecone environment from environment or YAML."""
        return os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

    @property
    def pinecone_index_name(self) -> str:
        """Get Pinecone index name from environment or YAML."""
        return os.getenv("PINECONE_INDEX_NAME", "cv-rag-index")

    # Model Configuration
    @property
    def claude_model(self) -> str:
        """Get Claude model name."""
        return os.getenv(
            "CLAUDE_MODEL",
            self._get_nested("rag.generation.claude_model", "claude-3-5-sonnet-20240620"),
        )

    @property
    def embedding_model(self) -> str:
        """Get embedding model name."""
        return os.getenv(
            "EMBEDDING_MODEL",
            self._get_nested("embedding.model_name", "all-MiniLM-L6-v2"),
        )

    # RAG Configuration
    @property
    def chunk_size(self) -> int:
        """Get maximum chunk size in tokens."""
        return int(self._get_nested("rag.chunking.max_tokens", 512))

    @property
    def chunk_overlap(self) -> int:
        """Get chunk overlap in tokens."""
        return int(self._get_nested("rag.chunking.overlap_tokens", 50))

    @property
    def chunking_strategy(self) -> str:
        """Get chunking strategy (paragraph, sentence, fixed)."""
        return self._get_nested("rag.chunking.strategy", "paragraph")

    @property
    def retrieval_top_k(self) -> int:
        """Get number of top chunks to retrieve."""
        return int(self._get_nested("rag.retrieval.top_k", 5))

    @property
    def similarity_threshold(self) -> float:
        """Get similarity threshold for retrieval."""
        return float(self._get_nested("rag.retrieval.similarity_threshold", 0.7))

    @property
    def max_tokens(self) -> int:
        """Get maximum tokens for Claude generation."""
        return int(self._get_nested("rag.generation.max_tokens", 1024))

    @property
    def temperature(self) -> float:
        """Get temperature for Claude generation."""
        return float(self._get_nested("rag.generation.temperature", 0.7))

    @property
    def system_prompt(self) -> str:
        """Get system prompt for Claude."""
        default_prompt = (
            "You are an AI assistant answering questions about a person's CV/resume. "
            "Use the provided context to answer accurately. "
            "If information is not in the context, state that clearly."
        )
        return self._get_nested("rag.generation.system_prompt", default_prompt)

    # Embedding Configuration
    @property
    def embedding_device(self) -> str:
        """Get device for embedding model (cpu or cuda)."""
        return self._get_nested("embedding.device", "cpu")

    @property
    def embedding_batch_size(self) -> int:
        """Get batch size for embedding generation."""
        return int(self._get_nested("embedding.batch_size", 32))

    @property
    def normalize_embeddings(self) -> bool:
        """Whether to normalize embeddings."""
        return bool(self._get_nested("embedding.normalize_embeddings", True))

    # Vector Store Configuration
    @property
    def vector_metric(self) -> str:
        """Get vector similarity metric (cosine, euclidean, dotproduct)."""
        return self._get_nested("vector_store.metric", "cosine")

    @property
    def vector_dimension(self) -> int:
        """Get vector dimension (384 for all-MiniLM-L6-v2)."""
        return int(self._get_nested("vector_store.dimension", 384))

    @property
    def vector_namespace(self) -> str:
        """Get Pinecone namespace for organizing data."""
        return self._get_nested("vector_store.namespace", "cv")

    # Conversation Configuration
    @property
    def max_conversation_turns(self) -> int:
        """Get maximum conversation history turns to keep."""
        return int(
            os.getenv(
                "MAX_CONVERSATION_TURNS",
                self._get_nested("conversation.max_history_turns", 5),
            )
        )

    @property
    def conversation_persist_to_disk(self) -> bool:
        """Whether to persist conversation history to disk."""
        return bool(self._get_nested("conversation.persist_to_disk", True))

    @property
    def conversation_save_path(self) -> str:
        """Get path for saving conversation history."""
        return self._get_nested(
            "conversation.save_path", "data/temp/conversation_history.json"
        )

    # Logging
    @property
    def log_level(self) -> str:
        """Get logging level."""
        return os.getenv("LOG_LEVEL", self._get_nested("logging.level", "INFO"))

    def validate(self) -> bool:
        """
        Validate required configuration settings.

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If required settings are missing or invalid
        """
        # Check required API keys
        _ = self.anthropic_api_key  # Will raise if missing
        _ = self.pinecone_api_key  # Will raise if missing

        # Validate numerical settings
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.retrieval_top_k <= 0:
            raise ValueError(f"retrieval_top_k must be positive, got {self.retrieval_top_k}")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be between 0 and 2, got {self.temperature}")

        return True
