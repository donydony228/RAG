"""
Data models and schemas for the RAG system.

This module defines the core data structures used throughout the application:
- Document: Represents a source document (e.g., CV PDF)
- Chunk: Represents a text chunk extracted from a document
- QueryResult: Represents the result of a RAG query
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime


@dataclass
class Document:
    """Represents a source document before chunking."""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        """Validate document after initialization."""
        if not self.content or not self.content.strip():
            raise ValueError("Document content cannot be empty")


@dataclass
class Chunk:
    """Represents a text chunk for embedding and retrieval."""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    embedding: Optional[List[float]] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        """Validate chunk after initialization."""
        if not self.content or not self.content.strip():
            raise ValueError("Chunk content cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """Create chunk from dictionary."""
        return cls(
            content=data["content"],
            metadata=data.get("metadata", {}),
            chunk_id=data.get("chunk_id", str(uuid.uuid4())),
            embedding=data.get("embedding"),
            created_at=data.get("created_at", datetime.now().isoformat()),
        )


@dataclass
class QueryResult:
    """Represents the result of a RAG query."""

    answer: str
    sources: List[Chunk] = field(default_factory=list)
    tokens_used: int = 0
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "answer": self.answer,
            "sources": [chunk.to_dict() for chunk in self.sources],
            "tokens_used": self.tokens_used,
            "retrieval_time_ms": self.retrieval_time_ms,
            "generation_time_ms": self.generation_time_ms,
            "session_id": self.session_id,
            "metadata": self.metadata,
        }


@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert turn to dictionary for serialization."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "tokens": self.tokens,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationTurn":
        """Create turn from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            tokens=data.get("tokens", 0),
        )
