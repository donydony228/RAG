"""
Conversation history management for the RAG system.

This module manages multi-turn conversation state, including
history storage, retrieval, and optional persistence to disk.
"""

from typing import List, Dict, Optional
from pathlib import Path
import json
import uuid
from datetime import datetime
from src.main.python.models.schemas import ConversationTurn
from src.main.python.utils.config import Config
from src.main.python.utils.logger import get_logger


logger = get_logger(__name__)


class ConversationManager:
    """Manages conversation history for multi-turn interactions."""

    def __init__(self, config: Config):
        """
        Initialize conversation manager.

        Args:
            config: Configuration object
        """
        self.config = config
        self.max_turns = config.max_conversation_turns
        self.persist_to_disk = config.conversation_persist_to_disk
        self.save_path = Path(config.conversation_save_path)

        # In-memory storage: session_id -> list of turns
        self.sessions: Dict[str, List[ConversationTurn]] = {}

        logger.info(
            f"Initialized ConversationManager "
            f"(max_turns={self.max_turns}, persist={self.persist_to_disk})"
        )

        # Load from disk if enabled
        if self.persist_to_disk:
            self._load_from_disk()

    def create_session(self) -> str:
        """
        Create a new conversation session.

        Returns:
            Session ID (UUID string)
        """
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = []
        logger.info(f"Created new session: {session_id}")
        return session_id

    def add_turn(
        self, session_id: str, question: str, answer: str, tokens: int = 0
    ) -> None:
        """
        Add a conversation turn to a session.

        Args:
            session_id: Session identifier
            question: User's question
            answer: Assistant's answer
            tokens: Number of tokens used in generation

        Raises:
            ValueError: If session_id doesn't exist
        """
        if session_id not in self.sessions:
            logger.warning(f"Session {session_id} not found, creating new")
            self.sessions[session_id] = []

        # Add user turn
        user_turn = ConversationTurn(role="user", content=question, tokens=0)
        self.sessions[session_id].append(user_turn)

        # Add assistant turn
        assistant_turn = ConversationTurn(
            role="assistant", content=answer, tokens=tokens
        )
        self.sessions[session_id].append(assistant_turn)

        # Limit history length
        if len(self.sessions[session_id]) > self.max_turns * 2:
            # Remove oldest pair (user + assistant)
            self.sessions[session_id] = self.sessions[session_id][-self.max_turns * 2 :]
            logger.debug(
                f"Trimmed session {session_id} to {len(self.sessions[session_id])} turns"
            )

        logger.info(
            f"Added turn to session {session_id} "
            f"(total turns: {len(self.sessions[session_id]) // 2})"
        )

        # Persist if enabled
        if self.persist_to_disk:
            self._save_to_disk()

    def get_history(
        self, session_id: str, max_turns: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        Get conversation history for a session.

        Args:
            session_id: Session identifier
            max_turns: Maximum number of turns to return (uses config default if None)

        Returns:
            List of conversation turns as dicts with 'role' and 'content'
        """
        if session_id not in self.sessions:
            logger.warning(f"Session {session_id} not found, returning empty history")
            return []

        turns = self.sessions[session_id]
        max_turns = max_turns or self.max_turns

        # Limit to last N pairs of turns
        if max_turns:
            turns = turns[-max_turns * 2 :]

        # Convert to dict format for Claude API
        history = [{"role": turn.role, "content": turn.content} for turn in turns]

        logger.debug(
            f"Retrieved {len(history)} conversation turns for session {session_id}"
        )

        return history

    def get_session_info(self, session_id: str) -> Dict:
        """
        Get information about a session.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with session metadata
        """
        if session_id not in self.sessions:
            return {"session_id": session_id, "exists": False}

        turns = self.sessions[session_id]
        num_turns = len(turns) // 2  # Each conversation turn is user + assistant

        return {
            "session_id": session_id,
            "exists": True,
            "num_turns": num_turns,
            "total_messages": len(turns),
            "created_at": turns[0].timestamp if turns else None,
            "last_updated": turns[-1].timestamp if turns else None,
        }

    def clear_session(self, session_id: str) -> None:
        """
        Clear conversation history for a session.

        Args:
            session_id: Session identifier
        """
        if session_id in self.sessions:
            self.sessions[session_id] = []
            logger.info(f"Cleared session {session_id}")

            if self.persist_to_disk:
                self._save_to_disk()
        else:
            logger.warning(f"Cannot clear non-existent session {session_id}")

    def delete_session(self, session_id: str) -> None:
        """
        Delete a session completely.

        Args:
            session_id: Session identifier
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Deleted session {session_id}")

            if self.persist_to_disk:
                self._save_to_disk()
        else:
            logger.warning(f"Cannot delete non-existent session {session_id}")

    def list_sessions(self) -> List[str]:
        """
        List all active session IDs.

        Returns:
            List of session IDs
        """
        return list(self.sessions.keys())

    def _save_to_disk(self) -> None:
        """Save all sessions to disk."""
        try:
            # Ensure directory exists
            self.save_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert sessions to serializable format
            data = {}
            for session_id, turns in self.sessions.items():
                data[session_id] = [turn.to_dict() for turn in turns]

            # Write to file
            with open(self.save_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved {len(data)} sessions to {self.save_path}")

        except Exception as e:
            logger.error(f"Failed to save conversations to disk: {e}")

    def _load_from_disk(self) -> None:
        """Load sessions from disk."""
        if not self.save_path.exists():
            logger.debug(f"No saved conversations found at {self.save_path}")
            return

        try:
            with open(self.save_path, "r") as f:
                data = json.load(f)

            # Convert back to ConversationTurn objects
            for session_id, turns_data in data.items():
                self.sessions[session_id] = [
                    ConversationTurn.from_dict(turn) for turn in turns_data
                ]

            logger.info(
                f"Loaded {len(data)} sessions from {self.save_path} "
                f"({sum(len(t) for t in self.sessions.values())} total turns)"
            )

        except Exception as e:
            logger.error(f"Failed to load conversations from disk: {e}")
            self.sessions = {}
