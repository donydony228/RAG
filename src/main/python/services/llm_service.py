"""
LLM service for integrating with Anthropic Claude API.

This module provides functionality to generate responses using Claude
with conversation history support and retry logic.
"""

from typing import List, Dict, Any, Tuple
from anthropic import Anthropic
from tenacity import retry, stop_after_attempt, wait_exponential
from src.main.python.utils.logger import get_logger
from src.main.python.utils.config import Config


logger = get_logger(__name__)


class LLMService:
    """Service for generating text using Claude API."""

    def __init__(self, config: Config):
        """
        Initialize LLM service with Claude API.

        Args:
            config: Configuration object containing API credentials

        Raises:
            Exception: If API initialization fails
        """
        self.config = config
        self.model = config.claude_model
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature
        self.system_prompt = config.system_prompt

        logger.info(f"Initializing LLM service with model: {self.model}")

        try:
            self.client = Anthropic(api_key=config.anthropic_api_key)
            logger.info("Successfully initialized Claude API client")
        except Exception as e:
            error_msg = f"Failed to initialize Claude API: {e}"
            logger.error(error_msg)
            raise Exception(error_msg) from e

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = None,
        temperature: float = None,
        system: str = None,
    ) -> Tuple[str, int]:
        """
        Generate a response using Claude API.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate (uses config default if None)
            temperature: Sampling temperature (uses config default if None)
            system: System prompt (uses config default if None)

        Returns:
            Tuple of (generated_text, tokens_used)

        Raises:
            Exception: If API call fails after retries
        """
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature
        system = system or self.system_prompt

        logger.debug(
            f"Generating response with {len(messages)} messages, "
            f"max_tokens={max_tokens}, temperature={temperature}"
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=messages,
            )

            # Extract text from response
            generated_text = response.content[0].text

            # Get token usage
            tokens_used = response.usage.input_tokens + response.usage.output_tokens

            logger.info(
                f"Generated response: {len(generated_text)} chars, "
                f"{tokens_used} tokens used "
                f"(input: {response.usage.input_tokens}, "
                f"output: {response.usage.output_tokens})"
            )

            return generated_text, tokens_used

        except Exception as e:
            error_msg = f"Failed to generate response from Claude: {e}"
            logger.error(error_msg)
            raise Exception(error_msg) from e

    def generate_with_history(
        self,
        query: str,
        context: str,
        conversation_history: List[Dict[str, str]] = None,
        max_tokens: int = None,
        temperature: float = None,
    ) -> Tuple[str, int]:
        """
        Generate a response with conversation history.

        Args:
            query: User's current question
            context: Retrieved context from vector store
            conversation_history: Previous conversation turns
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Tuple of (generated_response, tokens_used)

        Raises:
            Exception: If generation fails
        """
        conversation_history = conversation_history or []

        # Build messages array
        messages = []

        # Add conversation history
        for turn in conversation_history:
            messages.append({"role": turn["role"], "content": turn["content"]})

        # Add current query with context
        user_message = f"""Context from CV:
{context}

Question: {query}

Please answer the question based on the context provided above."""

        messages.append({"role": "user", "content": user_message})

        logger.debug(
            f"Generating with {len(conversation_history)} history turns, "
            f"context length: {len(context)} chars"
        )

        return self.generate(
            messages=messages, max_tokens=max_tokens, temperature=temperature
        )

    def count_message_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Estimate token count for messages.

        Note: This is an approximation using character count / 4.
        For exact counting, use tiktoken library.

        Args:
            messages: List of message dictionaries

        Returns:
            Estimated token count
        """
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        estimated_tokens = total_chars // 4  # Rough approximation

        logger.debug(f"Estimated {estimated_tokens} tokens for {len(messages)} messages")

        return estimated_tokens
