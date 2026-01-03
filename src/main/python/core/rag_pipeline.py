"""
Main RAG pipeline orchestrator.

This module coordinates all RAG components to provide a unified interface
for document ingestion and question answering.
"""

from typing import Optional
import time
from tqdm import tqdm
from src.main.python.models.schemas import QueryResult
from src.main.python.services.pdf_service import PDFService
from src.main.python.services.embedding_service import EmbeddingService
from src.main.python.services.vector_store_service import VectorStoreService
from src.main.python.services.llm_service import LLMService
from src.main.python.core.document_processor import DocumentProcessor
from src.main.python.core.retriever import Retriever
from src.main.python.core.context_builder import ContextBuilder
from src.main.python.utils.conversation_manager import ConversationManager
from src.main.python.utils.config import Config
from src.main.python.utils.logger import get_logger


logger = get_logger(__name__)


class RAGPipeline:
    """
    Main RAG pipeline orchestrator.

    Coordinates all services and components for:
    - Document ingestion (PDF → chunks → embeddings → vector store)
    - Query processing (query → retrieval → context → LLM → answer)
    """

    def __init__(self, config: Config = None):
        """
        Initialize RAG pipeline with all services and components.

        Args:
            config: Configuration object (loads default if None)
        """
        # Load configuration
        self.config = config or Config.load()
        logger.info("Initializing RAG Pipeline...")

        try:
            # Validate configuration
            self.config.validate()

            # Initialize services
            logger.info("Initializing services...")
            self.pdf_service = PDFService()
            self.embedding_service = EmbeddingService(self.config)
            self.vector_store = VectorStoreService(self.config)
            self.llm_service = LLMService(self.config)

            # Initialize core components
            logger.info("Initializing core components...")
            self.document_processor = DocumentProcessor(
                self.config, self.pdf_service
            )
            self.retriever = Retriever(
                self.config, self.vector_store, self.embedding_service
            )
            self.context_builder = ContextBuilder(self.config)
            self.conversation_manager = ConversationManager(self.config)

            logger.info("✅ RAG Pipeline initialized successfully")

        except Exception as e:
            error_msg = f"Failed to initialize RAG Pipeline: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def ingest_document(self, pdf_path: str, show_progress: bool = True) -> dict:
        """
        Ingest a PDF document into the RAG system.

        Workflow:
        1. Process PDF → chunks
        2. Generate embeddings for chunks
        3. Upsert chunks + embeddings to Pinecone

        Args:
            pdf_path: Path to PDF file
            show_progress: Whether to show progress bars

        Returns:
            Dictionary with ingestion statistics

        Raises:
            Exception: If ingestion fails
        """
        logger.info(f"Starting document ingestion: {pdf_path}")
        start_time = time.time()

        try:
            # Step 1: Process PDF into chunks
            logger.info("Step 1/3: Processing PDF...")
            chunks = self.document_processor.process_pdf(pdf_path)
            logger.info(f"Created {len(chunks)} chunks")

            if not chunks:
                raise ValueError("No chunks generated from PDF")

            # Step 2: Generate embeddings
            logger.info("Step 2/3: Generating embeddings...")
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = self.embedding_service.embed_batch(
                chunk_texts, show_progress=show_progress
            )
            logger.info(f"Generated {len(embeddings)} embeddings")

            # Attach embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding

            # Step 3: Upsert to Pinecone
            logger.info("Step 3/3: Uploading to vector store...")
            if show_progress:
                chunks_iter = tqdm(
                    [chunks], desc="Upserting to Pinecone", total=1
                )
            else:
                chunks_iter = [chunks]

            for chunk_batch in chunks_iter:
                self.vector_store.upsert_chunks(chunk_batch, embeddings)

            elapsed = time.time() - start_time

            stats = {
                "pdf_path": pdf_path,
                "chunks_created": len(chunks),
                "embeddings_generated": len(embeddings),
                "time_seconds": round(elapsed, 2),
                "success": True,
            }

            logger.info(
                f"✅ Document ingestion complete in {elapsed:.2f}s: "
                f"{len(chunks)} chunks indexed"
            )

            return stats

        except Exception as e:
            error_msg = f"Document ingestion failed: {e}"
            logger.error(error_msg)
            raise Exception(error_msg) from e

    def query(
        self,
        question: str,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> QueryResult:
        """
        Answer a question using RAG.

        Workflow:
        1. Retrieve relevant chunks from vector store
        2. Build context from chunks + conversation history
        3. Generate answer using Claude
        4. Update conversation history

        Args:
            question: User's question
            session_id: Conversation session ID (creates new if None)
            top_k: Number of chunks to retrieve

        Returns:
            QueryResult with answer, sources, and metadata

        Raises:
            ValueError: If question is empty
            Exception: If query processing fails
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        logger.info(f"Processing query: {question[:100]}...")
        retrieval_start = time.time()

        try:
            # Create session if needed
            if session_id is None:
                session_id = self.conversation_manager.create_session()
                logger.info(f"Created new session: {session_id}")

            # Step 1: Retrieve relevant chunks
            logger.info("Step 1/3: Retrieving relevant context...")
            chunks = self.retriever.retrieve(question, top_k=top_k)
            retrieval_time_ms = (time.time() - retrieval_start) * 1000
            logger.info(f"Retrieved {len(chunks)} relevant chunks")

            if not chunks:
                logger.warning("No relevant chunks found, providing generic response")
                answer = (
                    "I couldn't find relevant information in the CV to answer "
                    "that question. Please try rephrasing or ask a different question."
                )
                return QueryResult(
                    answer=answer,
                    sources=[],
                    retrieval_time_ms=retrieval_time_ms,
                    session_id=session_id,
                )

            # Step 2: Build context
            logger.info("Step 2/3: Building context...")
            context = self.retriever.format_context(chunks)
            conversation_history = self.conversation_manager.get_history(
                session_id
            )

            # Step 3: Generate answer
            logger.info("Step 3/3: Generating answer with Claude...")
            generation_start = time.time()

            answer, tokens_used = self.llm_service.generate_with_history(
                query=question,
                context=context,
                conversation_history=conversation_history,
            )

            generation_time_ms = (time.time() - generation_start) * 1000

            # Step 4: Update conversation history
            self.conversation_manager.add_turn(
                session_id=session_id,
                question=question,
                answer=answer,
                tokens=tokens_used,
            )

            # Create result
            result = QueryResult(
                answer=answer,
                sources=chunks,
                tokens_used=tokens_used,
                retrieval_time_ms=retrieval_time_ms,
                generation_time_ms=generation_time_ms,
                session_id=session_id,
                metadata={
                    "num_chunks_retrieved": len(chunks),
                    "conversation_turns": len(conversation_history) // 2,
                },
            )

            logger.info(
                f"✅ Query complete: {tokens_used} tokens, "
                f"{retrieval_time_ms:.0f}ms retrieval, "
                f"{generation_time_ms:.0f}ms generation"
            )

            return result

        except Exception as e:
            error_msg = f"Query processing failed: {e}"
            logger.error(error_msg)
            raise Exception(error_msg) from e

    def clear_index(self) -> None:
        """
        Clear all vectors from the Pinecone index.

        WARNING: This deletes all ingested documents!
        """
        logger.warning("Clearing Pinecone index...")
        try:
            self.vector_store.clear_namespace()
            logger.info("✅ Index cleared successfully")
        except Exception as e:
            error_msg = f"Failed to clear index: {e}"
            logger.error(error_msg)
            raise Exception(error_msg) from e

    def get_index_stats(self) -> dict:
        """
        Get statistics about the vector index.

        Returns:
            Dictionary with index statistics
        """
        try:
            stats = self.vector_store.get_index_stats()
            logger.info(f"Index stats: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {"error": str(e)}

    def create_session(self) -> str:
        """
        Create a new conversation session.

        Returns:
            Session ID
        """
        return self.conversation_manager.create_session()

    def get_session_info(self, session_id: str) -> dict:
        """
        Get information about a conversation session.

        Args:
            session_id: Session identifier

        Returns:
            Session information dictionary
        """
        return self.conversation_manager.get_session_info(session_id)

    def clear_session(self, session_id: str) -> None:
        """
        Clear conversation history for a session.

        Args:
            session_id: Session identifier
        """
        self.conversation_manager.clear_session(session_id)
        logger.info(f"Cleared session: {session_id}")
