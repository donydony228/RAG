"""
Command-line interface for the RAG system.

Provides commands for:
- ingest: Ingest a PDF document
- query: Ask a single question
- interactive: Multi-turn conversation mode
- clear: Clear the vector index
- stats: Show index statistics
"""

import argparse
import sys
from pathlib import Path
from src.main.python.core.rag_pipeline import RAGPipeline
from src.main.python.utils.logger import get_logger, setup_logging


# Setup logging
setup_logging()
logger = get_logger(__name__)


def ingest_command(args):
    """
    Handle the ingest command.

    Args:
        args: Parsed command-line arguments
    """
    try:
        print(f"\n🔄 Ingesting document: {args.pdf}")
        print("This may take a few moments...\n")

        # Initialize pipeline
        pipeline = RAGPipeline()

        # Ingest document
        stats = pipeline.ingest_document(args.pdf, show_progress=True)

        # Display results
        print(f"\n✅ Ingestion complete!")
        print(f"   • Chunks created: {stats['chunks_created']}")
        print(f"   • Embeddings generated: {stats['embeddings_generated']}")
        print(f"   • Time elapsed: {stats['time_seconds']}s")
        print(
            f"\nYou can now query your CV using: python -m src.main.python.main query \"<your question>\"\n"
        )

    except FileNotFoundError as e:
        print(f"\n❌ Error: PDF file not found: {args.pdf}")
        print(f"   Please check the file path and try again.\n")
        logger.error(e)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during ingestion: {e}\n")
        logger.error(e)
        sys.exit(1)


def query_command(args):
    """
    Handle the query command (single-shot Q&A).

    Args:
        args: Parsed command-line arguments
    """
    try:
        print(f"\n🔍 Question: {args.question}\n")
        print("Searching CV and generating answer...\n")

        # Initialize pipeline
        pipeline = RAGPipeline()

        # Query
        result = pipeline.query(args.question)

        # Display answer
        print("=" * 70)
        print("ANSWER:")
        print("=" * 70)
        print(f"\n{result.answer}\n")
        print("=" * 70)

        # Display metadata
        if args.verbose:
            print(f"\nMetadata:")
            print(f"  • Tokens used: {result.tokens_used}")
            print(f"  • Retrieval time: {result.retrieval_time_ms:.0f}ms")
            print(f"  • Generation time: {result.generation_time_ms:.0f}ms")
            print(f"  • Sources found: {len(result.sources)}")

            if result.sources:
                print(f"\n  Sources:")
                for i, chunk in enumerate(result.sources[:3], 1):
                    page = chunk.metadata.get("page", "N/A")
                    score = chunk.metadata.get("similarity_score", 0.0)
                    print(f"    [{i}] Page {page} (relevance: {score:.2f})")
        print()

    except Exception as e:
        print(f"\n❌ Error during query: {e}\n")
        logger.error(e)
        sys.exit(1)


def interactive_command(args):
    """
    Handle the interactive command (multi-turn conversation).

    Args:
        args: Parsed command-line arguments
    """
    try:
        print("\n" + "=" * 70)
        print("  CV QUESTION ANSWERING - Interactive Mode")
        print("=" * 70)
        print("\nType your questions and press Enter.")
        print("Commands:")
        print("  • 'exit' or 'quit' - Exit interactive mode")
        print("  • 'clear' - Clear conversation history")
        print("  • 'stats' - Show session statistics")
        print("=" * 70 + "\n")

        # Initialize pipeline
        pipeline = RAGPipeline()

        # Create session
        session_id = pipeline.create_session()
        print(f"Session ID: {session_id}\n")

        turn_number = 0

        while True:
            # Get user input
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nExiting interactive mode...\n")
                break

            # Handle special commands
            if question.lower() in ["exit", "quit"]:
                print("\nExiting interactive mode...\n")
                break

            if question.lower() == "clear":
                pipeline.clear_session(session_id)
                turn_number = 0
                print("\n✅ Conversation history cleared.\n")
                continue

            if question.lower() == "stats":
                info = pipeline.get_session_info(session_id)
                print(f"\n📊 Session Statistics:")
                print(f"   • Session ID: {info['session_id']}")
                print(f"   • Conversation turns: {info.get('num_turns', 0)}")
                print(f"   • Total messages: {info.get('total_messages', 0)}\n")
                continue

            if not question:
                continue

            turn_number += 1

            # Process query
            print("\n🤔 Thinking...\n")

            try:
                result = pipeline.query(question, session_id=session_id)

                # Display answer
                print("Assistant:", result.answer)

                if args.verbose:
                    print(
                        f"\n   [Tokens: {result.tokens_used}, "
                        f"Sources: {len(result.sources)}]"
                    )

                print()

            except Exception as e:
                print(f"❌ Error: {e}\n")
                logger.error(e)

    except Exception as e:
        print(f"\n❌ Error in interactive mode: {e}\n")
        logger.error(e)
        sys.exit(1)


def clear_command(args):
    """
    Handle the clear command (clear vector index).

    Args:
        args: Parsed command-line arguments
    """
    try:
        if not args.confirm:
            response = input(
                "\n⚠️  This will delete ALL ingested documents from the vector store.\n"
                "   Are you sure? (yes/no): "
            )
            if response.lower() not in ["yes", "y"]:
                print("\nOperation cancelled.\n")
                return

        print("\n🗑️  Clearing vector index...")

        # Initialize pipeline
        pipeline = RAGPipeline()

        # Clear index
        pipeline.clear_index()

        print("✅ Index cleared successfully.\n")
        print("You can ingest new documents using the 'ingest' command.\n")

    except Exception as e:
        print(f"\n❌ Error clearing index: {e}\n")
        logger.error(e)
        sys.exit(1)


def stats_command(args):
    """
    Handle the stats command (show index statistics).

    Args:
        args: Parsed command-line arguments
    """
    try:
        print("\n📊 Vector Store Statistics:\n")

        # Initialize pipeline
        pipeline = RAGPipeline()

        # Get stats
        stats = pipeline.get_index_stats()

        # Display stats
        if "error" in stats:
            print(f"   Error retrieving stats: {stats['error']}\n")
        else:
            # Display available information
            for key, value in stats.items():
                print(f"   • {key}: {value}")

        print()

    except Exception as e:
        print(f"\n❌ Error retrieving stats: {e}\n")
        logger.error(e)
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RAG System for CV Question Answering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest a CV
  python -m src.main.python.main ingest --pdf data/raw/cv.pdf

  # Ask a single question
  python -m src.main.python.main query "What are the candidate's key skills?"

  # Interactive conversation mode
  python -m src.main.python.main interactive

  # Clear the vector index
  python -m src.main.python.main clear --confirm

  # Show index statistics
  python -m src.main.python.main stats
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingest a PDF document into the vector store"
    )
    ingest_parser.add_argument(
        "--pdf", required=True, help="Path to PDF file to ingest"
    )

    # Query command
    query_parser = subparsers.add_parser(
        "query", help="Ask a single question (one-shot)"
    )
    query_parser.add_argument("question", help="Question to ask about the CV")
    query_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed metadata"
    )

    # Interactive command
    interactive_parser = subparsers.add_parser(
        "interactive", help="Start interactive conversation mode"
    )
    interactive_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed metadata"
    )

    # Clear command
    clear_parser = subparsers.add_parser(
        "clear", help="Clear all documents from vector store"
    )
    clear_parser.add_argument(
        "--confirm", action="store_true", help="Skip confirmation prompt"
    )

    # Stats command
    stats_parser = subparsers.add_parser(
        "stats", help="Show vector store statistics"
    )

    # Parse arguments
    args = parser.parse_args()

    # Handle commands
    if args.command == "ingest":
        ingest_command(args)
    elif args.command == "query":
        query_command(args)
    elif args.command == "interactive":
        interactive_command(args)
    elif args.command == "clear":
        clear_command(args)
    elif args.command == "stats":
        stats_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
