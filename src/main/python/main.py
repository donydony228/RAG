"""
Main entry point for the RAG system.

This module provides the command-line interface for the CV question answering system.
Run with: python -m src.main.python.main <command>
"""

import sys
from src.main.python.inference.cli import main


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}\n")
        sys.exit(1)
