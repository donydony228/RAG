"""
Logging configuration and utilities for the RAG system.

This module provides centralized logging setup based on YAML configuration.
"""

import logging
import logging.config
import os
from pathlib import Path
from typing import Optional
import yaml


def setup_logging(
    config_path: Optional[str] = None, default_level: int = logging.INFO
) -> None:
    """
    Setup logging configuration from YAML file.

    Args:
        config_path: Path to logging configuration YAML file.
                    If None, uses default path.
        default_level: Default logging level if config file not found.
    """
    if config_path is None:
        # Default path to logging config
        config_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "resources",
            "config",
            "logging.yaml",
        )

    config_path = Path(config_path).resolve()

    # Ensure logs directory exists
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f.read())
                logging.config.dictConfig(config)
        except Exception as e:
            logging.basicConfig(level=default_level)
            logging.error(f"Error loading logging config from {config_path}: {e}")
            logging.info("Using basic logging configuration")
    else:
        logging.basicConfig(
            level=default_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        logging.warning(f"Logging config not found at {config_path}, using basic config")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Name of the logger (usually __name__ of the calling module)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Initialize logging when module is imported
setup_logging()
