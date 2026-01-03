# RAG - Retrieval-Augmented Generation System

A production-ready Retrieval-Augmented Generation (RAG) system for enhanced AI responses with external knowledge integration.

## Quick Start

1. **Read CLAUDE.md first** - Contains essential rules for Claude Code
2. Follow the pre-task compliance checklist before starting any work
3. Use proper module structure under `src/main/python/`
4. Commit after every completed task

## What is RAG?

Retrieval-Augmented Generation combines the power of large language models with external knowledge retrieval to provide more accurate, contextual, and up-to-date responses.

## Project Structure

This project follows an **AI/ML MLOps-ready structure**:

```
RAG/
├── src/main/python/       # Python source code
│   ├── core/              # Core RAG algorithms
│   ├── utils/             # Data processing utilities
│   ├── models/            # Model definitions
│   ├── services/          # RAG services & pipelines
│   ├── api/               # API endpoints
│   ├── training/          # Training scripts
│   ├── inference/         # Inference engine
│   └── evaluation/        # Model evaluation
├── data/                  # Dataset management
│   ├── raw/              # Original datasets
│   ├── processed/        # Cleaned & indexed data
│   └── external/         # External knowledge sources
├── notebooks/             # Jupyter notebooks
│   ├── exploratory/      # Data exploration
│   └── experiments/      # RAG experiments
├── models/                # Trained models & embeddings
│   ├── trained/          # Production models
│   └── checkpoints/      # Training checkpoints
├── experiments/           # Experiment tracking
│   ├── configs/          # Experiment configs
│   └── results/          # Results & metrics
├── docs/                  # Documentation
├── output/                # Generated outputs
└── logs/                  # Application logs
```

## Features

- **Modular Architecture**: Clean separation of concerns
- **MLOps Ready**: Experiment tracking, model versioning
- **Scalable**: Designed for production deployment
- **Technical Debt Prevention**: Built-in safeguards
- **GitHub Auto-Backup**: Automatic version control

## Development Guidelines

- **Always search first** before creating new files
- **Extend existing** functionality rather than duplicating
- **Use Task agents** for operations >30 seconds
- **Single source of truth** for all functionality
- **Python-based** with clean, modular architecture
- **Commit frequently** after each completed feature

## Getting Started

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (once requirements.txt is created)
pip install -r requirements.txt

# Run the application
python -m src.main.python.main

# Run tests
pytest src/test/

# Launch Jupyter for experiments
jupyter notebook notebooks/
```

## Technology Stack

- **Language**: Python 3.8+
- **AI/ML**: (To be determined based on implementation)
- **Vector Database**: (To be determined)
- **APIs**: (To be determined)

## Contributing

Before contributing:
1. Read `CLAUDE.md` for development rules
2. Follow the pre-task compliance checklist
3. Use proper directory structure
4. Commit after each feature
5. Push to GitHub for backup

## License

(To be determined)

---

**Built with Claude Code** | Template by Chang Ho Chien | [HC AI 說人話channel](https://youtu.be/8Q1bRZaHH24)
