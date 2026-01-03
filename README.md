# RAG - CV Question Answering System

A production-ready Retrieval-Augmented Generation (RAG) system that uses Claude AI to answer questions based on your CV, with Pinecone vector database and local sentence transformer embeddings.

## What is This?

This system allows you to:
- **Ingest your CV** (PDF format) into a vector database
- **Ask questions** about your CV and get accurate AI-generated answers
- **Have conversations** with context awareness across multiple turns
- **Get source citations** showing which parts of your CV were used

## Technology Stack

- **LLM**: Anthropic Claude (claude-3-5-sonnet)
- **Vector Database**: Pinecone (serverless)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2, local, free)
- **PDF Processing**: pdfplumber
- **Interface**: Command-line (CLI)

## Quick Start

### 1. Installation

```bash
# Clone the repository (or use existing directory)
cd /path/to/RAG

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the project root with your API keys:

```bash
# Copy the example file
cp .env.example .env

# Edit .env with your actual API keys
ANTHROPIC_API_KEY=your-anthropic-api-key-here
PINECONE_API_KEY=your-pinecone-api-key-here
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=cv-rag-index
```

### 3. Ingest Your CV

Place your CV PDF in `data/raw/` and ingest it:

```bash
python -m src.main.python.main ingest --pdf data/raw/cv.pdf
```

### 4. Start Asking Questions!

**Single question:**
```bash
python -m src.main.python.main query "What are my key programming skills?"
```

**Interactive conversation mode (recommended):**
```bash
python -m src.main.python.main interactive
```

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

## Commands

### Ingest Documents

Ingest a PDF document into the vector database:

```bash
python -m src.main.python.main ingest --pdf path/to/cv.pdf
```

This will:
1. Extract text from the PDF
2. Chunk the text into semantic pieces
3. Generate embeddings using Sentence Transformers
4. Upload to Pinecone vector store

### Query (Single Question)

Ask a single question:

```bash
python -m src.main.python.main query "What programming languages does the candidate know?"

# With verbose output (shows tokens, sources, performance)
python -m src.main.python.main query "What is their work experience?" -v
```

### Interactive Mode

Start a multi-turn conversation:

```bash
python -m src.main.python.main interactive
```

In interactive mode, you can:
- Ask follow-up questions with conversation context
- Type `clear` to reset the conversation
- Type `stats` to see session information
- Type `exit` or `quit` to leave

**Example conversation:**
```
You: What programming languages does the candidate know?
Assistant: Based on the CV, the candidate is proficient in Python, JavaScript, and Java...

You: Tell me more about their Python experience.
Assistant: Looking at their work history, they have 5 years of Python experience...
```

### Clear Index

Clear all documents from the vector store:

```bash
# With confirmation prompt
python -m src.main.python.main clear

# Skip confirmation (for scripts)
python -m src.main.python.main clear --confirm
```

### Show Statistics

Display vector store statistics:

```bash
python -m src.main.python.main stats
```

## Features

- **Local Embeddings**: No API costs for embeddings (Sentence Transformers)
- **Conversation History**: Multi-turn conversations with context awareness
- **Source Citations**: See which parts of the CV were used for answers
- **Production Ready**: Comprehensive error handling and logging
- **Configurable**: Easily adjust chunk size, retrieval parameters, etc.
- **MLOps Structure**: Clean, maintainable codebase following best practices

## Configuration

Edit `src/main/resources/config/config.yaml` to customize:

- **Chunking**: Strategy (paragraph/sentence), chunk size, overlap
- **Retrieval**: Number of chunks (top_k), similarity threshold
- **Generation**: Max tokens, temperature, system prompt
- **Conversation**: Max history turns, persistence settings

## Architecture

```
PDF → Extract Text → Chunk → Embed → Store in Pinecone
                                           ↓
Question → Embed → Retrieve Context → Claude API → Answer
            ↑                             ↑
            └─── Conversation History ────┘
```

## Troubleshooting

### "API key not found"
Make sure you've created `.env` file with your API keys. Copy from `.env.example`.

### "PDF file not found"
Check that the PDF path is correct. Use absolute paths or relative to project root.

### "No relevant chunks found"
Your question might be too different from CV content. Try rephrasing or check that the CV was ingested correctly.

### Import errors
Make sure you've installed all dependencies: `pip install -r requirements.txt`

### Pinecone errors
Check that your Pinecone API key is correct and you have an active account.

## Development

For developers working on this codebase:

1. **Read CLAUDE.md first** - Contains essential development rules
2. Follow the pre-task compliance checklist before coding
3. Use proper module structure under `src/main/python/`
4. Commit after every completed task
5. Run tests: `pytest src/test/`

## Project Goals

This RAG system demonstrates:
- ✅ Clean, modular architecture
- ✅ Production-ready error handling
- ✅ Comprehensive logging
- ✅ Configuration-driven design
- ✅ Type hints and documentation
- ✅ Single source of truth principle
- ✅ Technical debt prevention

---

**Built with Claude Code** | Template by Chang Ho Chien | [HC AI 說人話channel](https://youtu.be/8Q1bRZaHH24)
