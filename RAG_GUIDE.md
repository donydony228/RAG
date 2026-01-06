# RAG System Architecture Guide

## 📚 Table of Contents
1. [What is RAG?](#what-is-rag)
2. [System Overview](#system-overview)
3. [Architecture Diagram](#architecture-diagram)
4. [Component Deep Dive](#component-deep-dive)
5. [Data Flow](#data-flow)
6. [Key Technologies](#key-technologies)

---

## What is RAG?

**RAG (Retrieval-Augmented Generation)** combines two powerful AI techniques:

1. **Retrieval**: Finding relevant information from a knowledge base (your CV)
2. **Generation**: Using an LLM (Claude) to generate natural language answers

**Why RAG?**
- LLMs have limited context windows and can't memorize all information
- RAG gives LLMs access to external knowledge dynamically
- More accurate than pure generation, more natural than pure search

---

## System Overview

Your RAG system answers questions about your CV by:

```
1. INGESTION PHASE (One-time setup)
   CV.pdf → Extract Text → Chunk → Embed → Store in Pinecone

2. QUERY PHASE (Every question)
   Question → Embed → Search Pinecone → Get Relevant Chunks → Ask Claude → Answer
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                       │
└─────────────────────────────────────────────────────────────┘

    data/raw/cv.pdf
         │
         ▼
    [PDFService]
    - pdfplumber with layout=True
    - Extracts text with proper spacing
    - Normalizes whitespace
         │
         ▼
    "Ching-Yuan Peng, New York University..."
         │
         ▼
    [ParagraphChunker]
    - Splits by double newlines
    - Combines paragraphs up to 512 tokens
    - Adds 50 token overlap
         │
         ▼
    [Chunk 1] "Education: NYU Master's..."
    [Chunk 2] "Work Experience: Deepaign Inc..."
    [Chunk 3] "Projects: Spotify Analytics..."
         │
         ▼
    [EmbeddingService]
    - Sentence Transformers (all-MiniLM-L6-v2)
    - Generates 384-dim vectors
    - Runs locally (no API cost)
         │
         ▼
    [Vector 1] [0.23, -0.45, 0.67, ...]  (384 dimensions)
    [Vector 2] [0.12, 0.78, -0.34, ...]
    [Vector 3] [-0.56, 0.23, 0.89, ...]
         │
         ▼
    [Pinecone Vector Store]
    - Stores vectors with metadata
    - Enables fast similarity search
    - Cloud-hosted, serverless


┌─────────────────────────────────────────────────────────────┐
│                       QUERY PIPELINE                         │
└─────────────────────────────────────────────────────────────┘

    User: "What is Desmond's education?"
         │
         ▼
    [EmbeddingService]
    - Embed the question
         │
         ▼
    Query Vector: [-0.12, 0.45, 0.23, ...]  (384 dimensions)
         │
         ▼
    [Pinecone Search]
    - Cosine similarity search
    - Returns top-k most similar chunks
    - Similarity scores: 0.1859, 0.1451, 0.0584
         │
         ▼
    [Retriever]
    - Filters by threshold (>0.05)
    - Keeps chunks with score ≥ 0.05
         │
         ▼
    Retrieved Chunks:
    [Chunk 1] "Education: NYU Master's..." (score: 0.1859)
    [Chunk 2] "Work Experience..." (score: 0.1451)
         │
         ▼
    [ContextBuilder]
    - Formats chunks into context string
    - Adds conversation history (if any)
    - Builds messages for Claude
         │
         ▼
    Context:
    """
    [Source 1] (Page 1) [Relevance: 0.19]:
    Education: NYU Master's in Data Science (2025-2027)...

    [Source 2] (Page 1) [Relevance: 0.15]:
    Work Experience: Deepaign Inc. Co-Founder...
    """
         │
         ▼
    [LLMService]
    - Sends context + question to Claude 3 Haiku
    - Claude generates natural language answer
         │
         ▼
    Answer:
    "Desmond's educational background includes:
    - NYU: Master of Science in Data Science (2025-2027)
    - National Taiwan University: Bachelor of Business Administration (2021-2025)
    - Relevant coursework: NLP, ML, Data Analysis, Database Management"
         │
         ▼
    [ConversationManager]
    - Stores question + answer in session history
    - Enables follow-up questions
         │
         ▼
    Display to User
```

---

## Component Deep Dive

### 1. **PDFService** (`src/main/python/services/pdf_service.py`)

**Purpose**: Extract text from PDF files

**Key Features**:
- Uses `pdfplumber` with `layout=True` for spatial awareness
- Normalizes excessive whitespace while preserving structure
- Returns text organized by page with metadata

**Why It Matters**:
- Original PDFs often have spacing issues ("NewYorkUniversity")
- Layout-aware extraction preserves word boundaries
- Proper spacing is critical for embedding quality

**Code Example**:
```python
# Extract with proper spacing
text = page.extract_text(layout=True, x_tolerance=2, y_tolerance=3)

# Normalize whitespace
text = re.sub(r' +', ' ', text)  # Multiple spaces → single space
```

---

### 2. **ParagraphChunker** (`src/main/python/utils/chunker.py`)

**Purpose**: Split CV text into semantically meaningful chunks

**Strategy**: Paragraph-based chunking
- Splits by double newlines (`\n\n`)
- Combines paragraphs until reaching 512 tokens
- Adds 50 token overlap between chunks

**Why Paragraph-Based?**
- CVs are naturally structured by sections (Education, Experience, Skills)
- Paragraph breaks align with semantic boundaries
- Better than fixed-size chunks which might split mid-sentence

**Overlap Strategy**:
```
Chunk 1: [Education Section] [Part of Work Experience]
Chunk 2:                     [Part of Work Experience] [Projects]
Chunk 3:                                              [Projects] [Skills]
         ^^^^^^^^^^^^^^^^^^ 50 token overlap ^^^^^^^^^^^^^^^^^^
```

**Benefit**: Questions about overlapping content can retrieve multiple relevant chunks

---

### 3. **EmbeddingService** (`src/main/python/services/embedding_service.py`)

**Purpose**: Convert text into numerical vectors (embeddings)

**Model**: `all-MiniLM-L6-v2` from Sentence Transformers
- **Dimension**: 384 (each embedding is a list of 384 numbers)
- **Runs locally**: No API costs, fast, private
- **Semantic Understanding**: Similar texts have similar vectors

**How Embeddings Work**:
```python
# Text to vector
"Master of Science in Data Science" → [0.23, -0.45, 0.67, ..., 0.12] (384 dims)
"educational background"            → [0.21, -0.42, 0.63, ..., 0.15] (384 dims)
                                      ^^^ Similar vectors! (high cosine similarity)

"Python programming"                → [-0.56, 0.78, -0.23, ..., 0.89] (384 dims)
                                      ^^^ Different vector (low similarity)
```

**Key Property**: Cosine similarity measures how similar two vectors are
- 1.0 = Identical
- 0.8-0.9 = Very similar
- 0.1-0.3 = Somewhat related (typical for CV Q&A)
- 0.0 = Orthogonal (unrelated)
- -1.0 = Opposite

---

### 4. **Pinecone Vector Store** (`src/main/python/services/vector_store_service.py`)

**Purpose**: Store and search embeddings at scale

**How It Works**:
```
Storage:
chunk_id: "chunk_001"
vector: [0.23, -0.45, 0.67, ...]  (384 numbers)
metadata: {
    "page": 1,
    "full_content": "Education: NYU Master's...",
    "source": "cv.pdf"
}

Search:
1. Get query vector: [-0.12, 0.45, 0.23, ...]
2. Calculate cosine similarity with all stored vectors
3. Return top-k most similar (sorted by score)
```

**Why Pinecone?**
- **Fast**: HNSW algorithm for approximate nearest neighbor search
- **Scalable**: Handles millions of vectors
- **Serverless**: No infrastructure management
- **Cloud-hosted**: Accessible from anywhere

**Index Configuration**:
- Dimension: 384 (matches embedding model)
- Metric: Cosine (best for text similarity)
- Namespace: "cv" (organizes data)

---

### 5. **Retriever** (`src/main/python/core/retriever.py`)

**Purpose**: Find relevant chunks for a query

**Process**:
1. Embed the question
2. Search Pinecone (returns top 5 by default)
3. Filter by similarity threshold (>0.05)
4. Return filtered chunks sorted by relevance

**Threshold Strategy**:
```python
# Example scores from Pinecone
Results: [
    {"score": 0.1859, "content": "Education..."},    # ✅ Retrieved (>0.05)
    {"score": 0.1451, "content": "Work exp..."},     # ✅ Retrieved (>0.05)
    {"score": 0.0584, "content": "Projects..."},     # ✅ Retrieved (>0.05)
    {"score": -0.0340, "content": "Skills..."}       # ❌ Filtered out (<0.05)
]
```

**Why Threshold 0.05?**
- Too high (0.3): Misses relevant chunks, retrieves nothing
- Too low (0.0): Retrieves irrelevant noise
- 0.05: Sweet spot for CV semantic search

---

### 6. **ContextBuilder** (`src/main/python/core/context_builder.py`)

**Purpose**: Format retrieved chunks into a prompt for Claude

**Output Format**:
```
[Source 1] (Page 1) [Relevance: 0.19]:
Ching-Yuan (Desmond) Peng
Education:
New York University (Sep 2025 - Jun 2027): Master of Science in Data Science
National Taiwan University (Sep 2021 - Jun 2025): Bachelor of Business Administration

[Source 2] (Page 1) [Relevance: 0.15]:
Professional Experience:
Co-Founder & Data Scientist, Deepaign Inc. (Feb 2025 - Sep 2025)
- Launched LangChain AI pipeline for political petitions...
```

**Also Handles**:
- Conversation history (last 5 turns)
- Token counting to stay within limits
- System prompt injection

---

### 7. **LLMService** (`src/main/python/services/llm_service.py`)

**Purpose**: Generate natural language answers using Claude

**Model**: Claude 3 Haiku (`claude-3-haiku-20240307`)
- Fast and cost-effective
- Good balance of quality vs. speed
- Handles up to 200K tokens context

**API Call Structure**:
```python
messages = [
    {"role": "user", "content": f"""
        Context from CV:
        {formatted_chunks}

        Question: {user_question}
    """}
]

response = client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=1024,
    temperature=0.7,
    system="You are an AI assistant answering questions about a person's CV...",
    messages=messages
)
```

**With Retry Logic**:
- 3 attempts with exponential backoff (2s, 4s, 8s)
- Handles transient API failures gracefully

---

### 8. **ConversationManager** (`src/main/python/utils/conversation_manager.py`)

**Purpose**: Maintain conversation context across multiple turns

**Session Management**:
```python
session = {
    "session_id": "uuid-1234",
    "history": [
        {"role": "user", "content": "What is Desmond's education?", "tokens": 0},
        {"role": "assistant", "content": "Desmond has...", "tokens": 471},
        {"role": "user", "content": "Tell me more about NYU", "tokens": 0},
        {"role": "assistant", "content": "At NYU...", "tokens": 523}
    ]
}
```

**Features**:
- Keeps last 5 conversation turns (configurable)
- Persists to disk (`data/temp/conversation_history.json`)
- Enables follow-up questions with context

**Example Conversation**:
```
You: What universities did Desmond attend?
Assistant: NYU and National Taiwan University

You: Tell me more about NYU  [Understands "NYU" refers to previous context]
Assistant: At NYU, Desmond is pursuing a Master's in Data Science...
```

---

### 9. **RAGPipeline** (`src/main/python/core/rag_pipeline.py`)

**Purpose**: Orchestrate the entire RAG workflow

**Main Methods**:

#### `ingest_document(pdf_path)`
```python
def ingest_document(self, pdf_path):
    # Step 1: Extract text from PDF
    chunks = self.document_processor.process_pdf(pdf_path)

    # Step 2: Generate embeddings
    embeddings = self.embedding_service.embed_batch([c.content for c in chunks])

    # Step 3: Upload to Pinecone
    self.vector_store.upsert_chunks(chunks, embeddings)
```

#### `query(question, session_id)`
```python
def query(self, question, session_id=None):
    # Step 1: Retrieve relevant chunks
    chunks = self.retriever.retrieve(question)

    # Step 2: Build context with history
    context = self.context_builder.build_context(chunks, session_id)

    # Step 3: Generate answer with Claude
    answer, tokens = self.llm_service.generate(context, question)

    # Step 4: Update conversation history
    self.conversation_manager.add_turn(session_id, question, answer, tokens)

    return QueryResult(answer=answer, sources=chunks, tokens_used=tokens)
```

---

## Data Flow

### Ingestion Flow (One-Time)

```
1. User Command:
   python -m src.main.python.main ingest --pdf data/raw/cv.pdf

2. PDF → Text:
   PDFService extracts: "Ching-Yuan (Desmond) Peng\n\nEducation..."

3. Text → Chunks:
   ParagraphChunker creates 3 chunks:
   - Chunk 1: Education section (2407 chars)
   - Chunk 2: Work experience (2678 chars)
   - Chunk 3: Projects & skills (1360 chars)

4. Chunks → Embeddings:
   EmbeddingService generates 3 vectors (384-dim each)
   - [0.23, -0.45, 0.67, ..., 0.12]
   - [0.12, 0.78, -0.34, ..., 0.45]
   - [-0.56, 0.23, 0.89, ..., 0.78]

5. Upload to Pinecone:
   VectorStoreService stores:
   {
     "id": "chunk-001",
     "values": [0.23, -0.45, ...],
     "metadata": {"page": 1, "content": "Education..."}
   }

6. Result:
   ✅ 3 chunks indexed in 1.4 seconds
```

### Query Flow (Every Question)

```
1. User Question:
   "What is Desmond's educational background?"

2. Question → Embedding:
   EmbeddingService: "education" → [-0.12, 0.45, 0.23, ...]

3. Semantic Search:
   Pinecone compares query vector with stored vectors:
   - Chunk 1 (Education): similarity = 0.1859 ✅
   - Chunk 2 (Work exp): similarity = 0.1451 ✅
   - Chunk 3 (Projects): similarity = 0.0584 ✅

4. Threshold Filter:
   Retriever keeps chunks with score > 0.05
   → 3 chunks retrieved

5. Context Building:
   ContextBuilder formats:
   """
   [Source 1] (Relevance: 0.19):
   Education: NYU Master's...

   [Source 2] (Relevance: 0.15):
   Work Experience...
   """

6. Claude Generation:
   LLMService sends to Claude:
   System: "You are an AI assistant answering questions about a CV..."
   User: "Context: [chunks]\n\nQuestion: What is Desmond's education?"

7. Claude Response:
   "Based on the CV, Desmond's education includes:
   - NYU: Master of Science in Data Science (2025-2027)
   - National Taiwan University: BBA (2021-2025)
   - Coursework: NLP, ML, Data Analysis, Database Management"

8. Update History:
   ConversationManager stores turn for follow-up questions

9. Display:
   User sees answer with metadata (tokens, time, sources)
```

---

## Key Technologies

### 1. **Sentence Transformers**
- **What**: Open-source library for semantic embeddings
- **Model**: all-MiniLM-L6-v2 (22M parameters)
- **Speed**: ~100 sentences/second on CPU
- **Cost**: Free (runs locally)
- **Quality**: Good for short texts like CV sections

### 2. **Pinecone**
- **What**: Managed vector database
- **Algorithm**: HNSW (Hierarchical Navigable Small World)
- **Performance**: Sub-10ms query latency
- **Scale**: Billions of vectors
- **Pricing**: Free tier: 100K vectors

### 3. **Claude 3 Haiku**
- **What**: Fast, lightweight LLM from Anthropic
- **Context**: 200K tokens (~150K words)
- **Speed**: ~2 seconds for typical responses
- **Cost**: $0.25 per 1M input tokens, $1.25 per 1M output tokens
- **Quality**: Better than GPT-3.5, faster than GPT-4

### 4. **pdfplumber**
- **What**: PDF text extraction library
- **Features**: Layout-aware extraction, table parsing
- **Advantage**: Better spacing preservation than PyPDF2
- **Configuration**: `layout=True` for spatial text analysis

---

## Performance Metrics

### Ingestion (1-page CV)
- **Text Extraction**: ~0.1 seconds
- **Chunking**: ~0.01 seconds
- **Embedding Generation**: ~0.5 seconds (batch of 3)
- **Pinecone Upload**: ~0.8 seconds
- **Total**: ~1.4 seconds

### Query (Single Question)
- **Question Embedding**: ~0.1 seconds
- **Pinecone Search**: ~0.8 seconds (3 chunks)
- **Context Building**: ~0.01 seconds
- **Claude Generation**: ~1.5 seconds
- **Total**: ~2.4 seconds

### Costs (Per 100 Questions)
- **Embeddings**: $0 (local)
- **Pinecone**: $0 (free tier)
- **Claude 3 Haiku**: ~$0.05-0.10
- **Total**: <$0.10 per 100 questions 🎉

---

## Configuration Guide

All settings in `src/main/python/resources/config/config.yaml`:

```yaml
rag:
  chunking:
    strategy: "paragraph"      # How to split text
    max_tokens: 512            # Max chunk size
    overlap_tokens: 50         # Overlap between chunks

  retrieval:
    top_k: 5                   # Retrieve up to 5 chunks
    similarity_threshold: 0.05 # Minimum similarity score

  generation:
    max_tokens: 1024           # Max response length
    temperature: 0.7           # Creativity (0=deterministic, 1=creative)
    claude_model: "claude-3-haiku-20240307"
```

**Tuning Tips**:
- **Low retrieval**: Lower `similarity_threshold` (try 0.03)
- **Too much context**: Reduce `top_k` (try 3)
- **Responses too long**: Reduce `max_tokens` (try 512)
- **More creative**: Increase `temperature` (try 0.9)

---

## Common Issues & Solutions

### Issue: "No relevant chunks found"
**Cause**: Similarity threshold too high
**Solution**: Lower threshold in `config.yaml`
```yaml
similarity_threshold: 0.03  # Lower for more recall
```

### Issue: "Chunk doesn't have information"
**Cause**: Not retrieving enough chunks
**Solution**: Increase top_k or lower threshold
```yaml
top_k: 10  # Retrieve more chunks
```

### Issue: "Response is incomplete"
**Cause**: Token limit too low
**Solution**: Increase max_tokens
```yaml
max_tokens: 2048  # Allow longer responses
```

### Issue: "PDF text has no spaces"
**Cause**: PDF extraction issue
**Solution**: Already fixed with `layout=True` + whitespace normalization

---

## Extending the System

### Add More Documents
```bash
# Ingest additional documents
python -m src.main.python.main ingest --pdf data/raw/resume2.pdf
python -m src.main.python.main ingest --pdf data/raw/portfolio.pdf

# All documents searchable together
python -m src.main.python.main query "What projects are mentioned?"
```

### Use Different LLM Models
Edit `.env`:
```bash
# Faster (cheaper)
CLAUDE_MODEL=claude-3-haiku-20240307

# Higher quality (more expensive)
CLAUDE_MODEL=claude-3-5-sonnet-20241022

# Most capable (most expensive)
CLAUDE_MODEL=claude-3-opus-20240229
```

### Improve Retrieval Quality
Try different embedding models in `config.yaml`:
```yaml
embedding:
  model_name: "all-mpnet-base-v2"  # Better quality (768-dim)
  # model_name: "all-MiniLM-L6-v2"  # Faster (384-dim)
```

---

## Testing & Debugging

### Test Ingestion
```bash
python -m src.main.python.main ingest --pdf data/raw/cv.pdf
# Should show: "✅ Ingestion complete! • Chunks created: 3"
```

### Test Retrieval (with scores)
```bash
python -m src.main.python.main query "education" -v
# Check logs for: "Result score: 0.1859 (threshold: 0.05)"
```

### Test Interactive Mode
```bash
python -m src.main.python.main interactive
You: What is the education?
Assistant: [Answer about NYU and National Taiwan University]
You: Tell me more about NYU
Assistant: [Follow-up answer with context]
```

### Check Index Stats
```bash
python -m src.main.python.main stats
# Shows: vector_count, dimension, metric
```

---

## Next Steps

### Immediate Improvements
1. **Add More CV Sections**: Skills, certifications, publications
2. **Improve Chunking**: Experiment with sentence-based or fixed-size
3. **Better Metadata**: Track section types (Education, Experience, etc.)

### Advanced Features
1. **Multi-Document Search**: Ingest portfolio, cover letters, etc.
2. **Citation Links**: Add page/section links in responses
3. **Semantic Caching**: Cache common questions for faster responses
4. **Reranking**: Use cross-encoder to improve retrieval quality

### Production Deployment
1. **API Wrapper**: Create FastAPI endpoint
2. **Web UI**: Build React/Next.js frontend
3. **Monitoring**: Add logging, metrics, error tracking
4. **Authentication**: Secure access with API keys

---

## Conclusion

Your RAG system is a **production-ready, end-to-end pipeline** that:

✅ **Ingests** CV PDFs with proper text extraction
✅ **Chunks** text into semantic pieces with overlap
✅ **Embeds** text locally with Sentence Transformers
✅ **Stores** vectors in Pinecone for fast similarity search
✅ **Retrieves** relevant chunks based on semantic similarity
✅ **Generates** natural language answers with Claude 3 Haiku
✅ **Maintains** conversation context for follow-up questions

**Total Cost**: <$0.10 per 100 questions 💰
**Response Time**: ~2.5 seconds per query ⚡
**Accuracy**: High-quality answers with source citations 🎯

You now have a fully functional RAG system that you can extend, customize, and deploy! 🚀
