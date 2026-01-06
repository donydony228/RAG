"""Debug script to inspect CV chunks."""
import sys
sys.path.insert(0, '/Users/desmond/Documents/作品集/RAG')

from src.main.python.services.pdf_service import PDFService
from src.main.python.utils.chunker import ParagraphChunker
from src.main.python.utils.config import Config

# Initialize services
config = Config.load()
pdf_service = PDFService()
chunker = ParagraphChunker(config)

# Extract text
pdf_path = "data/raw/cv.pdf"
documents = pdf_service.extract_text(pdf_path)

print(f"=== EXTRACTED {len(documents)} PAGES ===\n")
for doc in documents:
    page = doc.get('page', '?') if isinstance(doc, dict) else doc.metadata.get('page', '?')
    content = doc.get('text', '') if isinstance(doc, dict) else doc.content
    print(f"Page {page}:")
    print(f"Content length: {len(content)} characters")
    print(f"First 500 chars: {content[:500]}")
    print()

# Create chunks
all_chunks = []
for doc in documents:
    content = doc.get('text', '') if isinstance(doc, dict) else doc.content
    metadata = {'page': doc.get('page', 1)} if isinstance(doc, dict) else doc.metadata
    chunks = chunker.chunk(content, metadata)
    all_chunks.extend(chunks)

print(f"\n=== CREATED {len(all_chunks)} CHUNKS ===\n")
for i, chunk in enumerate(all_chunks, 1):
    print(f"Chunk {i}:")
    print(f"Length: {len(chunk.content)} characters")
    print(f"Content: {chunk.content[:300]}...")
    print()
