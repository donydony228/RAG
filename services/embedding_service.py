# Embedding service to generate embeddings for chunked email documents

from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np

# Reuse the model from email_chunker
from services.email_chunker import get_model, MODEL_NAME

def generate_embeddings(
    chunked_emails: List[Dict],
    batch_size: int = 32,
    show_progress: bool = True
) -> List[Dict]:
    """
    Generate embeddings for chunked email documents.
    
    Args:
        chunked_emails: List of chunked email documents
        batch_size: Number of documents to process at once
        show_progress: Show progress bar
    
    Returns:
        List of documents with embeddings added
    """
    # Load the model (uses lazy loading from email_chunker)
    model = get_model()
    
    # Extract all content texts
    contents = [email['content'] for email in chunked_emails]
    
    # Generate embeddings in batches
    embeddings = model.encode(
        contents,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True  # For cosine similarity
    )
    
    # Add embeddings to documents
    docs_with_embeddings = []
    for email, embedding in zip(chunked_emails, embeddings):
        docs_with_embeddings.append({
            **email,
            'embedding': embedding.tolist()  # Convert numpy to list for JSON
        })
    
    return docs_with_embeddings
