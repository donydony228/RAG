# Email chunker service

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Settings for Sentence Transformers
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

# Configuration
MAX_TOKENS = 128
CHUNK_SIZE = 100
CHUNK_OVERLAP = 20

# Global variables for lazy loading
_model = None
_tokenizer = None

def get_model():
    """
    Get the Sentence Transformer model with lazy loading.

    The model is only loaded on first use, not on module import.
    Subsequent calls return the cached model instance.

    Returns:
        SentenceTransformer: The loaded model instance
    """
    global _model

    if _model is None:
        print("Loading model for the first time. This may take 1-2 minutes...")
        _model = SentenceTransformer(MODEL_NAME)
        print("Model loaded successfully.")

    return _model

def get_tokenizer():
    """
    Get the tokenizer from the Sentence Transformer model.

    Returns:
        Tokenizer: The tokenizer instance
    """
    global _tokenizer

    if _tokenizer is None:
        model = get_model()
        _tokenizer = model.tokenizer

    return _tokenizer

def count_tokens(text: str) -> int:
    """
    Count tokens using the Sentence Transformers tokenizer.

    Args:
        text (str): Input text
    Returns:
        int: Number of tokens
    """
    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(text, add_special_tokens=True)
    return len(tokens)

def split_text_for_sentence_transformer(
    content: str,
    chunk_size: int = 100,
    overlap: int = 20
) -> list[str]:
    """
    Split email content into chunks for Sentence Transformers.

    Args:
        content (str): Email content
        chunk_size (int): Desired chunk size in tokens
        overlap (int): Desired overlap in tokens
    Returns:
        list[str]: List of text chunks
    """
    # Ensure model is loaded for count_tokens function
    get_model()

    # Find Subject line
    lines = content.split('\n')
    subject_line = lines[0] if 'Subject:' in lines[0] else ''

    # Remove Subject from body
    body_start = 1 if subject_line else 0
    body_content = '\n'.join(lines[body_start:])

    # Approximate: 100 tokens is roughly 400 characters
    chunk_size_chars = chunk_size * 4
    overlap_chars = overlap * 4

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_chars,
        chunk_overlap=overlap_chars,
        separators=["\n\n", "\n", ". ", " "],
        length_function=lambda x: count_tokens(x)
    )

    body_chunks = splitter.split_text(body_content)

    # Add Subject line back to each chunk if exists
    final_chunks = []
    for chunk in body_chunks:
        if subject_line:
            final_chunk = f"{subject_line}\n\n{chunk}"
        else:
            final_chunk = chunk

        # Check if adding subject exceeds chunk size
        if count_tokens(final_chunk) <= CHUNK_SIZE:
            final_chunks.append(final_chunk)
        else:
            # If too long with subject, use chunk without subject
            final_chunks.append(chunk)

    return final_chunks



def chunk_emails(formatted_emails: list[dict]) -> list[dict]:
    """
    Chunk formatted email documents for Sentence Transformers.

    Emails that exceed the token limit are split into multiple chunks.
    Each chunk preserves the original metadata and adds chunking information.

    Args:
        formatted_emails (list[dict]): List of formatted email documents
    Returns:
        list[dict]: List of chunked email documents
    """
    chunked_docs = []

    for email_doc in formatted_emails:
        content = email_doc['content']
        token_count = count_tokens(content)

        if token_count <= CHUNK_SIZE:
            # Email is short enough, no chunking needed
            email_doc['metadata']['is_chunked'] = False
            email_doc['metadata']['chunk_index'] = 0
            email_doc['metadata']['total_chunks'] = 1
            chunked_docs.append(email_doc)
        else:
            # Email is too long, needs chunking
            chunks = split_text_for_sentence_transformer(content)

            for i, chunk_text in enumerate(chunks):
                chunked_docs.append({
                    'id': f"{email_doc['id']}_chunk_{i}",
                    'content': chunk_text,
                    'metadata': {
                        **email_doc['metadata'],
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'is_chunked': True,
                        'original_email_id': email_doc['id']
                    }
                })

    return chunked_docs

