"""
Email ingestion pipeline that orchestrates the entire workflow.

Workflow:
1. Fetch emails from Gmail
2. Preprocess and clean
3. Chunk long emails
4. Generate embeddings
5. Upload to Pinecone
"""

from typing import Dict, List, Optional
from datetime import datetime
import logging
import os

from services.gmail_service import fetch_emails_from_multiple_accounts
from services.email_preprocessor import format_email
from services.email_chunker import chunk_emails
from services.embedding_service import generate_embeddings
from services.pinecone_service import PineconeService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Default accounts configuration
DEFAULT_ACCOUNTS = [
    {
        'label': '個人信箱',
        'credentials_path': 'credentials/credentials_account1.json',
        'token_path': 'credentials/token_account1.json'
    },
    {
        'label': '工作信箱',
        'credentials_path': 'credentials/credentials_account2.json',
        'token_path': 'credentials/token_account2.json'
    },
    {
        'label': '其他信箱',
        'credentials_path': 'credentials/credentials_account3.json',
        'token_path': 'credentials/token_account3.json'
    }
]

class EmailIngestionPipeline:
    """
    Orchestrates the complete email ingestion workflow.
    """
    
    def __init__(self, pinecone_api_key: Optional[str] = None):
        """
        Initialize the ingestion pipeline.
        
        Args:
            pinecone_api_key: Pinecone API key (optional)
        """
        self.pinecone_service = PineconeService(api_key=pinecone_api_key)
        self.pinecone_service.create_index_if_not_exists()
        
        self.stats = {
            'emails_fetched': 0,
            'emails_formatted': 0,
            'emails_skipped': 0,
            'emails_new': 0,
            'chunks_created': 0,
            'embeddings_generated': 0,
            'vectors_uploaded': 0,
            'start_time': None,
            'end_time': None,
        }
    
    def run(
        self,
        time_range: str = "7d",
        max_emails: int = 100,
        batch_size: int = 32,
        show_progress: bool = True,
        incremental: bool = True
    ) -> Dict:
        """
        Run the complete ingestion pipeline.

        Args:
            time_range: Time range for email fetching (e.g., "7d", "30d")
            max_emails: Maximum number of emails to fetch
            batch_size: Batch size for embedding generation
            show_progress: Show progress bars
            incremental: Skip emails that already exist in Pinecone (default: True)

        Returns:
            Dictionary with pipeline statistics
        """
        self.stats['start_time'] = datetime.now()

        try:
            # Step 1: Fetch emails
            logger.info(f"Step 1/6: Fetching emails (time_range={time_range}, max_per_account={max_emails})")
            emails = self._fetch_emails(time_range, max_emails)

            # Step 2: Preprocess
            logger.info("Step 2/6: Preprocessing emails")
            formatted_emails = self._preprocess_emails(emails)

            # Step 2.5: Filter existing emails (if incremental mode)
            if incremental:
                logger.info("Step 2.5/6: Filtering existing emails (incremental mode)")
                formatted_emails = self._filter_existing_emails(formatted_emails, show_progress)

                # Skip rest if no new emails
                if not formatted_emails:
                    logger.info("No new emails to process. Exiting.")
                    self.stats['end_time'] = datetime.now()
                    self.stats['duration'] = (
                        self.stats['end_time'] - self.stats['start_time']
                    ).total_seconds()
                    return self._get_final_stats({'total_uploaded': 0, 'failed_ids': [], 'success_rate': 100.0})

            # Step 3: Chunk
            logger.info("Step 3/6: Chunking long emails")
            chunked_emails = self._chunk_emails(formatted_emails)
            
            # Step 4: Generate embeddings
            logger.info("Step 4/6: Generating embeddings")
            docs_with_embeddings = self._generate_embeddings(
                chunked_emails,
                batch_size,
                show_progress
            )

            # Step 5: Upload to Pinecone
            logger.info("Step 5/6: Uploading to Pinecone")
            upload_stats = self._upload_to_pinecone(
                docs_with_embeddings,
                show_progress
            )
            
            self.stats['end_time'] = datetime.now()
            self.stats['duration'] = (
                self.stats['end_time'] - self.stats['start_time']
            ).total_seconds()
            
            logger.info("Pipeline completed successfully!")
            return self._get_final_stats(upload_stats)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def _fetch_emails(self, time_range: str, max_emails_per_account: int) -> List[Dict]:
        """Fetch emails from Gmail."""
        emails = fetch_emails_from_multiple_accounts(
            accounts=DEFAULT_ACCOUNTS,
            time_range=time_range,
            max_emails_per_account=max_emails_per_account,
            query=''
        )
        self.stats['emails_fetched'] = len(emails)
        logger.info(f"Fetched {len(emails)} emails")
        return emails
    
    def _preprocess_emails(self, emails: List[Dict]) -> List[Dict]:
        """Preprocess and format emails."""
        formatted = format_email(emails)
        self.stats['emails_formatted'] = len(formatted)
        return formatted

    def _filter_existing_emails(
        self,
        formatted_emails: List[Dict],
        show_progress: bool
    ) -> List[Dict]:
        """Filter out emails that already exist in Pinecone."""
        new_emails, filter_stats = self.pinecone_service.filter_new_emails(
            formatted_emails,
            show_progress=show_progress
        )

        self.stats['emails_skipped'] = filter_stats['existing']
        self.stats['emails_new'] = filter_stats['new']

        logger.info(
            f"Filtered emails: {filter_stats['new']} new, "
            f"{filter_stats['existing']} skipped (already exist)"
        )

        return new_emails

    def _chunk_emails(self, formatted_emails: List[Dict]) -> List[Dict]:
        """Chunk long emails."""
        chunked = chunk_emails(formatted_emails)
        self.stats['chunks_created'] = len(chunked)
        logger.info(f"Created {len(chunked)} chunks")
        return chunked
    
    def _generate_embeddings(
        self,
        chunked_emails: List[Dict],
        batch_size: int,
        show_progress: bool
    ) -> List[Dict]:
        """Generate embeddings for chunks."""
        docs_with_embeddings = generate_embeddings(
            chunked_emails,
            batch_size=batch_size,
            show_progress=show_progress
        )
        self.stats['embeddings_generated'] = len(docs_with_embeddings)
        logger.info(f"Generated {len(docs_with_embeddings)} embeddings")
        return docs_with_embeddings
    
    def _upload_to_pinecone(
        self,
        docs_with_embeddings: List[Dict],
        show_progress: bool
    ) -> Dict:
        """Upload vectors to Pinecone."""
        upload_stats = self.pinecone_service.upsert(
            docs_with_embeddings=docs_with_embeddings,
            show_progress=show_progress
        )
        self.stats['vectors_uploaded'] = upload_stats['total_uploaded']
        logger.info(f"Uploaded {upload_stats['total_uploaded']} vectors")
        return upload_stats
    
    def _get_final_stats(self, upload_stats: Dict) -> Dict:
        """Compile final statistics."""
        return {
            **self.stats,
            'success_rate': upload_stats['success_rate'],
            'failed_ids': upload_stats['failed_ids'],
        }
    
    def get_index_stats(self) -> Dict:
        """Get current Pinecone index statistics."""
        return self.pinecone_service.get_index_stats()


# Convenience function for simple usage
def ingest_emails(
    time_range: str = "7d",
    max_emails: int = 100,
    batch_size: int = 32,
    show_progress: bool = True,
    incremental: bool = True
) -> Dict:
    """
    Simple function to run the entire ingestion pipeline.

    Args:
        time_range: Time range for email fetching
        max_emails: Maximum number of emails to fetch
        batch_size: Batch size for embedding generation
        show_progress: Show progress bars
        incremental: Skip emails that already exist in Pinecone (default: True)

    Returns:
        Dictionary with pipeline statistics

    Example:
        >>> # Incremental update (skip existing emails)
        >>> stats = ingest_emails(time_range="7d", max_emails=100, incremental=True)
        >>> print(f"Uploaded {stats['vectors_uploaded']} new vectors")
        >>>
        >>> # Full re-ingestion (process all emails)
        >>> stats = ingest_emails(time_range="30d", max_emails=500, incremental=False)
    """
    pipeline = EmailIngestionPipeline()
    return pipeline.run(
        time_range=time_range,
        max_emails=max_emails,
        batch_size=batch_size,
        show_progress=show_progress,
        incremental=incremental
    )
