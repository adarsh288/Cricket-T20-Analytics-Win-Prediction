"""
Utility functions for web scraping.
Includes retry logic and basic logging helpers.
"""

import time
import logging
from typing import Callable, Any

# Set up basic logging - simple configuration for a fresher project
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Browser headers to avoid 403 Forbidden errors
# Why: ESPN Cricinfo blocks requests without proper User-Agent headers
# These headers make the scraper look like a real browser request
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9"
}


def retry_request(
    request_func: Callable,
    max_retries: int = 3,
    delay: float = 2.0
) -> Any:
    """
    Retry a request function if it fails.
    
    Why this structure: Web scraping often fails due to network issues or rate limits.
    A simple retry mechanism makes the scraper more robust without adding complexity.
    
    Args:
        request_func: The function to call (usually requests.get or similar)
        max_retries: Maximum number of retry attempts
        delay: Seconds to wait between retries
        
    Returns:
        The result of the request function if successful
        
    Raises:
        Exception: If all retries fail
    """
    for attempt in range(max_retries):
        try:
            result = request_func()
            return result
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Waiting {delay} seconds before retry...")
                time.sleep(delay)
            else:
                logger.error(f"All {max_retries} attempts failed")
                raise


def log_progress(current: int, total: int, item_name: str = "items") -> None:
    """
    Log progress during scraping.
    
    Why this structure: Simple progress logging helps track long-running scrape jobs
    without needing complex progress bar libraries.
    
    Args:
        current: Current item number
        total: Total number of items
        item_name: Name of items being processed (for log message)
    """
    if total > 0:
        percent = (current / total) * 100
        logger.info(f"Progress: {current}/{total} {item_name} ({percent:.1f}%)")
