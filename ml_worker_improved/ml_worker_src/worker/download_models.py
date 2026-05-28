"""
Download Essentia pre-trained models for audio analysis.

Models are downloaded from Essentia's official model repository.
https://essentia.upf.edu/models.html

Usage:
    python -m worker.download_models
"""

import os
import sys
import logging
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base URL for Essentia models
BASE_URL = "https://essentia.upf.edu/models"

# Models to download
# Based on https://essentia.upf.edu/models.html
MODELS = {
    # Discogs-EffNet embedding model (main model for embeddings)
    # This is the core model that extracts 1280-dim embeddings
    "discogs-effnet-bs64-1.pb": f"{BASE_URL}/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb",
    
    # Genre classification (400 genres from Discogs)
    "genre_discogs400-discogs-effnet-1.pb": f"{BASE_URL}/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb",
    "genre_discogs400-discogs-effnet-1.json": f"{BASE_URL}/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.json",
    
    # Mood models
    "mood_aggressive-discogs-effnet-1.pb": f"{BASE_URL}/classification-heads/mood_aggressive/mood_aggressive-discogs-effnet-1.pb",
    "mood_happy-discogs-effnet-1.pb": f"{BASE_URL}/classification-heads/mood_happy/mood_happy-discogs-effnet-1.pb",
    "mood_relaxed-discogs-effnet-1.pb": f"{BASE_URL}/classification-heads/mood_relaxed/mood_relaxed-discogs-effnet-1.pb",
    "mood_sad-discogs-effnet-1.pb": f"{BASE_URL}/classification-heads/mood_sad/mood_sad-discogs-effnet-1.pb",
    
    # Voice/Instrumental detection
    "voice_instrumental-discogs-effnet-1.pb": f"{BASE_URL}/classification-heads/voice_instrumental/voice_instrumental-discogs-effnet-1.pb",
    
    # Danceability
    "danceability-discogs-effnet-1.pb": f"{BASE_URL}/classification-heads/danceability/danceability-discogs-effnet-1.pb",
}


def download_file(url: str, dest_path: Path) -> bool:
    """Download a file from URL to destination path."""
    try:
        logger.info(f"Downloading: {url}")
        logger.info(f"  -> {dest_path}")
        
        # Create directory if needed
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress
        urllib.request.urlretrieve(url, str(dest_path))
        
        size_mb = dest_path.stat().st_size / (1024 * 1024)
        logger.info(f"  Downloaded: {size_mb:.1f} MB")
        return True
        
    except Exception as e:
        logger.error(f"  Failed: {e}")
        return False


def main():
    """Download all required models."""
    models_path = Path(os.getenv('MODELS_PATH', '/app/models'))
    
    logger.info("=" * 60)
    logger.info("Essentia Model Downloader")
    logger.info("=" * 60)
    logger.info(f"Models directory: {models_path}")
    logger.info("")
    
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    for filename, url in MODELS.items():
        dest_path = models_path / filename
        
        # Skip if already exists
        if dest_path.exists():
            logger.info(f"Skipping (exists): {filename}")
            skip_count += 1
            continue
        
        if download_file(url, dest_path):
            success_count += 1
        else:
            fail_count += 1
    
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Download complete!")
    logger.info(f"  Downloaded: {success_count}")
    logger.info(f"  Skipped: {skip_count}")
    logger.info(f"  Failed: {fail_count}")
    logger.info("=" * 60)
    
    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
