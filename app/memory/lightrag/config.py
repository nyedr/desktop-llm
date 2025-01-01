"""Configuration for LightRAG memory system."""

import os
from pathlib import Path

# Base directory for memory storage
LIGHTRAG_DATA_DIR = os.getenv(
    'LIGHTRAG_DATA_DIR', str(Path.home() / '.lightrag'))

# Task intervals (in seconds)
MEMORY_QUEUE_PROCESS_DELAY = 0.1
MEMORY_QUEUE_ERROR_RETRY_DELAY = 1.0
CLEANUP_INTERVAL = 3600  # 1 hour
OPTIMIZATION_INTERVAL = 7200  # 2 hours
MONITORING_INTERVAL = 300  # 5 minutes

# Retention settings
DEFAULT_RETENTION_DAYS = 30  # Default retention period for memories

# Embedding model settings
DEFAULT_EMBEDDING_MODEL = os.getenv(
    'EMBEDDING_MODEL', 'bge-m3')  # Default embedding model
# Default embedding dimensions
DEFAULT_EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', '1024'))
# Maximum tokens for embeddings
DEFAULT_MAX_TOKENS = int(os.getenv('MAX_TOKENS', '8192'))
