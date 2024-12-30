"""Configuration for LightRAG memory system."""

import os
from pathlib import Path

# Base directory for LightRAG data
LIGHTRAG_DATA_DIR = Path(os.getenv("LIGHTRAG_DATA_DIR", "./light_rag_data"))

# Ensure data directory exists
LIGHTRAG_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Database configuration
DB_PATH = LIGHTRAG_DATA_DIR / "memory_relations.db"

# Chunking configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))

# Embedding configuration
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_FALLBACK_MODEL = os.getenv(
    "EMBEDDING_FALLBACK_MODEL", "nomic-embed-text")
EMBEDDING_SIMILARITY_THRESHOLD = float(
    os.getenv("EMBEDDING_SIMILARITY_THRESHOLD", "0.7"))

# Entity extraction configuration
ENTITY_SPAN_MAX_WORDS = int(os.getenv("ENTITY_SPAN_MAX_WORDS", "3"))
ENTITY_TYPE_KEYWORDS = {
    key: value.split(",")
    for key, value in {
        "DATE": os.getenv("DATE_KEYWORDS", "today,tomorrow,yesterday,january,february,march,april,may,june,july,august,september,october,november,december,monday,tuesday,wednesday,thursday,friday,saturday,sunday"),
        "PERSON": os.getenv("PERSON_KEYWORDS", "mr,mrs,ms,dr,prof"),
        "ORG": os.getenv("ORG_KEYWORDS", "inc,corp,ltd,company,organization"),
        "GPE": os.getenv("GPE_KEYWORDS", "city,country,state,province")
    }.items()
}

# Relationship configuration
RELATIONSHIP_TYPE_MAP = {
    key: os.getenv(f"{key}_RELATION", default)
    for key, default in {
        "DATE": "occurred_on",
        "PERSON": "involves",
        "ORG": "involves",
        "GPE": "located_in",
        "TIME": "occurred_at",
        "MONEY": "costs",
        "PERCENT": "has_value",
        "PRODUCT": "uses",
        "EVENT": "part_of",
        "WORK_OF_ART": "references",
        "LAW": "governed_by",
        "LANGUAGE": "written_in",
        "FAC": "located_at"
    }.items()
}

# Memory processing configuration
MEMORY_QUEUE_PROCESS_DELAY = int(os.getenv("MEMORY_QUEUE_PROCESS_DELAY", "1"))
MEMORY_QUEUE_ERROR_RETRY_DELAY = int(
    os.getenv("MEMORY_QUEUE_ERROR_RETRY_DELAY", "60"))

# Task intervals (in seconds)
CLEANUP_INTERVAL = int(os.getenv("CLEANUP_INTERVAL", "3600"))
OPTIMIZATION_INTERVAL = int(os.getenv("OPTIMIZATION_INTERVAL", "86400"))
MONITORING_INTERVAL = int(os.getenv("MONITORING_INTERVAL", "600"))

# Retention policies (in days)
DEFAULT_RETENTION_DAYS = int(os.getenv("DEFAULT_RETENTION_DAYS", "30"))
SHORT_TERM_RETENTION_DAYS = int(os.getenv("SHORT_TERM_RETENTION_DAYS", "7"))
LONG_TERM_RETENTION_DAYS = int(os.getenv("LONG_TERM_RETENTION_DAYS", "365"))

# Search configuration
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "50"))
SEARCH_SIMILARITY_THRESHOLD = float(
    os.getenv("SEARCH_SIMILARITY_THRESHOLD", "0.75"))

# Entity confidence scores
ENTITY_CONFIDENCE = {
    "EXACT_MATCH": float(os.getenv("ENTITY_CONFIDENCE_EXACT", "1.0")),
    "HIGH": float(os.getenv("ENTITY_CONFIDENCE_HIGH", "0.9")),
    "MEDIUM": float(os.getenv("ENTITY_CONFIDENCE_MEDIUM", "0.7")),
    "LOW": float(os.getenv("ENTITY_CONFIDENCE_LOW", "0.5"))
}
