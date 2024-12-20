import logging
import sys
from typing import Optional
import structlog
from datetime import datetime

def setup_logging(
    level: str = "INFO",
    request_id: Optional[str] = None,
    environment: str = "development"
) -> None:
    """Configure structured logging for the application."""
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        logger_factory=structlog.PrintLoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        cache_logger_on_first_use=True,
    )

    # Create a logger with context
    logger = structlog.get_logger()
    logger = logger.bind(
        environment=environment,
        request_id=request_id,
        version="1.0.0"  # Add version tracking
    )

    # Add handler for stderr
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter('%(message)s')
    )
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    return logger

def get_request_logger(request_id: str):
    """Get a logger instance bound with the request ID."""
    return structlog.get_logger().bind(request_id=request_id)

def log_request_timing(logger, start_time: datetime, request_type: str):
    """Log request timing information."""
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info(
        "request_completed",
        request_type=request_type,
        duration_seconds=duration,
        start_time=start_time.isoformat(),
        end_time=end_time.isoformat()
    )
