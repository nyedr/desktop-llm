"""Centralized profiling utilities for timing operations."""

import logging
import time
from typing import Optional
from contextlib import asynccontextmanager
import os
from logging.handlers import RotatingFileHandler

# Set up console logger
logger = logging.getLogger(__name__)

# Set up metrics logger
metrics_logger = logging.getLogger("metrics")
metrics_logger.setLevel(logging.INFO)
metrics_logger.propagate = False  # Don't propagate to root logger

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Add rotating file handler for metrics
metrics_handler = RotatingFileHandler(
    "logs/metrics.log",
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
metrics_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))
metrics_logger.addHandler(metrics_handler)


class RequestProfile:
    """Simple request profiler."""

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.start_time = time.perf_counter()
        self.first_response_time: Optional[float] = None
        self.model_request_time: Optional[float] = None

    def log_operation(self, operation: str, start_time: float, end_time: float) -> None:
        """Log an operation's timing information."""
        duration = end_time - start_time
        elapsed_from_start = start_time - self.start_time

        # Log to console
        logger.info(
            f"[TIMING][{self.request_id}] Operation '{operation}' completed in {duration:.3f}s (+{elapsed_from_start:.3f}s from start)"
        )

        # Log to metrics file
        metrics_logger.info(
            f"request_id={self.request_id}, operation={operation}, duration={duration:.3f}, elapsed_from_start={elapsed_from_start:.3f}"
        )

    def record_first_response(self) -> None:
        """Record when first response was sent."""
        if self.first_response_time is None:
            self.first_response_time = time.perf_counter()
            time_to_first = self.first_response_time - self.start_time
            model_time = self.first_response_time - \
                (self.model_request_time or self.start_time)

            # Log to console
            logger.info(
                f"[TIMING][{self.request_id}] First response chunk ready after {time_to_first:.3f}s (model took {model_time:.3f}s)"
            )

            # Log to metrics file
            metrics_logger.info(
                f"request_id={self.request_id}, event=first_response, time_to_first={time_to_first:.3f}, model_time={model_time:.3f}"
            )

    def reset_first_response(self) -> None:
        """Reset first response timing for a new generation."""
        self.first_response_time = None
        # Log to metrics file
        metrics_logger.info(
            f"request_id={self.request_id}, event=reset_first_response"
        )

    def record_model_request(self) -> None:
        """Record when request is sent to the model."""
        self.model_request_time = time.perf_counter()
        elapsed = self.model_request_time - self.start_time

        # Log to console
        logger.info(
            f"[TIMING][{self.request_id}] Sending request to model (+{elapsed:.3f}s from start)"
        )

        # Log to metrics file
        metrics_logger.info(
            f"request_id={self.request_id}, event=model_request, elapsed={elapsed:.3f}"
        )


@asynccontextmanager
async def profile_request(request_id: str):
    """Profile a request's duration."""
    profiler = RequestProfile(request_id)

    # Log to console
    logger.info(f"[TIMING][{request_id}] Request started")

    # Log to metrics file
    metrics_logger.info(f"request_id={request_id}, event=request_start")

    try:
        yield profiler
    finally:
        duration = time.perf_counter() - profiler.start_time

        # Log to console
        logger.info(
            f"[TIMING][{request_id}] Request completed in {duration:.3f}s"
        )

        # Log to metrics file
        metrics_logger.info(
            f"request_id={request_id}, event=request_end, total_duration={duration:.3f}"
        )


@asynccontextmanager
async def profile_operation(operation: str, profiler: Optional[RequestProfile] = None, request_id: Optional[str] = None):
    """Profile an operation's duration."""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        if profiler:
            profiler.log_operation(operation, start_time, end_time)
        else:
            duration = end_time - start_time
            log_prefix = f"[{request_id}] " if request_id else ""

            # Log to console
            logger.info(
                f"[TIMING]{log_prefix}Operation '{operation}' completed in {duration:.3f}s"
            )

            # Log to metrics file
            metrics_logger.info(
                f"request_id={request_id or 'unknown'}, operation={operation}, duration={duration:.3f}"
            )
