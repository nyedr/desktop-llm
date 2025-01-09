"""Centralized profiling utilities for timing operations."""

import logging
import time
from typing import Optional
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


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
        logger.info(
            f"[TIMING][{self.request_id}] Operation '{operation}' completed in {duration:.3f}s (+{elapsed_from_start:.3f}s from start)"
        )

    def record_first_response(self) -> None:
        """Record when first response was sent."""
        if self.first_response_time is None:
            self.first_response_time = time.perf_counter()
            time_to_first = self.first_response_time - self.start_time
            model_time = self.first_response_time - \
                (self.model_request_time or self.start_time)
            # Log before yielding the first chunk to avoid mixing with response
            logger.info(
                f"[TIMING][{self.request_id}] First response chunk ready after {time_to_first:.3f}s (model took {model_time:.3f}s)"
            )

    def record_model_request(self) -> None:
        """Record when request is sent to the model."""
        self.model_request_time = time.perf_counter()
        elapsed = self.model_request_time - self.start_time
        logger.info(
            f"[TIMING][{self.request_id}] Sending request to model (+{elapsed:.3f}s from start)"
        )


@asynccontextmanager
async def profile_request(request_id: str):
    """Profile a request's duration."""
    profiler = RequestProfile(request_id)
    logger.info(f"[TIMING][{request_id}] Request started")
    try:
        yield profiler
    finally:
        duration = time.perf_counter() - profiler.start_time
        logger.info(
            f"[TIMING][{request_id}] Request completed in {duration:.3f}s")


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
            logger.info(
                f"[TIMING]{log_prefix}Operation '{operation}' completed in {duration:.3f}s")
