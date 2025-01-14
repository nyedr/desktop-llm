"""Centralized profiling utilities for timing operations."""

import logging
import time
from typing import Optional, Dict, Any
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
    """Request profiler with tool execution tracking and error monitoring."""

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.start_time = time.perf_counter()
        self.first_response_time: Optional[float] = None
        self.model_request_time: Optional[float] = None
        self.first_audio_time: Optional[float] = None

        # Tool execution tracking
        self.tool_timings: Dict[str, Dict[str, Any]] = {}
        self.total_tool_time: float = 0.0

        # Error tracking
        self.error_count: int = 0
        self.last_error: Optional[Dict[str, Any]] = None
        self.retry_count: int = 0
        self.max_retries: int = 3  # Maximum number of retries before giving up

    def log_operation(self, operation: str, start_time: float, end_time: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log an operation's timing information."""
        duration = end_time - start_time
        elapsed_from_start = start_time - self.start_time

        # Log to console
        logger.info(
            f"[TIMING][{self.request_id}] Operation '{operation}' completed in {duration:.3f}s (+{elapsed_from_start:.3f}s from start)"
        )

        # Log to metrics file with metadata
        log_data = {
            "request_id": self.request_id,
            "operation": operation,
            "duration": f"{duration:.3f}",
            "elapsed_from_start": f"{elapsed_from_start:.3f}"
        }
        if metadata:
            log_data.update(metadata)

        metrics_logger.info(", ".join(f"{k}={v}" for k, v in log_data.items()))

    def record_error(self, error: str, error_type: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Record an error and check if we should continue retrying.

        Returns:
            bool: True if we should retry, False if we should stop
        """
        self.error_count += 1
        self.retry_count += 1

        self.last_error = {
            "error": error,
            "error_type": error_type,
            "timestamp": time.perf_counter(),
            "retry_count": self.retry_count,
            **(metadata or {})
        }

        # Log error with retry information
        logger.error(
            f"[{self.request_id}] Error {self.error_count} (retry {self.retry_count}/{self.max_retries}): {error_type} - {error}"
        )

        # Log to metrics file
        metrics_logger.info(
            f"request_id={self.request_id}, event=error, error_type={error_type}, "
            f"error_count={self.error_count}, retry_count={self.retry_count}, error={error}"
        )

        # Return whether we should continue retrying
        return self.retry_count < self.max_retries

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
            f"request_id={self.request_id}, event=model_request, elapsed={elapsed:.3f}, retry_count={self.retry_count}"
        )

    def record_tool_execution(self, tool_name: str, duration: float, success: bool, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record timing for a tool execution."""
        if tool_name not in self.tool_timings:
            self.tool_timings[tool_name] = {
                "count": 0,
                "total_time": 0.0,
                "successful_calls": 0,
                "failed_calls": 0
            }

        self.tool_timings[tool_name]["count"] += 1
        self.tool_timings[tool_name]["total_time"] += duration
        if success:
            self.tool_timings[tool_name]["successful_calls"] += 1
        else:
            self.tool_timings[tool_name]["failed_calls"] += 1

        self.total_tool_time += duration

        # Log tool execution
        log_data = {
            "request_id": self.request_id,
            "event": "tool_execution",
            "tool": tool_name,
            "duration": f"{duration:.3f}",
            "success": str(success),
            "total_tool_time": f"{self.total_tool_time:.3f}"
        }
        if metadata:
            log_data.update(metadata)

        metrics_logger.info(", ".join(f"{k}={v}" for k, v in log_data.items()))

    def record_first_audio(self, text_chunk: str) -> None:
        """Record when first audio chunk is played.

        Args:
            text_chunk: The text being spoken
        """
        if self.first_audio_time is None:
            self.first_audio_time = time.perf_counter()
            time_to_audio = self.first_audio_time - self.start_time
            time_from_first_response = self.first_audio_time - \
                (self.first_response_time or self.first_audio_time)

            # Log to console
            logger.info(
                f"[TIMING][{self.request_id}] First audio played after {time_to_audio:.3f}s "
                f"(+{time_from_first_response:.3f}s from first response)"
            )

            # Log to metrics file
            metrics_logger.info(
                f"request_id={self.request_id}, event=first_audio, "
                f"time_to_audio={time_to_audio:.3f}, "
                f"time_from_first_response={time_from_first_response:.3f}, "
                # Log first 50 chars of text being spoken
                f"text_chunk={text_chunk[:50]}"
            )

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all timing information."""
        total_duration = time.perf_counter() - self.start_time
        return {
            "request_id": self.request_id,
            "total_duration": total_duration,
            "total_tool_time": self.total_tool_time,
            "tool_timings": self.tool_timings,
            "first_response_time": self.first_response_time - self.start_time if self.first_response_time else None,
            "model_request_time": self.model_request_time - self.start_time if self.model_request_time else None,
            "first_audio_time": self.first_audio_time - self.start_time if self.first_audio_time else None,
            "error_count": self.error_count,
            "retry_count": self.retry_count,
            "last_error": self.last_error
        }


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
        # Get and log summary
        summary = profiler.get_summary()

        # Log to console
        logger.info(
            f"[TIMING][{request_id}] Request completed in {summary['total_duration']:.3f}s"
        )

        # Log detailed summary to metrics file
        metrics_logger.info(
            f"request_id={request_id}, event=request_end, " +
            ", ".join(f"{k}={v}" for k, v in summary.items()
                      if k != 'request_id')
        )


@asynccontextmanager
async def profile_operation(operation: str, profiler: Optional[RequestProfile] = None, request_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
    """Profile an operation's duration."""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        if profiler:
            profiler.log_operation(operation, start_time, end_time, metadata)
        else:
            duration = end_time - start_time
            log_prefix = f"[{request_id}] " if request_id else ""

            # Log to console
            logger.info(
                f"[TIMING]{log_prefix}Operation '{operation}' completed in {duration:.3f}s"
            )

            # Log to metrics file with metadata
            log_data = {
                "request_id": request_id or "unknown",
                "operation": operation,
                "duration": f"{duration:.3f}"
            }
            if metadata:
                log_data.update(metadata)

            metrics_logger.info(
                ", ".join(f"{k}={v}" for k, v in log_data.items()))
