import logging
import sys
from typing import Optional
import structlog
from datetime import datetime


def _filter_redundant_messages(_, __, event_dict):
    """Filter out redundant log messages."""
    redundant_phrases = [
        "Service status updated",
        "Current functions before",
        "Checking attribute",
        "Class details",
        "Module attributes",
        "Processing module",
        "Registering function",
        "Current registered functions",
        "Starting registration",
        "Successfully registered",
        "Found BaseFunction subclass",
        "Skipping base class",
        "Adding to Python path",
        "Removed from Python path",
        "Final Python path",
        "Discovery complete",
        "Initializing Chroma Service",
        "Initializing LangChain Service",
        "Initializing MCP Service",
        "Initializing Model Service",
        "Initializing FunctionService",
        "Original Python path",
        "Updated Python path",
        "Final Python path",
        "Found global node_modules",
        "Processing function config",
        "Function config",
        "Added to Python path",
        "Restored original Python path"
    ]

    if any(phrase in event_dict.get("event", "") for phrase in redundant_phrases):
        return None
    return event_dict


def _minimize_verbose_logs(_, __, event_dict):
    """Minimize verbose log messages."""
    message = event_dict.get("event", "")

    # Minimize Python path logs
    if "Python path" in message and isinstance(message, str):
        paths = message.split("\n")
        if len(paths) > 3:
            event_dict["event"] = f"{paths[0]}\n  ... ({len(paths)-2} more paths) ...\n{paths[-1]}"

    # Minimize config dumps
    if isinstance(message, str) and len(message) > 500:
        event_dict["event"] = f"{message[:250]}... (truncated)"

    return event_dict


def _format_service_summary(_, __, event_dict):
    """Format service initialization summary."""
    if event_dict.get("event") == "service_summary":
        services = event_dict.get("services", {})
        if services:
            summary = "\nService Status Summary:\n"
            for service, status in services.items():
                summary += f"- {service.upper()}: {status}\n"
            event_dict["event"] = summary.strip()
        else:
            return None
    return event_dict


def setup_logging(
    level: str = "INFO",
    request_id: Optional[str] = None,
    environment: str = "development"
) -> None:
    """Configure structured logging for the application.

    Args:
        level: Default log level (INFO, DEBUG, WARNING, ERROR)
        request_id: Optional request ID for request tracing
        environment: Environment name (development, production)
    """

    # Component-specific log levels
    component_levels = {
        "app.services": "INFO",
        "app.functions": "WARNING",
        "app.routers": "INFO",
        "app.main": "INFO",
        "chromadb": "WARNING",
        "urllib3": "WARNING",
        "sentence_transformers": "WARNING",
        "asyncio": "WARNING"
    }

    # Configure structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        _filter_redundant_messages,
        _minimize_verbose_logs,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    # Add summary processor for service initialization
    processors.append(_format_service_summary)

    # Use pretty formatting in development, JSON in production
    if environment == "development":
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        logger_factory=structlog.PrintLoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(level)),
        cache_logger_on_first_use=True,
    )

    # Set component-specific log levels
    for component, level in component_levels.items():
        logging.getLogger(component).setLevel(logging.getLevelName(level))

    # Create a logger with context
    logger = structlog.get_logger()
    logger = logger.bind(
        environment=environment,
        request_id=request_id,
        version="1.0.0"
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
