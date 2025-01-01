"""Main FastAPI application."""

# Configure event loop policy for Windows - this must be done before anything else
import sys
import asyncio
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import logging
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from enum import Enum

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from fastapi.responses import JSONResponse

from app.dependencies.providers import Providers
from app.routers import chat, functions, completion, health

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Set specific loggers to higher levels to reduce noise
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('chromadb').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)


class ServiceStatus(Enum):
    """Service status enum."""
    UNINITIALIZED = "⚪"  # White circle
    INITIALIZING = "🔄"   # Rotating arrows
    READY = "✅"          # Green checkmark
    FAILED = "❌"         # Red X
    OFFLINE = "⭕"        # Red circle


class ServiceState:
    """Service state tracking."""

    def __init__(self):
        self.status = ServiceStatus.UNINITIALIZED
        self.error = None
        self.service = None

    def set_status(self, status: ServiceStatus, error: Exception = None):
        """Update service status and error state."""
        if self.status != status:  # Only log status changes
            self.status = status
            logger.info(f"Service status changed to: {status.value}")
        if error:
            logger.error(f"Service error: {error}")
            self.error = error


async def initialize_service(service_name: str, get_service_fn, state: ServiceState) -> bool:
    """Initialize a service and update its state."""
    try:
        state.set_status(ServiceStatus.INITIALIZING)
        service = get_service_fn()
        state.service = service

        if hasattr(service, 'initialize'):
            if isinstance(service, Providers.get_lightrag_manager().__class__):
                # Initialize LightRAG manager
                await service.initialize()
                if not getattr(service, '_initialized', False):
                    raise RuntimeError("LightRAG manager failed to initialize")
            else:
                await service.initialize()

        state.set_status(ServiceStatus.READY)
        logger.info(
            f"{service_name} initialized with status: {state.status.value}")
        return True
    except Exception as e:
        state.set_status(ServiceStatus.FAILED, e)
        return False

# Initialize service states
service_states = {
    "model": ServiceState(),
    "function": ServiceState(),
    "lightrag": ServiceState(),
    "mcp": ServiceState(),
    "agent": ServiceState()
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for FastAPI application."""
    app.state.service_states = service_states

    try:
        # Initialize base services first
        await initialize_service("LightRAG", Providers.get_lightrag_manager, service_states['lightrag'])
        await initialize_service("MCP", Providers.get_mcp_service, service_states['mcp'])

        # Initialize model and function services
        await initialize_service("Model", Providers.get_model_service, service_states['model'])
        await initialize_service("Function", Providers.get_function_service, service_states['function'])

        # Initialize agent last since it depends on other services
        await initialize_service("Agent", Providers.get_agent, service_states['agent'])

        # Log final service states
        for service_name, state in service_states.items():
            logger.info(f"{service_name} service status: {state.status.value}")

        yield

    finally:
        # Cleanup services in reverse order
        logger.info("Cleaning up services...")
        for service_name in reversed(list(service_states.keys())):
            state = service_states[service_name]
            if state.service and state.status == ServiceStatus.READY:
                try:
                    if hasattr(state.service, 'cleanup'):
                        await state.service.cleanup()
                    elif hasattr(state.service, 'close_session'):
                        await state.service.close_session()
                    state.set_status(ServiceStatus.OFFLINE)
                except Exception as e:
                    logger.error(
                        f"Error cleaning up {service_name} service: {e}")
                    state.set_status(ServiceStatus.FAILED, e)
        logger.info("Shutdown complete")

# Initialize FastAPI app
app = FastAPI(
    title="Desktop LLM API",
    description="API for interacting with local and remote language models.",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting middleware
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add a unique request ID to each request."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    request_id = getattr(request.state, "request_id", None)
    logger.info(f"[{request_id}] {request.method} {request.url}")

    try:
        response = await call_next(request)
        logger.info(f"[{request_id}] {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"[{request_id}] Error processing request: {e}")
        raise

# Include routers with API prefix
api_prefix = "/api/v1"
app.include_router(chat.router, prefix=api_prefix, tags=["chat"])
app.include_router(functions.router, prefix=api_prefix, tags=["functions"])
app.include_router(completion.router, prefix=api_prefix, tags=["completion"])
app.include_router(health.router, prefix=api_prefix, tags=["health"])


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": str(exc)}
    )
