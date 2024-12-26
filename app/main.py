"""Main FastAPI application."""

# Configure event loop policy for Windows - this must be done before anything else
import sys
import asyncio
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import logging
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict
from pathlib import Path
from enum import Enum

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from fastapi.responses import JSONResponse

from app.core.config import config
from app.dependencies.providers import Providers
from app.routers import chat, functions, completion, health
from app.functions.registry import function_registry

# Configure logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
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
logging.getLogger('app.functions.registry').setLevel(logging.INFO)

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
        self.status = status
        self.error = error
        logger.info(f"Service status updated: {self.status.value}")
        if error:
            logger.error(f"Service error: {error}", exc_info=True)


async def initialize_service(service_name: str, get_service_fn, state: ServiceState) -> bool:
    """Initialize a service and update its state."""
    try:
        state.set_status(ServiceStatus.INITIALIZING)
        service = get_service_fn()
        state.service = service

        if hasattr(service, 'initialize'):
            await service.initialize()

        state.set_status(ServiceStatus.READY)
        logger.info(
            f"{service_name} initialized successfully {ServiceStatus.READY.value}")
        return True
    except Exception as e:
        state.set_status(ServiceStatus.FAILED, e)
        logger.error(
            f"{service_name} initialization failed {ServiceStatus.FAILED.value}: {e}", exc_info=True)
        return False


async def register_default_functions(function_service):
    """Register default functions during startup."""
    try:
        # Add workspace root to Python path
        workspace_root = str(Path(__file__).parent.parent)
        logger.info(f"Adding workspace root to Python path: {workspace_root}")
        if workspace_root not in sys.path:
            sys.path.insert(0, workspace_root)
            logger.debug(f"Updated Python path: {sys.path}")

        try:
            # Discover all function types
            base_dir = Path(__file__).parent

            # Discover tools
            tools_dir = base_dir / "functions" / "types" / "tools"
            await function_registry.discover_functions(tools_dir)

            # Discover filters
            filters_dir = base_dir / "functions" / "types" / "filters"
            await function_registry.discover_functions(filters_dir)

            # Discover pipelines
            pipelines_dir = base_dir / "functions" / "types" / "pipelines"
            await function_registry.discover_functions(pipelines_dir)

            logger.info("Function discovery complete ✅")
            functions = function_registry.list_functions()
            logger.info(f"Registered functions: {functions}")
        except Exception as e:
            logger.error(f"Error discovering functions ❌: {e}", exc_info=True)
            raise

        # Remove workspace root from Python path
        if workspace_root in sys.path:
            sys.path.remove(workspace_root)
            logger.info(
                f"Removed workspace root from Python path: {workspace_root}")

    except Exception as e:
        logger.error(f"Error registering functions ❌: {e}", exc_info=True)
        raise


async def cleanup_services(service_states: Dict[str, ServiceState], is_shutting_down: bool) -> None:
    """Clean up services in reverse order."""
    logger.info("Starting service cleanup...")
    for service_name, state in reversed(list(service_states.items())):
        if state.service and state.status == ServiceStatus.READY:
            try:
                if hasattr(state.service, 'close_session'):
                    await state.service.close_session()
                if is_shutting_down:
                    state.set_status(ServiceStatus.OFFLINE)
            except Exception as e:
                logger.error(f"Error cleaning up {service_name} service: {e}")
                if is_shutting_down:
                    state.set_status(ServiceStatus.FAILED, e)
    logger.info("Application shutdown complete")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for FastAPI application."""
    # Initialize service states
    service_states = {
        'mcp': ServiceState(),
        'chroma': ServiceState(),
        'langchain': ServiceState()
    }
    app.state.service_states = service_states

    try:
        logger.info("Starting service initialization...")

        # Initialize MCP service
        mcp_success = await initialize_service(
            "MCP service",
            Providers.get_mcp_service,
            service_states['mcp']
        )

        # Initialize Chroma service
        chroma_success = await initialize_service(
            "Chroma service",
            Providers.get_chroma_service,
            service_states['chroma']
        )

        # Initialize LangChain service if dependencies are available
        if chroma_success:
            langchain_state = service_states['langchain']
            try:
                langchain_state.set_status(ServiceStatus.INITIALIZING)
                langchain_service = Providers.get_langchain_service()
                langchain_state.service = langchain_service

                # Only use MCP service if it's available
                mcp_service = service_states['mcp'].service if mcp_success else None

                await langchain_service.initialize(
                    service_states['chroma'].service,
                    mcp_service
                )
                langchain_state.set_status(ServiceStatus.READY)
                logger.info(
                    f"LangChain service initialized successfully {ServiceStatus.READY.value}")
            except Exception as e:
                langchain_state.set_status(ServiceStatus.FAILED, e)
                logger.error(
                    f"LangChain service initialization failed {ServiceStatus.FAILED.value}: {e}", exc_info=True)

        # Register functions
        try:
            function_service = Providers.get_function_service()
            logger.info(
                "Got function service, registering default functions...")
            await register_default_functions(function_service)
            logger.info(
                f"Functions registered successfully {ServiceStatus.READY.value}")
        except Exception as e:
            logger.error(
                f"Failed to register functions {ServiceStatus.FAILED.value}: {e}", exc_info=True)

        # Log final service states
        logger.info("\nService Status Summary:")
        for service_name, state in service_states.items():
            status_msg = f"{service_name.upper()}: {state.status.value}"
            if state.error:
                status_msg += f" - Error: {str(state.error)}"
            logger.info(status_msg)

        try:
            yield
        except asyncio.CancelledError:
            logger.debug("Application received cancellation signal")
            raise
        except Exception as e:
            logger.error(
                f"Error during application runtime: {e}", exc_info=True)
            raise

    except asyncio.CancelledError:
        logger.debug("Application startup cancelled")
        raise
    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)
        raise
    finally:
        try:
            # Check if we're actually shutting down or just reloading
            is_shutting_down = not getattr(app.state, "reload", True)
            await cleanup_services(service_states, is_shutting_down)
        except asyncio.CancelledError:
            logger.debug("Cleanup cancelled")
            raise
        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)

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
        logger.error(
            f"[{request_id}] Error processing request: {e}", exc_info=True)
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
    logger.error(f"Global exception handler caught: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": str(exc)}
    )
