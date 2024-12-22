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
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from sse_starlette.sse import EventSourceResponse
from fastapi.responses import JSONResponse

from app.core.config import config
from app.dependencies.providers import Providers
from app.routers import chat, functions, completion, health
from app.functions.registry import function_registry
from app.services.model_service import ModelService
from app.services.function_service import FunctionService
from app.functions.registry import FunctionRegistry

# Configure logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)


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

            logger.info("Function discovery complete")
            functions = function_registry.list_functions()
            logger.info(f"Registered functions: {functions}")
        except Exception as e:
            logger.error(f"Error discovering functions: {e}", exc_info=True)
            raise

        # Remove workspace root from Python path
        if workspace_root in sys.path:
            sys.path.remove(workspace_root)
            logger.info(
                f"Removed workspace root from Python path: {workspace_root}")

    except Exception as e:
        logger.error(f"Error registering functions: {e}", exc_info=True)
        raise


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for FastAPI application."""
    mcp_service = None
    chroma_service = None
    langchain_service = None

    try:
        # Initialize services
        logger.info("Starting service initialization...")

        # Initialize MCP service first
        try:
            mcp_service = Providers.get_mcp_service()
            logger.info("Initializing MCP service...")
            await mcp_service.initialize()
            logger.info("MCP service initialized successfully")
        except Exception as e:
            logger.error(
                f"Failed to initialize MCP service: {e}", exc_info=True)
            raise

        # Initialize Chroma service
        try:
            chroma_service = Providers.get_chroma_service()
            logger.info("Initializing Chroma service...")
            await chroma_service.initialize()
            logger.info("Chroma service initialized successfully")
        except Exception as e:
            logger.error(
                f"Failed to initialize Chroma service: {e}", exc_info=True)
            raise

        # Initialize LangChain service
        try:
            langchain_service = Providers.get_langchain_service()
            logger.info("Initializing LangChain service...")
            await langchain_service.initialize(chroma_service, mcp_service)
            logger.info("LangChain service initialized successfully")
        except Exception as e:
            logger.error(
                f"Failed to initialize LangChain service: {e}", exc_info=True)
            raise

        # Register functions
        try:
            function_service = Providers.get_function_service()
            logger.info(
                "Got function service, registering default functions...")
            await register_default_functions(function_service)
            logger.info("Functions registered successfully")
        except Exception as e:
            logger.error(
                f"Failed to register functions: {e}", exc_info=True)
            raise

        logger.info("All services initialized successfully")
        yield

    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)
        raise
    finally:
        # Clean up services in reverse order
        logger.info("Starting service cleanup...")

        if langchain_service:
            try:
                # Add cleanup for langchain service if needed
                pass
            except Exception as e:
                logger.error(f"Error cleaning up LangChain service: {e}")

        if chroma_service:
            try:
                # Add cleanup for chroma service if needed
                pass
            except Exception as e:
                logger.error(f"Error cleaning up Chroma service: {e}")

        if mcp_service:
            try:
                await mcp_service.close_session()
            except Exception as e:
                logger.error(f"Error cleaning up MCP service: {e}")

        logger.info("Application shutdown complete")

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
