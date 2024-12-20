"""Main FastAPI application."""

import logging
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from sse_starlette.sse import EventSourceResponse

from app.core.config import config
from app.dependencies.providers import Providers
from app.routers import chat, functions, completion, health
from app.functions import registry
from app.functions.types.tools import WeatherTool
from app.functions.types.filters import TextModifierFilter
from app.functions.types.pipelines import MultiStepPipeline

# Configure logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

def register_default_functions(function_service):
    """Register default functions during startup."""
    try:
        # Register built-in tools
        function_service.register_function(
            name="get_current_weather",
            function_class=WeatherTool
        )
        
        # Register filters
        function_service.register_function(
            name="text_modifier",
            function_class=TextModifierFilter
        )
        
        # Register pipelines
        function_service.register_function(
            name="multi_step_processor",
            function_class=MultiStepPipeline
        )
        
        logger.info("Successfully registered default functions")
    except Exception as e:
        logger.error(f"Error registering default functions: {e}", exc_info=True)
        raise

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for FastAPI application."""
    logger.info("Starting application initialization...")
    
    try:
        # Initialize providers
        model_service = Providers.get_model_service()
        function_service = Providers.get_function_service()
        agent = Providers.get_agent()
        
        # Register default functions
        register_default_functions(function_service)
        
        # Load initial models
        request_id = str(uuid.uuid4())
        await model_service.get_all_models(request_id)
        
        logger.info("Application initialization complete")
        yield
        
    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)
        raise
    finally:
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
        logger.error(f"[{request_id}] Error processing request: {e}", exc_info=True)
        raise

# Include routers with API prefix
api_prefix = "/api/v1"
app.include_router(chat.router, prefix=api_prefix, tags=["chat"])
app.include_router(functions.router, prefix=api_prefix, tags=["functions"])
app.include_router(completion.router, prefix=api_prefix, tags=["completion"])
app.include_router(health.router, prefix=api_prefix, tags=["health"])

@app.on_event("startup")
async def startup_event():
    """Initialize app state on startup."""
    try:
        logger.info("Running startup event")
        # Any additional startup tasks can be added here
    except Exception as e:
        logger.error(f"Error during startup event: {e}", exc_info=True)
        raise
