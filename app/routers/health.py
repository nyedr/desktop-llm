"""Health check endpoints."""

import logging
import psutil
from typing import Dict, Any
from fastapi import APIRouter, Request, Depends

from app.dependencies.providers import get_model_service, get_function_service
from app.services.model_service import ModelService
from app.services.function_service import FunctionService
from app.functions.base import FunctionType

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
async def health_check(
    request: Request,
    model_service: ModelService = Depends(get_model_service),
    function_service: FunctionService = Depends(get_function_service)
) -> Dict[str, Any]:
    """Check the health of the API and its components."""
    request_id = str(id(request))
    try:
        # Get system metrics
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # Check Ollama health first
        ollama_healthy, ollama_status = await model_service.check_ollama_health(request_id)

        # Get available models if Ollama is healthy
        models = await model_service.get_all_models(request_id) if ollama_healthy else {}

        # Get registered functions
        functions = function_service.list_functions()

        # Separate functions by type
        tools = [f for f in functions if f.get("type") == FunctionType.TOOL]
        filters = [f for f in functions if f.get(
            "type") == FunctionType.FILTER]
        pipelines = [f for f in functions if f.get(
            "type") == FunctionType.PIPELINE]

        return {
            "status": "healthy",
            "system": {
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent
                },
                "disk": {
                    "total": disk.total,
                    "free": disk.free,
                    "percent": disk.percent
                }
            },
            "components": {
                "models": {
                    "status": "healthy" if ollama_healthy else "unhealthy",
                    "count": len(models),
                    "available": list(models.keys()) if models else [],
                    "ollama_status": {
                        "connected": ollama_healthy,
                        "message": ollama_status,
                        "base_url": model_service.base_url
                    }
                },
                "tools": {
                    "status": "healthy",
                    "count": len(tools),
                    "registered": [t.get("name") for t in tools]
                },
                "filters": {
                    "status": "healthy",
                    "count": len(filters),
                    "registered": [f.get("name") for f in filters]
                },
                "pipelines": {
                    "status": "healthy",
                    "count": len(pipelines),
                    "registered": [p.get("name") for p in pipelines]
                }
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "error": str(e)
        }
