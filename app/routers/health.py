"""Health check endpoints."""

import logging
import psutil
from typing import Dict, Any
from fastapi import APIRouter, Request, Depends

from app.dependencies.providers import get_model_service, get_function_service
from app.services.model_service import ModelService
from app.services.function_service import FunctionService

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
        
        # Check component health
        models = await model_service.get_all_models(request_id)
        functions = function_service.list_functions()
        
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
                    "status": "healthy" if models else "unhealthy",
                    "count": len(models),
                    "available": list(models.keys()) if models else []
                },
                "functions": {
                    "status": "healthy",
                    "count": len(functions),
                    "registered": [f["name"] for f in functions]
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }
