"""Health check endpoints."""

import logging
import psutil
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Request, Depends
from pydantic import BaseModel, Field

from app.dependencies.providers import get_model_service, get_function_service
from app.services.model_service import ModelService
from app.services.function_service import FunctionService
from app.functions.base import FunctionType


logger = logging.getLogger(__name__)
router = APIRouter()


class ServiceState(BaseModel):
    """Service state model."""
    status: str
    status_icon: str
    error: Optional[str] = None


class SystemMetrics(BaseModel):
    """System metrics model."""
    memory: Dict[str, Any] = Field(
        description="Memory metrics including total, available, and percent usage")
    disk: Dict[str, Any] = Field(
        description="Disk metrics including total, free, and percent usage")


class OllamaStatus(BaseModel):
    """Ollama status model."""
    connected: bool
    message: str
    base_url: str


class ModelsComponent(BaseModel):
    """Models component status."""
    status: str
    count: int
    available: List[str]
    ollama_status: OllamaStatus


class FunctionComponent(BaseModel):
    """Function component status."""
    status: str
    count: int
    registered: List[str]


class Components(BaseModel):
    """Components status model."""
    services: Dict[str, ServiceState]
    models: ModelsComponent
    tools: FunctionComponent
    filters: FunctionComponent
    pipelines: FunctionComponent


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    system: SystemMetrics
    components: Components
    error: Optional[str] = None


@router.get("/health", response_model=HealthResponse)
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
        tools = [f for f in functions if f.get(
            "type") == FunctionType.TOOL.value]
        filters = [f for f in functions if f.get(
            "type") == FunctionType.FILTER.value]
        pipelines = [f for f in functions if f.get(
            "type") == FunctionType.PIPELINE.value]

        # Get service states
        service_states = getattr(request.app.state, 'service_states', {})
        services_status = {}
        overall_status = "healthy"

        for service_name, state in service_states.items():
            service_info = {
                "status": state.status.name.lower(),
                "status_icon": state.status.value,
                "error": str(state.error) if state.error else None
            }
            services_status[service_name] = service_info

            # Update overall status if any critical service is down
            if service_name in ['mcp', 'chroma'] and state.status.name in ['FAILED', 'OFFLINE']:
                overall_status = "degraded"

        return {
            "status": overall_status,
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
                "services": services_status,
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
