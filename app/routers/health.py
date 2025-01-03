"""Health check endpoints."""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Request, Depends
from pydantic import BaseModel, Field

from app.dependencies.providers import Providers
from app.services.model_service import ModelService
from app.services.function_service import FunctionService
from app.models.function import FunctionType


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


class EndpointStatus(BaseModel):
    """Endpoint status model."""
    connected: bool
    message: str
    base_url: str


class ModelsComponent(BaseModel):
    """Models component status."""
    status: str
    count: int
    available: List[str]
    endpoints_status: Dict[str, EndpointStatus]


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
    status: str = Field(description="Overall health status")
    components: Dict[str, Any] = Field(
        description="Status of system components")


@router.get("/health", response_model=HealthResponse)
async def health_check(
    request: Request,
    request_id: Optional[str] = None,
    model_service: ModelService = Depends(Providers.get_model_service),
    function_service: FunctionService = Depends(
        Providers.get_function_service)
) -> Dict[str, Any]:
    """Check system health."""
    try:
        # Check endpoints health
        endpoints_health = await model_service.check_health(request_id)

        # Get available models if OpenAI endpoint is healthy
        models = await model_service.get_models() if endpoints_health.get("openai", (False, ""))[0] else []

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
            if service_name in ['mcp', 'lightrag'] and state.status.name in ['FAILED', 'OFFLINE']:
                overall_status = "degraded"

        return {
            "status": overall_status,
            "components": {
                "services": services_status,
                "models": {
                    "status": "healthy" if endpoints_health.get("openai", (False, ""))[0] else "unhealthy",
                    "count": len(models),
                    "available": models,
                    "endpoints_status": {
                        endpoint: {
                            "connected": status[0],
                            "message": status[1],
                            "base_url": model_service.base_url if endpoint == "ollama" else str(model_service.client.base_url)
                        }
                        for endpoint, status in endpoints_health.items()
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
            "components": {
                "services": {},
                "models": {
                    "status": "unhealthy",
                    "count": 0,
                    "available": [],
                    "endpoints_status": {}
                },
                "tools": {"status": "unknown", "count": 0, "registered": []},
                "filters": {"status": "unknown", "count": 0, "registered": []},
                "pipelines": {"status": "unknown", "count": 0, "registered": []}
            }
        }
