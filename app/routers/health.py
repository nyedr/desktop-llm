"""Health check endpoints."""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Request, Depends
from pydantic import BaseModel, Field

from app.dependencies.providers import Providers
from app.services.model_service import ModelService
from app.services.function_service import FunctionService
from app.models.function_base import FunctionType


logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/health",
    tags=["health"],
    responses={
        500: {"description": "Internal server error"},
        503: {"description": "Service unavailable - Critical components offline"},
    }
)


class ServiceState(BaseModel):
    """Service state model."""
    status: str = Field(description="Current status of the service")
    status_icon: str = Field(
        description="Icon representing the service status")
    error: Optional[str] = Field(
        None, description="Error message if service is unhealthy")


class SystemMetrics(BaseModel):
    """System metrics model."""
    memory: Dict[str, Any] = Field(
        description="Memory metrics including total, available, and percent usage")
    disk: Dict[str, Any] = Field(
        description="Disk metrics including total, free, and percent usage")


class EndpointStatus(BaseModel):
    """Endpoint status model."""
    connected: bool = Field(description="Whether the endpoint is connected")
    message: str = Field(description="Status message or error details")
    base_url: str = Field(description="Base URL of the endpoint")


class ModelsComponent(BaseModel):
    """Models component status."""
    status: str = Field(description="Overall status of the models component")
    count: int = Field(description="Number of available models")
    available: List[str] = Field(description="List of available model names")
    endpoints_status: Dict[str, EndpointStatus] = Field(
        description="Status of model endpoints")


class FunctionComponent(BaseModel):
    """Function component status."""
    status: str = Field(description="Status of the function component")
    count: int = Field(description="Number of registered functions")
    registered: List[str] = Field(
        description="List of registered function names")


class Components(BaseModel):
    """Components status model."""
    services: Dict[str, ServiceState] = Field(
        description="Status of system services")
    models: ModelsComponent = Field(description="Status of model components")
    tools: FunctionComponent = Field(description="Status of tool functions")
    filters: FunctionComponent = Field(
        description="Status of filter functions")
    pipelines: FunctionComponent = Field(
        description="Status of pipeline functions")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(description="Overall health status")
    components: Dict[str, Any] = Field(
        description="Status of system components")


@router.get("",
            response_model=HealthResponse,
            summary="System Health Check",
            description="""
    Check the health status of all system components.
    
    This endpoint performs a comprehensive health check of:
    - Core services (LightRAG, ModelService, FunctionService)
    - Model endpoints (OpenAI, Ollama)
    - Function components (Tools, Filters, Pipelines)
    - System resources
    
    The response includes detailed status information for each component
    and an overall system health assessment.
    """,
            response_description="Detailed health status of all system components",
            responses={
                200: {
                    "description": "Health check completed successfully",
                    "content": {
                        "application/json": {
                            "example": {
                                "status": "healthy",
                                "components": {
                                    "services": {
                                        "lightrag": {"status": "ready", "status_icon": "✅", "error": None}
                                    },
                                    "models": {
                                        "status": "healthy",
                                        "count": 2,
                                        "available": ["gpt-3.5-turbo", "gpt-4"],
                                        "endpoints_status": {
                                            "openai": {"connected": True, "message": "Connected", "base_url": "https://api.openai.com"}
                                        }
                                    },
                                    "tools": {"status": "healthy", "count": 5, "registered": ["tool1", "tool2"]},
                                    "filters": {"status": "healthy", "count": 2, "registered": ["filter1"]},
                                    "pipelines": {"status": "healthy", "count": 1, "registered": ["pipeline1"]}
                                }
                            }
                        }
                    }
                }
            }
            )
async def health_check(
    request: Request,
    request_id: Optional[str] = None,
    model_service: ModelService = Depends(Providers.get_model_service),
    function_service: FunctionService = Depends(
        Providers.get_function_service)
) -> Dict[str, Any]:
    """Check system health.

    Args:
        request: The FastAPI request object
        request_id: Optional request ID for tracking
        model_service: The model service for checking LLM endpoints
        function_service: The function service for checking function components

    Returns:
        A dictionary containing:
        - Overall system status
        - Detailed component statuses
        - Service states
        - Model availability
        - Function registrations

    Raises:
        HTTPException: If critical components are unreachable
    """
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
            if service_name in ['lightrag'] and state.status.name in ['FAILED', 'OFFLINE']:
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
