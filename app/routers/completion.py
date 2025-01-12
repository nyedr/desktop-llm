"""Completion endpoints."""

import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Request, BackgroundTasks, Query, Depends
from pydantic import BaseModel, Field

from app.dependencies.providers import Providers
from app.services.model_service import ModelService
from app.core.config import config

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/completion",
    tags=["completion"],
    responses={
        400: {"description": "Bad request - Invalid input parameters"},
        500: {"description": "Internal server error"},
        429: {"description": "Too many requests - Rate limit exceeded"},
    }
)


class GenerateCompletionForm(BaseModel):
    """Request model for generating completions."""
    model: Optional[str] = Field(
        None, description="The model name to use for generation.")
    prompt: str = Field(...,
                        description="The prompt to generate a response for.")
    temperature: Optional[float] = Field(
        None, description="Sampling temperature for generation.")
    max_tokens: Optional[int] = Field(
        None, description="Maximum number of tokens to generate.")
    suffix: Optional[str] = Field(
        None, description="The text to append after the model response.")
    images: Optional[List[str]] = Field(
        None, description="A list of base64-encoded images for multimodal models.")
    task: Optional[str] = Field(
        None, description="Optional task type for prompt building.")


@router.get("/models",
            summary="List Available Models",
            description="""
    Get a paginated list of available language models.
    
    This endpoint returns information about all models that can be used for completions and chat,
    including their capabilities and configurations.
    """,
            response_description="Paginated list of available models",
            responses={
                200: {
                    "description": "Successful response",
                    "content": {
                        "application/json": {
                            "example": {
                                "data": [{"name": "model1", "type": "chat"}, {"name": "model2", "type": "completion"}],
                                "page": 1,
                                "limit": 10,
                                "total": 2
                            }
                        }
                    }
                }
            }
            )
async def list_models(
    request: Request,
    page: int = Query(1, ge=1, description="Page number."),
    limit: int = Query(
        10, ge=1, le=100, description="Number of models per page."),
    model_service: ModelService = Depends(Providers.get_model_service)
) -> Dict[str, Any]:
    """Get available models with pagination.

    Args:
        request: The FastAPI request object
        page: The page number to return (starts at 1)
        limit: The number of models per page (max 100)
        model_service: The model service for fetching model information

    Returns:
        A dictionary containing:
        - data: List of model information
        - page: Current page number
        - limit: Number of items per page
        - total: Total number of models

    Raises:
        HTTPException: If there are errors fetching models
    """
    request_id = str(id(request))
    logger.debug(
        f"[{request_id}] Fetching models page {page} with limit {limit}")

    try:
        # Get all models
        models = await model_service.get_all_models(request_id)
        models_list = list(models.values())

        # Apply pagination
        total = len(models_list)
        start = (page - 1) * limit
        end = start + limit
        paginated_models = models_list[start:end]

        return {
            "data": paginated_models,
            "page": page,
            "limit": limit,
            "total": total
        }

    except Exception as e:
        logger.error(f"[{request_id}] Error fetching models: {e}")
        raise


@router.post("/generate",
             summary="Generate Completion",
             description="""
    Generate a completion using a specified language model.
    
    Features:
    - Support for various language models
    - Configurable generation parameters
    - Optional image input for multimodal models
    - Task-specific prompt building
    """,
             response_description="Generated completion response",
             responses={
                 200: {
                     "description": "Successful response",
                     "content": {
                         "application/json": {
                             "example": {
                                 "model": "model-name",
                                 "choices": [{"text": "Generated completion text"}]
                             }
                         }
                     }
                 }
             }
             )
async def generate_completion(
    form_data: GenerateCompletionForm,
    request: Request,
    background_tasks: BackgroundTasks,
    model_service: ModelService = Depends(Providers.get_model_service)
) -> Dict[str, Any]:
    """Generate a completion using the specified model.

    Args:
        form_data: The completion request parameters
        request: The FastAPI request object
        background_tasks: FastAPI background tasks handler
        model_service: The model service for generating completions

    Returns:
        A dictionary containing:
        - model: The name of the model used
        - choices: List containing the generated completion

    Raises:
        HTTPException: If the model is not available or there are generation errors
        ValueError: If the specified model is not available
    """
    request_id = str(id(request))
    logger.debug(
        f"[{request_id}] Generating completion for prompt: {form_data.prompt[:100]}...")

    try:
        # Get model from form data or use default
        model = form_data.model or config.llm.model

        # Check if model is available
        models = await model_service.get_all_models(request_id)
        if model not in models:
            logger.error(f"[{request_id}] Model {model} not available")
            raise ValueError(f"Model {model} not available")

        # Generate completion
        completion = await model_service.generate(
            prompt=form_data.prompt,
            model=model,
            temperature=form_data.temperature or config.llm.temperature,
            max_tokens=form_data.max_tokens or config.llm.max_tokens,
            images=form_data.images
        )

        return {
            "model": model,
            "choices": [{
                "text": completion
            }]
        }

    except Exception as e:
        logger.error(f"[{request_id}] Error generating completion: {e}")
        raise
