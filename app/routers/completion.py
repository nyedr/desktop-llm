"""Completion endpoints."""

import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Request, BackgroundTasks, Query, Depends
from pydantic import BaseModel, Field

from app.dependencies.providers import Providers
from app.services.model_service import ModelService
from app.core.config import config

logger = logging.getLogger(__name__)
router = APIRouter()


class GenerateCompletionForm(BaseModel):
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


@router.get("/models")
async def list_models(
    request: Request,
    page: int = Query(1, ge=1, description="Page number."),
    limit: int = Query(
        10, ge=1, le=100, description="Number of models per page."),
    model_service: ModelService = Depends(Providers.get_model_service)
) -> Dict[str, Any]:
    """Get available models with pagination."""
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


@router.post("/completions")
async def generate_completion(
    form_data: GenerateCompletionForm,
    request: Request,
    background_tasks: BackgroundTasks,
    model_service: ModelService = Depends(Providers.get_model_service)
) -> Dict[str, Any]:
    """Generate a completion using the specified model."""
    request_id = str(id(request))
    logger.debug(
        f"[{request_id}] Generating completion for prompt: {form_data.prompt[:100]}...")

    try:
        # Get model from form data or use default
        model = form_data.model or config.DEFAULT_MODEL

        # Check if model is available
        models = await model_service.get_all_models(request_id)
        if model not in models:
            logger.error(f"[{request_id}] Model {model} not available")
            raise ValueError(f"Model {model} not available")

        # Generate completion
        completion = await model_service.generate(
            prompt=form_data.prompt,
            model=model,
            temperature=form_data.temperature or config.MODEL_TEMPERATURE,
            max_tokens=form_data.max_tokens or config.MAX_TOKENS,
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
