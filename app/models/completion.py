"""Completion response models."""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel


class CompletionResponse(BaseModel):
    """Response from a completion request."""
    model: str
    created_at: str
    response: str
    done: bool
    context: Optional[List[int]] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
