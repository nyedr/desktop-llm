"""Pipeline functions that handle complex workflows."""

from app.functions.types.pipelines.multi_step import MultiStepPipeline
from app.functions.registry import registry

# Register pipeline class
registry.register(MultiStepPipeline)

__all__ = ['MultiStepPipeline']
