"""Pipeline functions that handle complex workflows."""

from app.functions.types.pipelines.multi_step import MultiStepPipeline
from app.functions.registry import registry

# Register pipeline instances
multi_step = MultiStepPipeline()
registry.register(multi_step)

__all__ = ['MultiStepPipeline']
