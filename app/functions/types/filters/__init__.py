"""Filter functions that process inlet/outlet data."""

from app.functions.types.filters.text_modifier import TextModifierFilter
from app.functions.registry import registry

# Register filter class
registry.register(TextModifierFilter)

__all__ = ['TextModifierFilter']
