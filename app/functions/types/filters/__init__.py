"""Filter functions that process inlet/outlet data."""

from app.functions.types.filters.text_modifier import TextModifierFilter
from app.functions.registry import registry

# Register filter instances
text_modifier = TextModifierFilter()
registry.register(text_modifier)

__all__ = ['TextModifierFilter']
