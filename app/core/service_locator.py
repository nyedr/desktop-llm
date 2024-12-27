"""Service locator pattern to avoid circular dependencies."""

from typing import Any, Dict, Optional


class _ServiceLocator:
    """Simple service locator to avoid circular dependencies."""

    def __init__(self):
        self._services: Dict[str, Any] = {}

    def get_service(self, service_name: str) -> Any:
        """Get a service by name."""
        if service_name not in self._services:
            raise KeyError(f"Service {service_name} not registered")
        return self._services[service_name]

    def register_service(self, service_name: str, service: Any) -> None:
        """Register a service."""
        self._services[service_name] = service

    def get_mcp_service(self) -> Any:
        """Get the MCP service."""
        return self.get_service("mcp_service")


# Module-level singleton
_instance = None


def get_service_locator() -> _ServiceLocator:
    """Get the global service locator instance."""
    global _instance
    if _instance is None:
        _instance = _ServiceLocator()
    return _instance
