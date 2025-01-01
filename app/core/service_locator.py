"""Service locator pattern to avoid circular dependencies."""

from typing import Any, Dict, Optional


class _ServiceLocator:
    """Simple service locator to avoid circular dependencies."""

    def __init__(self):
        self._services: Dict[str, Any] = {}

    def get_service(self, service_name: str) -> Optional[Any]:
        """Get a service by name.

        Args:
            service_name: Name of the service to retrieve

        Returns:
            The service instance if found, None otherwise
        """
        return self._services.get(service_name)

    def register_service(self, service_name: str, service: Any) -> None:
        """Register a service.

        Args:
            service_name: Name to register the service under
            service: The service instance to register
        """
        self._services[service_name] = service

    def clear(self) -> None:
        """Clear all registered services."""
        self._services.clear()

    def has_service(self, service_name: str) -> bool:
        """Check if a service is registered.

        Args:
            service_name: Name of the service to check

        Returns:
            True if the service is registered, False otherwise
        """
        return service_name in self._services


# Module-level singleton
_instance = None


def get_service_locator() -> _ServiceLocator:
    """Get the global service locator instance."""
    global _instance
    if _instance is None:
        _instance = _ServiceLocator()
    return _instance
