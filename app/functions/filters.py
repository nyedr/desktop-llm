"""Filter-related functionality for the function system."""

import json
import logging
from typing import Dict, Any, List, Tuple, Union
from app.functions.base import Filter, FilterResponse
from app.functions.utils import validate_filter_response, create_error_response
from app.models.chat import ChatStreamEvent

logger = logging.getLogger(__name__)


async def apply_filters(
    filters: List[Filter],
    data: Dict[str, Any],
    request_id: str,
    direction: str = "inlet",
    as_event: bool = False,
    filter_name: str = "data_filters"
) -> Union[Tuple[Dict[str, Any], bool], ChatStreamEvent]:
    """Apply a list of filters to data.

    Args:
        filters: List of filters to apply
        data: Data to filter
        request_id: Request ID for logging
        direction: Direction of filtering ("inlet" or "outlet")
        as_event: Whether to return result as a ChatStreamEvent (for outlet filtering)
        filter_name: Name of the filter group for logging and error reporting

    Returns:
        If as_event is False: Tuple of (filtered data, success flag)
        If as_event is True: ChatStreamEvent with filtered message or error
    """
    try:
        # Ensure we're working with a dictionary
        if hasattr(data, "model_dump"):
            data = data.model_dump()

        # Sort filters by priority (ascending for inlet, descending for outlet)
        sorted_filters = sorted(
            filters,
            key=lambda f: f.priority,
            reverse=(direction == "outlet")
        )

        filtered_data = data
        for filter_obj in sorted_filters:
            try:
                if direction == "inlet":
                    new_data = await filter_obj.inlet(filtered_data)
                else:
                    new_data = await filter_obj.outlet(filtered_data)

                # Create and validate filter response
                filter_response = FilterResponse(
                    success=True,
                    modified_data=new_data,
                    filter_name=filter_obj.name,
                    changes_made=new_data != filtered_data
                )
                validate_filter_response(filter_response)
                filtered_data = filter_response.modified_data

            except Exception as e:
                logger.error(
                    f"[{request_id}] Error in filter {filter_obj.name}: {e}")
                error = create_error_response(
                    error=e,
                    function_type="filter",
                    function_name=filter_obj.name
                )
                if as_event:
                    return ChatStreamEvent(
                        event="error",
                        data=json.dumps({"error": error.error})
                    )
                return data, False

        if as_event:
            if "content" in filtered_data:
                print(filtered_data['content'], end="", flush=True)
            return ChatStreamEvent(
                event="message",
                data=json.dumps(filtered_data)
            )
        return filtered_data, True

    except Exception as e:
        logger.error(
            f"[{request_id}] Error applying filters in {filter_name}: {e}")
        error = create_error_response(
            error=e,
            function_type="filter",
            function_name=filter_name
        )
        if as_event:
            return ChatStreamEvent(
                event="error",
                data=json.dumps({"error": error.error})
            )
        return data, False
