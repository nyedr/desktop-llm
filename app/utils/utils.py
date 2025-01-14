import logging
from datetime import datetime, timedelta
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def format_timestamp(time_diff: timedelta) -> str:
    """Format a timedelta into a human-readable string.

    Args:
        time_diff: The time difference to format

    Returns:
        A human-readable string like "2 days ago", "3 hours ago", etc.
    """
    total_seconds = int(time_diff.total_seconds())

    days = total_seconds // (24 * 3600)
    remaining_seconds = total_seconds % (24 * 3600)
    hours = remaining_seconds // 3600
    remaining_seconds %= 3600
    minutes = remaining_seconds // 60
    seconds = remaining_seconds % 60

    if days >= 365:
        years = days // 365
        return f"{years} {'year' if years == 1 else 'years'} ago"
    elif days >= 30:
        months = days // 30
        return f"{months} {'month' if months == 1 else 'months'} ago"
    elif days > 7:
        weeks = days // 7
        return f"{weeks} {'week' if weeks == 1 else 'weeks'} ago"
    elif days > 0:
        return f"{days} {'day' if days == 1 else 'days'} ago"
    elif hours > 0:
        if minutes > 0:
            return f"{hours} {'hour' if hours == 1 else 'hours'} and {minutes} {'minute' if minutes == 1 else 'minutes'} ago"
        return f"{hours} {'hour' if hours == 1 else 'hours'} ago"
    elif minutes > 0:
        if seconds > 0:
            return f"{minutes} {'minute' if minutes == 1 else 'minutes'} and {seconds} {'second' if seconds == 1 else 'seconds'} ago"
        return f"{minutes} {'minute' if minutes == 1 else 'minutes'} ago"
    else:
        return f"{seconds} {'second' if seconds == 1 else 'seconds'} ago" if seconds > 0 else "just now"


def format_timestamp_date(timestamp: datetime) -> str:
    return f"{timestamp.strftime('%I:%M:%S%p').lower()} on {timestamp.strftime('%B %d, %Y')}"


def parse_attempt_completion(content: str) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
    """Parse attempt completion XML from content.

    Args:
        content: The content to parse

    Returns:
        Tuple of:
        - bool: Whether completion XML was found
        - str: The result text if found, None otherwise
        - str: The command if found, None otherwise 
        - str: The answer if found, None otherwise
    """
    try:
        if "<attempt_completion>" not in content or "</attempt_completion>" not in content:
            return False, None, None, None

        # Extract result
        result_start = content.find("<result>") + 10
        result_end = content.find("</result>")
        if result_start == -1 or result_end == -1:
            return False, None, None, None
        result = content[result_start:result_end].strip()

        # Extract optional command
        command = None
        if "<command>" in content and "</command>" in content:
            command_start = content.find("<command>") + 9
            command_end = content.find("</command>")
            command = content[command_start:command_end].strip()

        # Extract optional answer
        answer = None
        if "<answer>" in content and "</answer>" in content:
            answer_start = content.find("<answer>") + 8
            answer_end = content.find("</answer>")
            answer = content[answer_start:answer_end].strip()

        return True, result, command, answer

    except Exception as e:
        logger.error(f"Error parsing completion XML: {str(e)}")
        return False, None, None, None
