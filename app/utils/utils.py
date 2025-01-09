from datetime import datetime, timedelta


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

    if days >= 365:
        years = days // 365
        return f"{years} {'year' if years == 1 else 'years'} ago"
    elif days >= 30:
        months = days // 30
        return f"{months} {'month' if months == 1 else 'months'} ago"
    elif days > 0:
        return f"{days} {'day' if days == 1 else 'days'} ago"
    elif hours > 0:
        return f"{hours} {'hour' if hours == 1 else 'hours'} ago"
    elif minutes > 0:
        return f"{minutes} {'minute' if minutes == 1 else 'minutes'} ago"
    else:
        return "just now"


def format_timestamp_date(timestamp: datetime) -> str:
    return f"{timestamp.strftime('%I:%M:%S%p').lower()} on {timestamp.strftime('%B %d, %Y')}"
