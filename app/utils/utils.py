from datetime import datetime


def format_timestamp(timestamp: datetime) -> str:
    if timestamp.days > 365:
        years = timestamp.days // 365
        timestamp_str = f"{years} {'year' if years == 1 else 'years'} ago"
    elif timestamp.days > 30:
        months = timestamp.days // 30
        timestamp_str = f"{months} {'month' if months == 1 else 'months'} ago"
    elif timestamp.days > 0:
        timestamp_str = f"{timestamp.days} {'day' if timestamp.days == 1 else 'days'} ago"
    elif timestamp.seconds > 3600:
        hours = timestamp.seconds // 3600
        timestamp_str = f"{hours} {'hour' if hours == 1 else 'hours'} ago"
    elif timestamp.seconds > 60:
        minutes = timestamp.seconds // 60
        timestamp_str = f"{minutes} {'minute' if minutes == 1 else 'minutes'} ago"
    else:
        timestamp_str = "just now"

    return timestamp_str


def format_timestamp_date(timestamp: datetime) -> str:
    return f"{timestamp.strftime('%I:%M:%S%p').lower()} on {timestamp.strftime('%B %d, %Y')}"
