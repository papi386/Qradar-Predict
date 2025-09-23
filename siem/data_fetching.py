import requests
from datetime import datetime, timedelta
from typing import List

def fetch_recent_offense_ids(ip: str, api_key: str, hours: int = 3) -> List[int]:
    """
    Fetch offense IDs from QRadar in the last `hours` (default = 3).

    Args:
        ip (str): QRadar host IP
        api_key (str): QRadar API token
        hours (int): Number of hours to look back

    Returns:
        list[int]: List of offense IDs
    """
    base_url = f"https://{ip}/api/siem/offenses"
    headers = {
        'SEC': api_key,
        'Version': '16.0',
        'Accept': 'application/json'
    }

    # timestamp for last X hours in ms
    since_ts = int((datetime.utcnow() - timedelta(hours=hours)).timestamp() * 1000)

    params = {
        "filter": f"start_time >= {since_ts}"
    }

    try:
        response = requests.get(base_url, headers=headers, params=params, verify=False, timeout=30)
        response.raise_for_status()
        offenses = response.json()
        # Extract just the IDs
        return [o.get("id") for o in offenses if "id" in o]

    except requests.exceptions.RequestException as e:
        print(f"Error fetching offenses: {e}")
        return []
