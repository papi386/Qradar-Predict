
import requests
import time
import ipaddress
import re
from datetime import datetime, timedelta



#config
QRADAR_HOST = "PUT YOUR QRADAR HOST"
API_TOKEN = "PUT YOUR API TOKEN"
API_VERSION = '16.0'
SEARCH_WINDOW_MINUTES = 10


def get_qradar_headers():
    """Return headers for QRadar API requests."""
    return {'SEC': API_TOKEN, 'Version': API_VERSION, 'Accept': 'application/json'}


def get_offense_details(offense_id: int):
    """Retrieve offense details from QRadar API."""
    url = f"{QRADAR_HOST}/api/siem/offenses/{offense_id}"
    headers = get_qradar_headers()
    try:
        response = requests.get(url, headers=headers, verify=False, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"Erreur : Offense #{offense_id} introuvable.")
        else:
            print(f"Erreur HTTP : {e}")
        return None
    except Exception as e:
        print(f"Erreur lors de la récupération de l'offense: {e}")
        return None


def run_aql_query(aql_query: str):
    """Execute an AQL query and return results."""
    headers = get_qradar_headers()
    search_endpoint = f"{QRADAR_HOST}/api/ariel/searches"
    try:
        response = requests.post(
            search_endpoint,
            headers=headers,
            params={'query_expression': aql_query},
            verify=False,
            timeout=60
        )
        response.raise_for_status()
        search_id = response.json().get('search_id')
        if not search_id:
            return None

        status_endpoint = f"{search_endpoint}/{search_id}"
        for _ in range(30):
            time.sleep(2)
            status_response = requests.get(status_endpoint, headers=headers, verify=False, timeout=60)
            status_response.raise_for_status()
            if status_response.json().get('status') == "COMPLETED":
                results_response = requests.get(
                    f"{status_endpoint}/results", headers=headers, verify=False, timeout=60
                )
                results_response.raise_for_status()
                return results_response.json()
        print("Timeout de la recherche AQL.")
        return None
    except requests.exceptions.RequestException:
        return None


def investigate_offense(offense_id: int, search_window_minutes: int = SEARCH_WINDOW_MINUTES):
    """
    Investigate a given offense ID:
    - Fetch offense details
    - Run event and flow queries
    - Extract IPs and countries
    Returns a dict with investigation results.
    """
    off = {"description": "", "ips": set(), "countries": set()}

    # Step 1: Get offense details
    offense = get_offense_details(offense_id)
    if not offense:
        return off

    off["description"] = offense.get("description", "N/A").strip()
    offense_source_ip = offense.get("offense_source")
    start_time_ms = offense.get("start_time")

    if not (offense_source_ip and start_time_ms):
        return off

    try:
        ipaddress.ip_address(offense_source_ip)
    except ValueError:
        return off

    # Step 2: Build time window
    start_time = datetime.fromtimestamp(start_time_ms / 1000)
    search_start = (start_time - timedelta(minutes=search_window_minutes / 2)).strftime('%Y-%m-%d %H:%M:%S')
    search_end = (start_time + timedelta(minutes=search_window_minutes / 2)).strftime('%Y-%m-%d %H:%M:%S')

    # Step 3: Search events
    aql_events = (f"SELECT UTF8(payload) as payload_text FROM events "
                  f"WHERE sourceip='{offense_source_ip}' OR destinationip='{offense_source_ip}' "
                  f"START '{search_start}' STOP '{search_end}'")
    event_results = run_aql_query(aql_events)

    if event_results and 'events' in event_results:
        for event in event_results['events']:
            payload = event.get('payload_text', 'N/A').strip()

            # Extract srcip
            src_ip_match = re.search(r'srcip=([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)', payload)
            if src_ip_match:
                off["ips"].add(src_ip_match.group(1))

            # Extract srccountry (quoted string)
            src_country_match = re.search(r'srccountry="([^"]+)"', payload)
            if src_country_match:
                off["countries"].add(src_country_match.group(1))

    # Step 4: Search flows
    aql_flows = (f"SELECT APPLICATIONNAME(applicationid) as app, sourceip, destinationip, "
                 f"sourceport, destinationport "
                 f"FROM flows WHERE sourceip='{offense_source_ip}' OR destinationip='{offense_source_ip}' "
                 f"START '{search_start}' STOP '{search_end}'")
    flow_results = run_aql_query(aql_flows)

    if flow_results and 'flows' in flow_results:
        for flow in flow_results['flows']:
            src_ip = flow.get('sourceip')
            if src_ip:
                off["ips"].add(src_ip)
            dst_ip = flow.get('destinationip')
            if dst_ip:
                off["ips"].add(dst_ip)

    return off


