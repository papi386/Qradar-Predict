import ipaddress
import tldextract
import math
import pandas as pd
from typing import List



# 🔹 Shannon entropy calculation
def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    probabilities = [float(s.count(c)) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probabilities)




# 🔹 Count IPs in off["ips"]
def count_ips(off: dict) -> int:
    return len(off.get("ips", []))


# 🔹 Count unique countries from IPs
def count_unique_countries(off: dict) -> int:
    return len(off['countries'])


# 🔹 Check multiple IP ranges (/24)
def has_multiple_ip_ranges(ips) -> int:
    networks = set()
    for ip in ips:
        try:
            net = ipaddress.ip_network(ip + "/24", strict=False)
            networks.add(net)
        except:
            pass
    return 1 if len(networks) > 1 else 0


# 🔹 Extract domain features from a list of domains
def analyze_domains(domains, bad_domains=None) -> dict:

    domains_seen = set() #the domains in our offenses are already in a set but just in case :)
    entropies = []

    for d in domains:
        ext = tldextract.extract(d)
        domain = f"{ext.domain}.{ext.suffix}".lower()
        domains_seen.add(domain)
        entropies.append(shannon_entropy(domain))

    avg_entropy = sum(entropies) / len(entropies) if entropies else 0

    return {
        "num_domains": len(domains_seen),
        "avg_domain_entropy": avg_entropy
    }


# 🔹 Analyze URLs (length-based features)
"""
def analyze_urls(urls) -> dict:
    lengths = [len(u) for u in urls]
    avg_length = sum(lengths) / len(lengths) if lengths else 0
    return {
        "num_urls": len(urls),
        "avg_length_of_urls": avg_length
    }
"""

"""
# 🔹 Master feature extraction from off (list of dict)
def extract_features_from_off(off: dict) -> pd.DataFrame:
    ips = list(off.get("ips", []))
    countries = list(off.get("countries", []))
    domains = list(off.get("domains", [])) if "domains" in off else []
    urls = list(off.get("urls", [])) if "urls" in off else []
    hashes = list(off.get("hashes", [])) if "hashes" in off else []

    ip_features = {
        "num_ips": len(ips),
        "unique_ip_countries": len(set(countries)),
        "has_multiple_ip_ranges": has_multiple_ip_ranges(ips)
    }

    domain_features = analyze_domains(domains)

    feature_dict = {
        "event_info": off.get("description", ""),
        **ip_features,
        **domain_features
    }

    # Return as a one-row DataFrame
    return pd.DataFrame([feature_dict])
"""
def extract_features_from_off(offenses: List[dict]) -> pd.DataFrame:
    """
    Extract features from a list of offense dictionaries and return a DataFrame.
    """
    dataset = []

    for off in offenses:
        ips = list(off.get("ips", []))
        countries = list(off.get("countries", []))
        domains = list(off.get("domains", [])) if "domains" in off else []
        urls = list(off.get("urls", [])) if "urls" in off else []
        hashes = list(off.get("hashes", [])) if "hashes" in off else []

        ip_features = {
            "num_ips": len(ips),
            "unique_ip_countries": len(set(countries)),
            "has_multiple_ip_ranges": has_multiple_ip_ranges(ips),
        }

        domain_features = analyze_domains(domains)

        feature_dict = {
            "event_info": off.get("description", ""),
            **ip_features,
            #add features depending on attack types
        }

        dataset.append(feature_dict)

    # Return DataFrame with one row per offense
    return pd.DataFrame(dataset)