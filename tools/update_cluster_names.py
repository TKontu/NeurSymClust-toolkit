#!/usr/bin/env python3
"""
Extract cluster names from named summary files and update JSON files.

Reads names from:
- results_semantic_clustering_advanced/hdbscan_clusters_summary_named.txt
- results_semantic_clustering_noise/noise_clusters_summary_named.txt

Updates:
- results_semantic_clustering_advanced/hdbscan_clusters.json
- results_semantic_clustering_noise/noise_clusters.json
- results_semantic_clustering_combined/combined_clusters.json (if exists)
"""

import json
import re
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent


def extract_cluster_names(summary_file: Path) -> dict:
    """
    Extract cluster names from a named summary file.

    Looks for patterns like:
    - PRIMARY CLUSTER 0 (14 methods) # Project Scheduling & Dependency Management
    - SECONDARY CLUSTER  0 (5 methods) # System Metaphor & Shared Mental Models
    - CLUSTER 0 (14 methods) # Some Name

    Returns dict mapping cluster_id (int) to name (str)
    """
    names = {}

    # Pattern matches: CLUSTER/PRIMARY CLUSTER/SECONDARY CLUSTER + number + anything + # name
    pattern = r'(?:PRIMARY |SECONDARY )?CLUSTER\s+(\d+)\s+\([^)]+\)\s*#\s*(.+)'

    with open(summary_file, 'r') as f:
        for line in f:
            match = re.search(pattern, line.strip())
            if match:
                cluster_id = int(match.group(1))
                name = match.group(2).strip()
                names[cluster_id] = name
                logger.debug(f"Found cluster {cluster_id}: {name}")

    logger.info(f"Extracted {len(names)} cluster names from {summary_file.name}")
    return names


def update_json_with_names(json_file: Path, names: dict, cluster_key_prefix: str = "cluster_"):
    """
    Update a JSON file with cluster names.

    Adds a 'name' field to each cluster in the 'clusters' dict.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    updated_count = 0

    for key, cluster_data in data.get('clusters', {}).items():
        # Extract cluster number from key (e.g., "cluster_0" -> 0, "noise_cluster_5" -> 5)
        match = re.search(r'(\d+)$', key)
        if match:
            cluster_id = int(match.group(1))
            if cluster_id in names:
                cluster_data['name'] = names[cluster_id]
                updated_count += 1
                logger.debug(f"Updated {key} with name: {names[cluster_id]}")

    # Save updated JSON
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"Updated {updated_count} clusters in {json_file.name}")
    return updated_count


def update_combined_json(combined_json: Path, primary_names: dict, secondary_names: dict):
    """
    Update combined JSON with both primary and secondary cluster names.

    Keys are like "P0", "P1", "S0", "S1", etc.
    """
    with open(combined_json, 'r') as f:
        data = json.load(f)

    updated_count = 0

    for key, cluster_data in data.get('clusters', {}).items():
        if key.startswith('P'):
            cluster_id = int(key[1:])
            if cluster_id in primary_names:
                cluster_data['name'] = primary_names[cluster_id]
                updated_count += 1
        elif key.startswith('S'):
            cluster_id = int(key[1:])
            if cluster_id in secondary_names:
                cluster_data['name'] = secondary_names[cluster_id]
                updated_count += 1

    with open(combined_json, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"Updated {updated_count} clusters in {combined_json.name}")
    return updated_count


def main():
    """Main entry point."""
    logger.info("Extracting cluster names and updating JSON files")

    # Paths
    primary_summary = BASE_DIR / "results_semantic_clustering_advanced" / "hdbscan_clusters_summary_named.txt"
    primary_json = BASE_DIR / "results_semantic_clustering_advanced" / "hdbscan_clusters.json"

    secondary_summary = BASE_DIR / "results_semantic_clustering_noise" / "noise_clusters_summary_named.txt"
    secondary_json = BASE_DIR / "results_semantic_clustering_noise" / "noise_clusters.json"

    combined_json = BASE_DIR / "results_semantic_clustering_combined" / "combined_clusters.json"

    # Extract names
    primary_names = {}
    secondary_names = {}

    if primary_summary.exists():
        primary_names = extract_cluster_names(primary_summary)
    else:
        logger.warning(f"Primary summary not found: {primary_summary}")

    if secondary_summary.exists():
        secondary_names = extract_cluster_names(secondary_summary)
    else:
        logger.warning(f"Secondary summary not found: {secondary_summary}")

    # Update primary JSON
    if primary_json.exists() and primary_names:
        update_json_with_names(primary_json, primary_names, "cluster_")

    # Update secondary JSON
    if secondary_json.exists() and secondary_names:
        update_json_with_names(secondary_json, secondary_names, "noise_cluster_")

    # Update combined JSON
    if combined_json.exists() and (primary_names or secondary_names):
        update_combined_json(combined_json, primary_names, secondary_names)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nPrimary cluster names extracted: {len(primary_names)}")
    print(f"Secondary cluster names extracted: {len(secondary_names)}")
    print(f"\nUpdated files:")
    if primary_json.exists():
        print(f"  - {primary_json}")
    if secondary_json.exists():
        print(f"  - {secondary_json}")
    if combined_json.exists():
        print(f"  - {combined_json}")

    # Print all names for verification
    if primary_names:
        print("\n" + "-" * 60)
        print("Primary Cluster Names:")
        print("-" * 60)
        for cid in sorted(primary_names.keys()):
            print(f"  P{cid}: {primary_names[cid]}")

    if secondary_names:
        print("\n" + "-" * 60)
        print("Secondary Cluster Names:")
        print("-" * 60)
        for cid in sorted(secondary_names.keys()):
            print(f"  S{cid}: {secondary_names[cid]}")


if __name__ == "__main__":
    main()
