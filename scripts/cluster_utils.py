#!/usr/bin/env python3
"""
Utility functions for working with semantic cluster categories.

Provides dynamic loading of cluster names and synergy group mappings
from the clustering result files.
"""

import json
from pathlib import Path
from typing import Dict, Optional


def load_cluster_mappings(
    clusters_path: str = 'results_semantic_clustering_combined/combined_clusters.json',
    categories_path: str = 'results_semantic_clustering_combined/dendrogram_categories.json'
) -> Dict:
    """
    Load cluster-to-name mappings from clustering results.

    Returns a dict with:
    - cluster_names: {cluster_id: full_cluster_name}
    - cluster_to_synergy: {cluster_id: category_key}
    - synergy_display_names: {category_key: "Human Readable Name"}
    """

    clusters_file = Path(clusters_path)
    categories_file = Path(categories_path)

    result = {
        'cluster_names': {},
        'cluster_to_synergy': {},
        'synergy_display_names': {}
    }

    # Load cluster names from combined_clusters.json
    if clusters_file.exists():
        with open(clusters_file, 'r') as f:
            clusters_data = json.load(f)

        for cluster_id, cluster_data in clusters_data.get('clusters', {}).items():
            result['cluster_names'][cluster_id] = cluster_data.get('name', cluster_id)

    # Load category mappings from dendrogram_categories.json
    # New format: {"categories": [{"name": "...", "clusters": [...], ...}, ...]}
    if categories_file.exists():
        with open(categories_file, 'r') as f:
            categories_data = json.load(f)

        for cat in categories_data.get('categories', []):
            # Create a key from the category name
            cat_name = cat.get('name', 'Unknown')
            cat_key = cat_name.lower().replace(' ', '_').replace('&', 'and').replace('-', '_')
            cat_key = ''.join(c for c in cat_key if c.isalnum() or c == '_')

            # Build reverse mapping: cluster_id -> category_key
            for cluster_id in cat.get('clusters', []):
                result['cluster_to_synergy'][cluster_id] = cat_key

            # Store display name
            result['synergy_display_names'][cat_key] = cat_name

    # For unmapped clusters (standalone or 'U'), use cluster name as synergy
    for cluster_id, cluster_name in result['cluster_names'].items():
        if cluster_id not in result['cluster_to_synergy']:
            # Create a synergy key from the cluster name
            synergy_key = cluster_id  # Use cluster ID as key
            result['cluster_to_synergy'][cluster_id] = synergy_key
            # Use shortened cluster name as display name
            if cluster_name:
                # Take first part before '&' or first 30 chars
                short_name = cluster_name.split('&')[0].strip()
                if len(short_name) > 35:
                    short_name = short_name[:32] + '...'
                result['synergy_display_names'][synergy_key] = short_name
            else:
                result['synergy_display_names'][synergy_key] = cluster_id

    return result


def get_category_display_name(
    cluster_id: str,
    mappings: Optional[Dict] = None
) -> str:
    """
    Get a human-readable display name for a cluster ID.

    Args:
        cluster_id: The cluster ID (e.g., 'P12', 'S5', 'U')
        mappings: Optional pre-loaded mappings from load_cluster_mappings()

    Returns:
        Human-readable name (e.g., 'Agile Scaling', 'Flow Optimization')
    """
    if mappings is None:
        mappings = load_cluster_mappings()

    # Get synergy key for this cluster
    synergy_key = mappings['cluster_to_synergy'].get(cluster_id, cluster_id)

    # Get display name for the synergy
    return mappings['synergy_display_names'].get(synergy_key, cluster_id)


def get_method_category_name(
    method_name: str,
    clusters_path: str = 'results_semantic_clustering_combined/combined_clusters.json'
) -> str:
    """
    Get the category display name for a method by its name.

    Args:
        method_name: The method name
        clusters_path: Path to combined_clusters.json

    Returns:
        Human-readable category name
    """
    clusters_file = Path(clusters_path)

    if not clusters_file.exists():
        return 'Unknown'

    with open(clusters_file, 'r') as f:
        clusters_data = json.load(f)

    # Find which cluster contains this method
    method_name_lower = method_name.strip().lower()

    for cluster_id, cluster_data in clusters_data.get('clusters', {}).items():
        for method in cluster_data.get('methods', []):
            if method.get('Method', '').strip().lower() == method_name_lower:
                return get_category_display_name(cluster_id)

    return 'Unknown'


if __name__ == '__main__':
    # Test the functions
    print("Loading cluster mappings...")
    mappings = load_cluster_mappings()

    print(f"\nCluster names loaded: {len(mappings['cluster_names'])}")
    print(f"Synergy mappings loaded: {len(mappings['cluster_to_synergy'])}")
    print(f"Synergy display names: {len(mappings['synergy_display_names'])}")

    print("\nSample mappings (cluster -> synergy display name):")
    for cid in ['P12', 'P19', 'S22', 'S17', 'U', 'P0']:
        display = get_category_display_name(cid, mappings)
        print(f"  {cid} -> {display}")

    print("\nAll synergy display names:")
    for key, name in sorted(mappings['synergy_display_names'].items()):
        print(f"  {key}: {name}")
