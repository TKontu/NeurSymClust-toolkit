#!/usr/bin/env python3
"""
Extract synergy groups from dendrogram hierarchy using UMAP 5D reduced embeddings.

Uses the Ward linkage of UMAP-reduced cluster centroids to identify natural super-groups
that can be used as synergy definitions in build_method_portfolios.py

IMPORTANT: Uses UMAP 5D space (not 1024D) for better cluster separation.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.pyplot as plt
import umap
import yaml
import aiohttp
import asyncio
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent
EMBEDDINGS_PATH = BASE_DIR / "results_semantic_clustering" / "embeddings.npy"
INPUT_PATH = BASE_DIR / "input" / "methods_deduplicated.csv"
COMBINED_CLUSTERS_PATH = BASE_DIR / "results_semantic_clustering_combined" / "combined_clusters.json"
CLUSTER_NAMES_PATH = BASE_DIR / "input" / "cluster_names.json"
OUTPUT_DIR = BASE_DIR / "results_semantic_clustering_combined"

# UMAP configuration (must match semantic_clustering_combined.py)
UMAP_N_COMPONENTS = 5
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = 'cosine'
UMAP_RANDOM_STATE = 42


def load_data():
    """Load embeddings and cluster assignments."""

    # Load embeddings
    embeddings = np.load(EMBEDDINGS_PATH)

    # Load methods
    methods_df = pd.read_csv(INPUT_PATH, delimiter="|")

    # Load cluster assignments
    with open(COMBINED_CLUSTERS_PATH, 'r') as f:
        clusters_data = json.load(f)

    # Load cluster names from source of truth
    cluster_names_map = {}
    if CLUSTER_NAMES_PATH.exists():
        with open(CLUSTER_NAMES_PATH, 'r') as f:
            names_data = json.load(f)
        for cid, name in names_data.get('primary_clusters', {}).items():
            cluster_names_map[cid] = name
        for cid, name in names_data.get('secondary_clusters', {}).items():
            cluster_names_map[cid] = name

    return embeddings, methods_df, clusters_data, cluster_names_map


def reduce_centroids_umap(centroid_matrix, n_centroids):
    """Reduce cluster centroids from 1024D to 5D using UMAP.

    IMPORTANT: This must match semantic_clustering_combined.py's compute_umap_reduced_centroids()
    to produce consistent dendrograms.
    """
    print(f"  Running UMAP reduction on centroids: {centroid_matrix.shape[1]}D -> {UMAP_N_COMPONENTS}D...")

    reducer = umap.UMAP(
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=min(UMAP_N_NEIGHBORS, n_centroids - 1),  # Can't have more neighbors than points
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=UMAP_RANDOM_STATE
    )
    reduced = reducer.fit_transform(centroid_matrix)
    print(f"  UMAP complete: {reduced.shape}")
    return reduced


def compute_cluster_centroids(embeddings, methods_df, clusters_data, cluster_names_map):
    """Compute centroid embedding for each cluster in original 1024D space.

    Returns 1024D centroids - UMAP reduction is done AFTER centroid computation,
    matching the approach in semantic_clustering_combined.py
    """
    centroids = {}
    cluster_names = {}
    cluster_sizes = {}

    for cluster_id, cluster_data in clusters_data['clusters'].items():
        if cluster_id == 'U':
            continue

        # Get method indices for this cluster
        method_indices = [m['Index'] for m in cluster_data['methods']]

        # Get embeddings in original 1024D space
        mask = methods_df['Index'].isin(method_indices)
        cluster_embeddings = embeddings[mask.values]

        if len(cluster_embeddings) > 0:
            centroids[cluster_id] = cluster_embeddings.mean(axis=0)
            # Use cluster_names_map (source of truth) if available
            cluster_names[cluster_id] = cluster_names_map.get(
                cluster_id,
                cluster_data.get('name', cluster_id)
            )
            cluster_sizes[cluster_id] = len(method_indices)

    return centroids, cluster_names, cluster_sizes


def analyze_dendrogram_cuts(Z, cluster_ids, cluster_names, cluster_sizes):
    """Analyze different cut heights and their resulting groups."""

    print("\n" + "=" * 80)
    print("DENDROGRAM CUT ANALYSIS")
    print("=" * 80)

    # Get the range of distances in the linkage matrix
    distances = Z[:, 2]
    max_dist = distances.max()

    # Try different numbers of groups
    results = {}

    for n_groups in [5, 8, 10, 12, 15]:
        # Cut dendrogram to get n_groups
        group_labels = fcluster(Z, n_groups, criterion='maxclust')

        # Build groups
        groups = {}
        for idx, (cluster_id, group_label) in enumerate(zip(cluster_ids, group_labels)):
            group_key = f"group_{group_label}"
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append({
                'cluster_id': cluster_id,
                'name': cluster_names[cluster_id],
                'size': cluster_sizes[cluster_id]
            })

        results[n_groups] = groups

        print(f"\n--- {n_groups} Groups ---")
        for group_key in sorted(groups.keys(), key=lambda x: int(x.split('_')[1])):
            group = groups[group_key]
            total_methods = sum(c['size'] for c in group)
            cluster_list = [c['cluster_id'] for c in group]
            print(f"  {group_key} ({total_methods} methods): {cluster_list}")
            for c in group:
                print(f"    - {c['cluster_id']}: {c['name'][:50]}...")

    return results


def generate_category_definitions(Z, cluster_ids, cluster_names, cluster_sizes,
                                   cut_distance=None, n_groups=None):
    """Generate category definitions from dendrogram cut.

    Args:
        Z: Linkage matrix
        cluster_ids: List of cluster IDs
        cluster_names: Dict mapping cluster ID to name
        cluster_sizes: Dict mapping cluster ID to size
        cut_distance: Ward distance at which to cut (e.g., 0.6). Takes precedence over n_groups.
        n_groups: Number of groups to create (fallback if cut_distance not specified)

    Returns a list of categories, each containing:
    - name: Full category name (to be filled by LLM)
    - clusters: List of cluster IDs in this category
    - cluster_names: List of cluster names for reference
    - strength/bonus: Based on category cohesion
    """

    # Cut dendrogram by distance or number of groups
    if cut_distance is not None:
        group_labels = fcluster(Z, cut_distance, criterion='distance')
        print(f"  Cutting at Ward distance {cut_distance}")
    else:
        if n_groups is None:
            n_groups = 12
        group_labels = fcluster(Z, n_groups, criterion='maxclust')
        print(f"  Cutting to get {n_groups} groups")

    # Build groups
    groups = {}
    for idx, (cluster_id, group_label) in enumerate(zip(cluster_ids, group_labels)):
        if group_label not in groups:
            groups[group_label] = []
        groups[group_label].append({
            'cluster_id': cluster_id,
            'name': cluster_names[cluster_id],
            'size': cluster_sizes[cluster_id]
        })

    # Generate category definitions as a list
    categories = []

    for group_label, clusters in sorted(groups.items()):
        # Skip single-cluster groups
        if len(clusters) < 2:
            continue

        cluster_ids_in_group = [c['cluster_id'] for c in clusters]
        cluster_names_in_group = [c['name'] for c in clusters]

        # Determine strength based on category cohesion (smaller = stronger)
        if len(clusters) <= 3:
            strength = 'high'
            bonus = 1.2
        elif len(clusters) <= 5:
            strength = 'medium'
            bonus = 1.15
        else:
            strength = 'low'
            bonus = 1.1

        categories.append({
            'name': None,  # Will be filled by LLM
            'clusters': cluster_ids_in_group,
            'cluster_names': cluster_names_in_group,
            'strength': strength,
            'bonus': bonus
        })

    return categories


def load_config():
    """Load LLM configuration from config.yaml."""
    config_path = BASE_DIR / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


async def _call_llm(prompt: str, llm_config: dict, max_tokens: int = 1000) -> str:
    """Call LLM API using aiohttp (same approach as ranking_analyzer.py)."""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{llm_config['base_url']}/chat/completions",
            json={
                "model": llm_config['model'],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": llm_config.get('temperature', 0.3),
                "max_tokens": max_tokens
            },
            headers={"Authorization": f"Bearer {llm_config['api_key']}"},
            timeout=aiohttp.ClientTimeout(total=llm_config.get('timeout', 120))
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"LLM API error: {response.status} - {error_text}")

            data = await response.json()

            # Check response structure
            if 'choices' not in data or len(data['choices']) == 0:
                logger.error(f"Invalid response structure: {data}")
                raise ValueError(f"No choices in response: {data}")

            choice = data['choices'][0]
            content = choice['message']['content']

            if content is None or not content.strip():
                logger.error(f"LLM returned empty content. Response: {data}")
                raise ValueError("LLM returned empty content")

            return content.strip()


def generate_category_names_with_llm(categories: list) -> list:
    """Use LLM to generate full category names.

    Each category name should be descriptive and capture what unifies the clusters.
    """

    config = load_config()
    llm_config = config['llm']

    logger.info(f"Generating names for {len(categories)} categories using LLM...")

    # Build prompt with all categories
    categories_text = ""
    for i, cat in enumerate(categories):
        categories_text += f"\nCategory {i+1} (clusters: {', '.join(cat['clusters'])}):\n"
        for cname in cat['cluster_names']:
            categories_text += f"  - {cname}\n"

    prompt = f"""You are naming categories that group related product development method clusters.

Each category contains semantically similar clusters that share a common theme.
Generate a descriptive name (3-6 words) for each category that captures what unifies its clusters.

The name should:
- Be descriptive and capture the unifying theme
- Use professional product development terminology
- NOT use generic words like "Methods", "Practices", or "Category" at the end
- Be suitable as a category label in a toolkit

Categories to name:
{categories_text}

Respond with a JSON array of category names in the same order as the input.
Example format:
["Team Dynamics & Organizational Culture", "Lean Manufacturing & Flow Optimization", "Agile Transformation & Scaling"]

JSON response:"""

    try:
        # Run async LLM call
        response_text = asyncio.run(_call_llm(prompt, llm_config, max_tokens=1000))

        # Parse JSON from response
        # Handle potential markdown code blocks
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        response_text = response_text.strip()

        names = json.loads(response_text)

        # Apply names to categories
        for i, name in enumerate(names):
            if i < len(categories):
                categories[i]['name'] = name
                logger.info(f"  Category {i+1}: {name}")

    except Exception as e:
        logger.error(f"LLM naming failed: {e}")
        logger.info("Using placeholder names instead")

    # Generate fallback names for any categories without names
    for i, cat in enumerate(categories):
        if cat['name'] is None:
            # Create a simple name from first cluster name
            first_cluster = cat['cluster_names'][0] if cat['cluster_names'] else "Unknown"
            # Take first few words as category name
            words = first_cluster.split()[:4]
            cat['name'] = ' '.join(words)
            logger.info(f"  Fallback name for Category {i+1}: {cat['name']}")

    return categories


def create_synergy_dendrogram(Z, cluster_ids, cluster_names, cluster_sizes, synergies, output_dir):
    """Create a dendrogram visualization with synergy groups highlighted."""

    # Sort clusters for consistent ordering
    sorted_indices = sorted(range(len(cluster_ids)),
                           key=lambda i: (0 if cluster_ids[i].startswith('P') else 1,
                                         int(cluster_ids[i][1:])))

    sorted_cluster_ids = [cluster_ids[i] for i in sorted_indices]

    # Create labels
    labels = [f"{cid}: {cluster_names[cid][:30]}..." for cid in sorted_cluster_ids]

    # Create figure
    fig, ax = plt.subplots(figsize=(24, 14))

    # Plot dendrogram
    dendro = dendrogram(
        Z,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=7,
        ax=ax
    )

    # Color labels by primary/secondary
    xlbls = ax.get_xmajorticklabels()
    for lbl in xlbls:
        text = lbl.get_text()
        if text.startswith('P'):
            lbl.set_color('#2E86AB')
        else:
            lbl.set_color('#A23B72')

    ax.set_title("Cluster Dendrogram with Synergy Groups\n(Cut at height for ~12 groups)", fontsize=14)
    ax.set_xlabel("Cluster ID: Name", fontsize=11)
    ax.set_ylabel("Ward Distance", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / "synergy_dendrogram.png", dpi=150)
    plt.close()

    print(f"\nSaved synergy dendrogram to {output_dir / 'synergy_dendrogram.png'}")


def save_categories(categories, output_dir):
    """Save category definitions to JSON.

    Output format:
    {
        "categories": [
            {
                "name": "Full Category Name",
                "clusters": ["P15", "S11", "S13", "S18"],
                "cluster_names": ["Team Communication...", ...],
                "strength": "high",
                "bonus": 1.2
            },
            ...
        ]
    }
    """

    output = {
        "description": "Category definitions grouping semantically similar clusters (UMAP 5D based)",
        "categories": categories
    }

    output_path = output_dir / "dendrogram_categories.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved categories JSON to {output_path}")

    # Also create a summary text file
    summary_path = output_dir / "dendrogram_categories_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("CATEGORY DEFINITIONS (UMAP 5D based)\n")
        f.write("=" * 60 + "\n\n")

        for i, cat in enumerate(categories):
            f.write(f"{i+1}. {cat['name']}\n")
            f.write(f"   Strength: {cat['strength']} (bonus: {cat['bonus']})\n")
            f.write(f"   Clusters:\n")
            for cid, cname in zip(cat['clusters'], cat['cluster_names']):
                f.write(f"     - {cid}: {cname}\n")
            f.write("\n")

    print(f"Saved categories summary to {summary_path}")


def main():
    print("=" * 80)
    print("EXTRACTING DENDROGRAM-BASED SYNERGIES (UMAP 5D)")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    embeddings, methods_df, clusters_data, cluster_names_map = load_data()
    print(f"  Loaded {len(embeddings)} embeddings")
    print(f"  Loaded {len(cluster_names_map)} cluster names from source of truth")

    # Compute centroids in original 1024D space
    print("\nComputing cluster centroids in 1024D space...")
    centroids, cluster_names, cluster_sizes = compute_cluster_centroids(
        embeddings, methods_df, clusters_data, cluster_names_map
    )

    print(f"  Found {len(centroids)} clusters with 1024D centroids")

    # Sort clusters: Primary first, then Secondary
    cluster_ids = sorted(centroids.keys(),
                        key=lambda x: (0 if x.startswith('P') else 1, int(x[1:])))

    # Build centroid matrix in 1024D
    centroid_matrix_1024d = np.array([centroids[cid] for cid in cluster_ids])
    print(f"  Centroid matrix shape: {centroid_matrix_1024d.shape}")

    # UMAP reduce centroids to 5D (matching semantic_clustering_combined.py approach)
    print("\nReducing centroids with UMAP...")
    centroid_matrix = reduce_centroids_umap(centroid_matrix_1024d, len(cluster_ids))
    print(f"  Reduced centroid matrix shape: {centroid_matrix.shape}")

    # Compute linkage on UMAP 5D centroids
    print("\nComputing Ward linkage on UMAP 5D centroids...")
    Z = linkage(centroid_matrix, method='ward')

    # Analyze different cuts
    results = analyze_dendrogram_cuts(Z, cluster_ids, cluster_names, cluster_sizes)

    # Generate category definitions by cutting at Ward distance 0.6
    # This creates tighter groupings where closely related clusters stay together
    # (e.g., P9 "Cross-Functional Team Structure" + S21 "Adaptive Organization Design")
    CUT_DISTANCE = 1.0
    print("\n" + "=" * 80)
    print(f"GENERATING CATEGORY DEFINITIONS (cut at distance {CUT_DISTANCE}, UMAP 5D based)")
    print("=" * 80)

    categories = generate_category_definitions(Z, cluster_ids, cluster_names, cluster_sizes, cut_distance=CUT_DISTANCE)

    print(f"\nGenerated {len(categories)} categories:")
    for i, cat in enumerate(categories):
        print(f"\n  Category {i+1}:")
        print(f"    Clusters: {cat['clusters']}")
        print(f"    Strength: {cat['strength']} (bonus: {cat['bonus']})")
        for cname in cat['cluster_names']:
            print(f"      - {cname[:60]}")

    # Use LLM to generate full category names
    print("\n" + "=" * 80)
    print("GENERATING CATEGORY NAMES WITH LLM")
    print("=" * 80)
    categories = generate_category_names_with_llm(categories)

    # Create visualization
    create_synergy_dendrogram(Z, cluster_ids, cluster_names, cluster_sizes, categories, OUTPUT_DIR)

    # Save categories
    save_categories(categories, OUTPUT_DIR)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"\nOutput files in {OUTPUT_DIR}:")
    print("  - synergy_dendrogram.png: Visualization (UMAP 5D based)")
    print("  - dendrogram_categories.json: JSON format for build_method_portfolios.py")
    print("  - dendrogram_categories_summary.txt: Human-readable summary")


if __name__ == "__main__":
    main()
