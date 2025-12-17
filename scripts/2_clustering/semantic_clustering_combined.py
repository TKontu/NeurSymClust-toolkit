#!/usr/bin/env python3
"""
Create a combined hierarchical visualization showing both primary and secondary clusters.

Primary clusters: 21 clusters from initial HDBSCAN (415 methods)
Secondary clusters: 26 clusters from noise re-clustering (158 methods)
Still unclustered: 22 methods
"""

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent
EMBEDDINGS_PATH = BASE_DIR / "results_semantic_clustering" / "embeddings.npy"
PRIMARY_CLUSTERS_PATH = BASE_DIR / "results_semantic_clustering_advanced" / "hdbscan_clusters.csv"
SECONDARY_CLUSTERS_PATH = BASE_DIR / "results_semantic_clustering_noise" / "noise_clusters.csv"
INPUT_PATH = BASE_DIR / "input" / "methods_deduplicated.csv"
OUTPUT_DIR = BASE_DIR / "results_semantic_clustering_combined"


def load_all_data():
    """Load all clustering results and embeddings."""

    # Load original data
    methods_df = pd.read_csv(INPUT_PATH, delimiter="|")
    embeddings = np.load(EMBEDDINGS_PATH)

    # Load primary cluster assignments
    primary_df = pd.read_csv(PRIMARY_CLUSTERS_PATH)

    # Load secondary cluster assignments (for noise points)
    secondary_df = pd.read_csv(SECONDARY_CLUSTERS_PATH)

    logger.info(f"Loaded {len(methods_df)} methods")
    logger.info(f"Primary clusters: {primary_df[primary_df['cluster'] != -1]['cluster'].nunique()}")
    logger.info(f"Secondary clusters: {secondary_df[secondary_df['noise_cluster'] != -1]['noise_cluster'].nunique()}")

    return methods_df, embeddings, primary_df, secondary_df


def build_combined_clusters(methods_df, primary_df, secondary_df):
    """
    Build a combined cluster assignment.

    Naming convention:
    - Primary clusters: P0, P1, P2, ... P20
    - Secondary clusters: S0, S1, S2, ... S25
    - Still unclustered: U (unclustered)
    """

    # Create mapping from Index to combined cluster
    combined = {}

    # Primary clusters
    for _, row in primary_df.iterrows():
        idx = row['Index']
        cluster = row['cluster']
        if cluster != -1:
            combined[idx] = f"P{cluster}"

    # Secondary clusters (for noise points)
    for _, row in secondary_df.iterrows():
        idx = row['Index']
        noise_cluster = row['noise_cluster']
        if noise_cluster != -1:
            combined[idx] = f"S{noise_cluster}"
        elif idx not in combined:
            combined[idx] = "U"  # Still unclustered

    # Create combined dataframe
    combined_df = methods_df.copy()
    combined_df['combined_cluster'] = combined_df['Index'].map(combined)

    # Count clusters
    primary_count = len([c for c in combined_df['combined_cluster'].unique() if c.startswith('P')])
    secondary_count = len([c for c in combined_df['combined_cluster'].unique() if c.startswith('S')])
    unclustered_count = (combined_df['combined_cluster'] == 'U').sum()

    logger.info(f"Combined: {primary_count} primary, {secondary_count} secondary, {unclustered_count} unclustered")

    return combined_df


def compute_cluster_centroids(combined_df, embeddings, methods_df):
    """Compute centroid embedding for each cluster."""

    centroids = {}
    cluster_sizes = {}

    for cluster_name in combined_df['combined_cluster'].unique():
        if cluster_name == 'U':
            continue

        cluster_indices = combined_df[combined_df['combined_cluster'] == cluster_name]['Index'].tolist()

        # Get embeddings for this cluster (Index is 1-based, array is 0-based)
        mask = methods_df['Index'].isin(cluster_indices)
        cluster_embeddings = embeddings[mask.values]

        centroid = cluster_embeddings.mean(axis=0)
        centroids[cluster_name] = centroid
        cluster_sizes[cluster_name] = len(cluster_indices)

    return centroids, cluster_sizes


def load_cluster_semantic_names(names_json_path: Path = None) -> dict:
    """
    Load semantic names for clusters from the standalone cluster_names.json file.

    This file is the source of truth and won't be overwritten by this script.
    """
    if names_json_path is None:
        names_json_path = BASE_DIR / "input" / "cluster_names.json"

    if not names_json_path.exists():
        logger.warning(f"Cluster names file not found: {names_json_path}")
        return {}

    with open(names_json_path, 'r') as f:
        data = json.load(f)

    semantic_names = {}

    # Load primary cluster names (P0, P1, etc.)
    for cluster_id, name in data.get('primary_clusters', {}).items():
        semantic_names[cluster_id] = name

    # Load secondary cluster names (S0, S1, etc.)
    for cluster_id, name in data.get('secondary_clusters', {}).items():
        semantic_names[cluster_id] = name

    return semantic_names


def compute_umap_reduced_centroids(centroids: dict, n_components: int = 5) -> dict:
    """Reduce cluster centroids using UMAP."""
    import umap

    cluster_names = sorted(centroids.keys(), key=lambda x: (0 if x.startswith('P') else 1, int(x[1:])))
    centroid_matrix = np.array([centroids[name] for name in cluster_names])

    logger.info(f"Running UMAP on {len(cluster_names)} cluster centroids (1024D -> {n_components}D)")

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=min(15, len(cluster_names) - 1),  # Can't have more neighbors than points
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )
    reduced = reducer.fit_transform(centroid_matrix)

    reduced_centroids = {name: reduced[i] for i, name in enumerate(cluster_names)}
    return reduced_centroids


def create_cluster_dendrogram(centroids, cluster_sizes, output_dir, cluster_semantic_names=None,
                              suffix="", title_extra="", ylabel="Ward Distance"):
    """Create dendrogram of cluster centroids."""

    # Sort clusters: Primary first, then Secondary
    cluster_names = sorted(centroids.keys(), key=lambda x: (0 if x.startswith('P') else 1, int(x[1:])))

    # Build centroid matrix
    centroid_matrix = np.array([centroids[name] for name in cluster_names])

    # Compute linkage
    Z = linkage(centroid_matrix, method='ward')

    # Create labels with semantic names if available
    if cluster_semantic_names:
        labels = [f"{cluster_semantic_names.get(name, name)} ({cluster_sizes[name]})" for name in cluster_names]
    else:
        labels = [f"{name} ({cluster_sizes[name]})" for name in cluster_names]

    # Color mapping
    def get_color(name):
        if name.startswith('P'):
            return '#2E86AB'  # Blue for primary
        else:
            return '#A23B72'  # Purple for secondary

    colors = [get_color(name) for name in cluster_names]

    # Create figure - A4 portrait size (8.27 x 11.69 inches) for easy printing, rotated layout
    fig, ax = plt.subplots(figsize=(8.27, 11.69))

    # Plot dendrogram with improved readability - rotated 90 degrees (orientation='right')
    dendro = dendrogram(
        Z,
        labels=labels,
        orientation='right',  # Rotate 90 degrees - tree grows to the right
        leaf_font_size=10,  # Increased from 7 for better readability
        ax=ax,
        color_threshold=0,
        above_threshold_color='gray'
    )

    # Increase line width for better visibility
    for line in ax.get_lines():
        line.set_linewidth(2.0)  # Thicker lines for better readability

    # Color the labels based on cluster type (P or S)
    # With orientation='right', labels are on y-axis
    ylbls = ax.get_ymajorticklabels()
    for i, lbl in enumerate(ylbls):
        # Get original cluster name from the order in dendrogram
        leaf_idx = dendro['leaves'][i]
        original_name = cluster_names[leaf_idx]
        if original_name.startswith('P'):
            lbl.set_color('#2E86AB')
        else:
            lbl.set_color('#A23B72')

    title = f"Hierarchical Clustering of Method Clusters{title_extra}\n(Blue: Primary Clusters, Purple: Secondary Clusters)"
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20, loc='right')  # Centered title
    ax.set_xlabel(ylabel, fontsize=13)  # Distance is now on x-axis
    ax.set_ylabel("Cluster (method count)", fontsize=13)  # Labels are now on y-axis

    # Increase tick label size for x-axis (distance scale)
    ax.tick_params(axis='x', labelsize=11)

    # X-axis shows distances from low to high (left to right)
    # Small clusters merge first (near labels on left), larger merges to the right

    plt.tight_layout()
    filename = f"combined_dendrogram{suffix}.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')  # Higher DPI for better quality
    plt.close()

    logger.info(f"Saved dendrogram to {output_dir / filename}")

    return Z, cluster_names


def create_full_dendrogram(combined_df, embeddings, methods_df, output_dir):
    """Create a dendrogram showing all methods colored by their cluster type."""

    # Compute linkage on all embeddings
    logger.info("Computing full linkage matrix (this may take a moment)...")
    Z = linkage(embeddings, method='ward')

    # Create color mapping based on cluster type
    def get_leaf_color(idx):
        # idx is 0-based row index
        method_idx = methods_df.iloc[idx]['Index']
        cluster = combined_df[combined_df['Index'] == method_idx]['combined_cluster'].values[0]
        if cluster.startswith('P'):
            return '#2E86AB'  # Blue
        elif cluster.startswith('S'):
            return '#A23B72'  # Purple
        else:
            return '#888888'  # Gray for unclustered

    # Create figure
    fig, ax = plt.subplots(figsize=(24, 10))

    # Plot truncated dendrogram (showing top-level structure)
    dendro = dendrogram(
        Z,
        truncate_mode='lastp',
        p=50,  # Show last 50 merges
        leaf_rotation=90,
        leaf_font_size=8,
        ax=ax,
        show_contracted=True
    )

    ax.set_title("Full Method Hierarchy (truncated to top 50 clusters)", fontsize=14)
    ax.set_xlabel("Cluster Size", fontsize=11)
    ax.set_ylabel("Ward Distance", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / "full_dendrogram_truncated.png", dpi=150)
    plt.close()

    logger.info(f"Saved full dendrogram to {output_dir / 'full_dendrogram_truncated.png'}")


def save_combined_results(combined_df, centroids, cluster_sizes, output_dir, cluster_semantic_names=None):
    """Save combined clustering results."""

    # Save combined assignments
    combined_df.to_csv(output_dir / "combined_clusters.csv", index=False)

    # Save summary
    with open(output_dir / "combined_clusters_summary.txt", "w") as f:
        f.write("Combined Semantic Clustering Results\n")
        f.write("=" * 80 + "\n")

        primary_clusters = [c for c in cluster_sizes.keys() if c.startswith('P')]
        secondary_clusters = [c for c in cluster_sizes.keys() if c.startswith('S')]
        unclustered = (combined_df['combined_cluster'] == 'U').sum()

        f.write(f"Primary clusters: {len(primary_clusters)} ({sum(cluster_sizes[c] for c in primary_clusters)} methods)\n")
        f.write(f"Secondary clusters: {len(secondary_clusters)} ({sum(cluster_sizes[c] for c in secondary_clusters)} methods)\n")
        f.write(f"Still unclustered: {unclustered} methods\n")
        f.write(f"Total: {len(combined_df)} methods\n")
        f.write("=" * 80 + "\n\n")

        # Primary clusters
        f.write("\n" + "=" * 80 + "\n")
        f.write("PRIMARY CLUSTERS (from initial HDBSCAN)\n")
        f.write("=" * 80 + "\n")

        for cluster in sorted(primary_clusters, key=lambda x: int(x[1:])):
            cluster_methods = combined_df[combined_df['combined_cluster'] == cluster]
            f.write(f"\n{cluster} ({len(cluster_methods)} methods)\n")
            f.write("-" * 40 + "\n")
            for _, row in cluster_methods.iterrows():
                f.write(f"  [{row['Index']}] {row['Method']}\n")

        # Secondary clusters
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("SECONDARY CLUSTERS (from noise re-clustering)\n")
        f.write("=" * 80 + "\n")

        for cluster in sorted(secondary_clusters, key=lambda x: int(x[1:])):
            cluster_methods = combined_df[combined_df['combined_cluster'] == cluster]
            f.write(f"\n{cluster} ({len(cluster_methods)} methods)\n")
            f.write("-" * 40 + "\n")
            for _, row in cluster_methods.iterrows():
                f.write(f"  [{row['Index']}] {row['Method']}\n")

        # Unclustered
        unclustered_methods = combined_df[combined_df['combined_cluster'] == 'U']
        if len(unclustered_methods) > 0:
            f.write("\n\n" + "=" * 80 + "\n")
            f.write(f"UNCLUSTERED ({len(unclustered_methods)} methods)\n")
            f.write("=" * 80 + "\n\n")
            for _, row in unclustered_methods.iterrows():
                f.write(f"  [{row['Index']}] {row['Method']}\n")

    logger.info(f"Saved combined summary to {output_dir / 'combined_clusters_summary.txt'}")

    # Save JSON (convert numpy types to native Python types)
    cluster_json = {
        "primary_clusters": int(len(primary_clusters)),
        "secondary_clusters": int(len(secondary_clusters)),
        "unclustered": int(unclustered),
        "clusters": {}
    }

    for cluster in sorted(cluster_sizes.keys(), key=lambda x: (0 if x.startswith('P') else 1, int(x[1:]))):
        cluster_methods = combined_df[combined_df['combined_cluster'] == cluster]
        methods_list = []
        for _, row in cluster_methods[["Index", "Method", "Description", "Source"]].iterrows():
            methods_list.append({
                "Index": int(row["Index"]),
                "Method": str(row["Method"]),
                "Description": str(row["Description"]),
                "Source": str(row["Source"])
            })
        cluster_data = {
            "type": "primary" if cluster.startswith('P') else "secondary",
            "methods": methods_list,
            "count": len(cluster_methods)
        }
        # Preserve semantic name if available
        if cluster_semantic_names and cluster in cluster_semantic_names:
            cluster_data["name"] = cluster_semantic_names[cluster]
        cluster_json["clusters"][cluster] = cluster_data

    # Add unclustered
    unclustered_methods = combined_df[combined_df['combined_cluster'] == 'U']
    if len(unclustered_methods) > 0:
        methods_list = []
        for _, row in unclustered_methods[["Index", "Method", "Description", "Source"]].iterrows():
            methods_list.append({
                "Index": int(row["Index"]),
                "Method": str(row["Method"]),
                "Description": str(row["Description"]),
                "Source": str(row["Source"])
            })
        cluster_json["clusters"]["U"] = {
            "type": "unclustered",
            "methods": methods_list,
            "count": len(unclustered_methods)
        }

    with open(output_dir / "combined_clusters.json", "w") as f:
        json.dump(cluster_json, f, indent=2)

    logger.info(f"Saved JSON to {output_dir / 'combined_clusters.json'}")


def main():
    """Main entry point."""
    logger.info("Creating combined clustering visualization")

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load all data
    methods_df, embeddings, primary_df, secondary_df = load_all_data()

    # Build combined cluster assignments
    combined_df = build_combined_clusters(methods_df, primary_df, secondary_df)

    # Compute cluster centroids
    centroids, cluster_sizes = compute_cluster_centroids(combined_df, embeddings, methods_df)

    # Load semantic names from standalone file (input/cluster_names.json)
    # This file is the source of truth and won't be overwritten
    cluster_semantic_names = load_cluster_semantic_names()
    if cluster_semantic_names:
        logger.info(f"Loaded {len(cluster_semantic_names)} semantic cluster names from input/cluster_names.json")
    else:
        logger.warning("No semantic cluster names found - using P0, S0, etc.")
        logger.warning("Create input/cluster_names.json to add semantic names")

    # Save results with semantic names embedded
    save_combined_results(combined_df, centroids, cluster_sizes, OUTPUT_DIR, cluster_semantic_names)

    # Create cluster-level dendrograms (multiple variants)
    print("\n" + "=" * 80)
    print("Creating cluster dendrograms...")
    print("=" * 80)

    # 1. Original 1024D centroids
    print("\n1. Dendrogram from 1024D cluster centroids...")
    Z, cluster_names = create_cluster_dendrogram(
        centroids, cluster_sizes, OUTPUT_DIR, cluster_semantic_names,
        suffix="_1024d",
        title_extra=" (Original 1024D Embeddings)",
        ylabel="Ward Distance (1024D)"
    )

    # 2. UMAP-reduced centroids
    print("\n2. Dendrogram from UMAP-reduced cluster centroids...")
    umap_centroids = compute_umap_reduced_centroids(centroids, n_components=5)
    create_cluster_dendrogram(
        umap_centroids, cluster_sizes, OUTPUT_DIR, cluster_semantic_names,
        suffix="_umap5d",
        title_extra=" (UMAP 5D Reduced)",
        ylabel="Ward Distance (UMAP 5D)"
    )

    # Also create default (copy of 1024D for backwards compatibility)
    create_cluster_dendrogram(
        centroids, cluster_sizes, OUTPUT_DIR, cluster_semantic_names,
        suffix="",
        title_extra=" (Original 1024D Embeddings)",
        ylabel="Ward Distance (1024D)"
    )

    # Create full method dendrogram (truncated)
    print("\n" + "=" * 80)
    print("Creating full method dendrogram...")
    print("=" * 80)
    create_full_dendrogram(combined_df, embeddings, methods_df, OUTPUT_DIR)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    primary_count = len([c for c in cluster_sizes.keys() if c.startswith('P')])
    secondary_count = len([c for c in cluster_sizes.keys() if c.startswith('S')])
    unclustered = (combined_df['combined_cluster'] == 'U').sum()

    print(f"\nPrimary clusters (P): {primary_count}")
    print(f"Secondary clusters (S): {secondary_count}")
    print(f"Total clusters: {primary_count + secondary_count}")
    print(f"Unclustered methods: {unclustered}")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nFiles:")
    print("  - combined_dendrogram.png: Hierarchical view (1024D, default)")
    print("  - combined_dendrogram_1024d.png: Hierarchical view (original 1024D embeddings)")
    print("  - combined_dendrogram_umap5d.png: Hierarchical view (UMAP 5D reduced)")
    print("  - full_dendrogram_truncated.png: All methods hierarchy (truncated)")
    print("  - combined_clusters.csv: All method assignments")
    print("  - combined_clusters_summary.txt: Human-readable summary")
    print("  - combined_clusters.json: Machine-readable data")
    print("\nDendrogram comparison:")
    print("  - 1024D: Ward distance in original semantic space (more 'honest')")
    print("  - UMAP 5D: Ward distance in reduced space (may show clearer separation)")


if __name__ == "__main__":
    main()
