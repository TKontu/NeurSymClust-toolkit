#!/usr/bin/env python3
"""
Advanced semantic clustering using Hierarchical Clustering and UMAP + HDBSCAN.

This script provides alternative clustering methods that work better for
overlapping/continuous data compared to K-means.
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
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import umap
import hdbscan
from collections import Counter

# Suppress sklearn deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
INPUT_PATH = BASE_DIR / "input" / "methods_deduplicated.csv"
EMBEDDINGS_PATH = BASE_DIR / "results_semantic_clustering" / "embeddings.npy"
OUTPUT_DIR = BASE_DIR / "results_semantic_clustering_advanced"


def load_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Load methods and embeddings."""
    df = pd.read_csv(INPUT_PATH, delimiter="|")
    logger.info(f"Loaded {len(df)} methods")

    embeddings = np.load(EMBEDDINGS_PATH)
    logger.info(f"Loaded embeddings with shape: {embeddings.shape}")

    return df, embeddings


def compute_cluster_quality_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    name: str = "Clustering"
) -> dict:
    """
    Compute clustering quality metrics.

    Args:
        embeddings: The embedding vectors
        labels: Cluster labels (-1 for noise in HDBSCAN)
        name: Name for logging

    Returns:
        Dictionary with quality metrics
    """
    # Filter out noise points (label == -1) for metric calculation
    mask = labels != -1
    n_clustered = mask.sum()
    n_noise = (~mask).sum()
    n_clusters = len(set(labels[mask]))

    if n_clusters < 2:
        logger.warning(f"{name}: Need at least 2 clusters for quality metrics, found {n_clusters}")
        return {
            "n_clusters": n_clusters,
            "n_clustered": int(n_clustered),
            "n_noise": int(n_noise),
            "silhouette_score": None,
            "calinski_harabasz_score": None,
            "davies_bouldin_score": None
        }

    X = embeddings[mask]
    y = labels[mask]

    # Compute metrics
    silhouette = silhouette_score(X, y)
    calinski = calinski_harabasz_score(X, y)
    davies = davies_bouldin_score(X, y)

    metrics = {
        "n_clusters": n_clusters,
        "n_clustered": int(n_clustered),
        "n_noise": int(n_noise),
        "silhouette_score": float(silhouette),
        "calinski_harabasz_score": float(calinski),
        "davies_bouldin_score": float(davies)
    }

    logger.info(f"{name} quality metrics: silhouette={silhouette:.3f}, calinski={calinski:.1f}, davies={davies:.3f}")

    return metrics


def hierarchical_clustering(
    embeddings: np.ndarray,
    method: str = "ward",
    cut_thresholds: list[float] = None
) -> dict:
    """
    Perform hierarchical clustering.

    Args:
        embeddings: The embedding vectors
        method: Linkage method ('ward', 'complete', 'average', 'single')
        cut_thresholds: Distance thresholds to cut the dendrogram

    Returns:
        Dictionary with linkage matrix and cluster assignments at different cuts
    """
    logger.info(f"Computing hierarchical clustering with method='{method}'")

    # Compute linkage matrix
    Z = linkage(embeddings, method=method)

    # If no thresholds provided, compute some based on the dendrogram
    if cut_thresholds is None:
        # Use different number of clusters
        cluster_counts = [5, 8, 10, 13, 15, 20, 25, 30]
        results = {}
        for n_clusters in cluster_counts:
            labels = fcluster(Z, n_clusters, criterion='maxclust')
            results[n_clusters] = labels - 1  # 0-indexed
    else:
        results = {}
        for threshold in cut_thresholds:
            labels = fcluster(Z, threshold, criterion='distance')
            n_clusters = len(set(labels))
            results[n_clusters] = labels - 1

    return {"linkage": Z, "cluster_assignments": results}


def umap_hdbscan_clustering(
    embeddings: np.ndarray,
    n_components: int = 5,
    min_cluster_size: int = 10,
    min_samples: int = 5
) -> dict:
    """
    Perform UMAP dimensionality reduction followed by HDBSCAN clustering.

    Args:
        embeddings: The embedding vectors
        n_components: UMAP target dimensions
        min_cluster_size: Minimum cluster size for HDBSCAN
        min_samples: Minimum samples for core points in HDBSCAN

    Returns:
        Dictionary with reduced embeddings, labels, and probabilities
    """
    logger.info(f"Running UMAP (n_components={n_components})")

    # UMAP reduction
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )
    reduced = reducer.fit_transform(embeddings)
    logger.info(f"UMAP reduced to shape: {reduced.shape}")

    # Also create 2D for visualization
    reducer_2d = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )
    reduced_2d = reducer_2d.fit_transform(embeddings)

    logger.info(f"Running HDBSCAN (min_cluster_size={min_cluster_size}, min_samples={min_samples})")

    # HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    labels = clusterer.fit_predict(reduced)
    probabilities = clusterer.probabilities_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    logger.info(f"HDBSCAN found {n_clusters} clusters, {n_noise} noise points")

    return {
        "reduced_5d": reduced,
        "reduced_2d": reduced_2d,
        "labels": labels,
        "probabilities": probabilities,
        "n_clusters": n_clusters,
        "n_noise": n_noise
    }


def save_hierarchical_results(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    hier_results: dict,
    output_dir: Path
) -> dict:
    """Save hierarchical clustering results."""
    output_dir.mkdir(exist_ok=True)

    # Save dendrogram
    plt.figure(figsize=(20, 10))
    dendrogram(
        hier_results["linkage"],
        truncate_mode='lastp',
        p=50,  # Show last 50 merges
        leaf_rotation=90,
        leaf_font_size=8,
        show_contracted=True
    )
    plt.title("Hierarchical Clustering Dendrogram (truncated)")
    plt.xlabel("Cluster Size")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(output_dir / "dendrogram.png", dpi=150)
    plt.close()
    logger.info(f"Saved dendrogram to {output_dir / 'dendrogram.png'}")

    # Compute and collect quality metrics for each k
    all_metrics = {}

    # Save cluster assignments for each cut
    for n_clusters, labels in hier_results["cluster_assignments"].items():
        cluster_df = df.copy()
        cluster_df["cluster"] = labels
        cluster_df = cluster_df.sort_values(["cluster", "Index"])

        # Compute quality metrics
        metrics = compute_cluster_quality_metrics(
            embeddings, labels, name=f"Hierarchical k={n_clusters}"
        )
        all_metrics[n_clusters] = metrics

        # Save CSV
        cluster_df.to_csv(output_dir / f"hierarchical_k{n_clusters}.csv", index=False)

        # Save summary
        with open(output_dir / f"hierarchical_k{n_clusters}_summary.txt", "w") as f:
            f.write(f"Hierarchical Clustering Results (k={n_clusters})\n")
            f.write("=" * 80 + "\n")
            f.write(f"Silhouette Score: {metrics['silhouette_score']:.3f}\n")
            f.write(f"Calinski-Harabasz Index: {metrics['calinski_harabasz_score']:.1f}\n")
            f.write(f"Davies-Bouldin Index: {metrics['davies_bouldin_score']:.3f}\n")
            f.write("=" * 80 + "\n\n")

            for cluster_id in range(n_clusters):
                cluster_methods = cluster_df[cluster_df["cluster"] == cluster_id]
                f.write(f"\n{'='*80}\n")
                f.write(f"CLUSTER {cluster_id} ({len(cluster_methods)} methods)\n")
                f.write(f"{'='*80}\n\n")

                for _, row in cluster_methods.iterrows():
                    f.write(f"  [{row['Index']}] {row['Method']}\n")

        logger.info(f"Saved hierarchical k={n_clusters} results")

    # Save quality metrics summary
    with open(output_dir / "hierarchical_quality_metrics.txt", "w") as f:
        f.write("Hierarchical Clustering Quality Metrics\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'k':>4} {'Silhouette':>12} {'Calinski-H':>12} {'Davies-B':>10}  Interpretation\n")
        f.write("-" * 80 + "\n")
        for k in sorted(all_metrics.keys()):
            m = all_metrics[k]
            sil = m['silhouette_score']
            # Interpret silhouette score
            if sil >= 0.7:
                interp = "Strong structure"
            elif sil >= 0.5:
                interp = "Reasonable structure"
            elif sil >= 0.25:
                interp = "Weak structure"
            else:
                interp = "No structure"
            f.write(f"{k:>4} {sil:>12.3f} {m['calinski_harabasz_score']:>12.1f} {m['davies_bouldin_score']:>10.3f}  {interp}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Metric Interpretation:\n")
        f.write("  - Silhouette Score: -1 to 1, higher is better (>0.5 good, >0.7 strong)\n")
        f.write("  - Calinski-Harabasz: Higher is better (no fixed scale)\n")
        f.write("  - Davies-Bouldin: Lower is better (<1 is good)\n")

    return all_metrics


def save_hdbscan_results(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    hdbscan_results: dict,
    output_dir: Path
) -> dict:
    """Save UMAP + HDBSCAN results."""
    output_dir.mkdir(exist_ok=True)

    labels = hdbscan_results["labels"]
    reduced_2d = hdbscan_results["reduced_2d"]

    # Compute quality metrics
    metrics = compute_cluster_quality_metrics(embeddings, labels, name="HDBSCAN")

    # Create visualization - A4 landscape optimized, focused on data region
    plt.figure(figsize=(11.69, 8.27))

    # Color by cluster, noise points in gray
    unique_labels = set(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for idx, label in enumerate(unique_labels):
        mask = labels == label
        if label == -1:
            plt.scatter(
                reduced_2d[mask, 0], reduced_2d[mask, 1],
                c='lightgray', alpha=0.5, s=30  # No label - legend removed
            )
        else:
            plt.scatter(
                reduced_2d[mask, 0], reduced_2d[mask, 1],
                c=[colors[idx]], alpha=0.7, s=50  # No label - legend removed, larger points
            )

    plt.title(f"UMAP + HDBSCAN Clustering\n{hdbscan_results['n_clusters']} clusters, {hdbscan_results['n_noise']} noise points",
              fontsize=18, fontweight='bold')  # Larger, bold title
    plt.xlabel("UMAP 1", fontsize=14)  # Larger axis labels
    plt.ylabel("UMAP 2", fontsize=14)

    # Focus on data-rich region: X=2...17, Y=0...15
    plt.xlim(2, 17)
    plt.ylim(0, 15)

    # Increase tick label sizes
    plt.tick_params(axis='both', labelsize=12)

    # No legend - removed completely as requested
    plt.tight_layout()
    plt.savefig(output_dir / "umap_hdbscan_clusters.png", dpi=300)  # Higher DPI
    plt.close()
    logger.info(f"Saved UMAP visualization to {output_dir / 'umap_hdbscan_clusters.png'}")

    # Save cluster assignments
    cluster_df = df.copy()
    cluster_df["cluster"] = labels
    cluster_df["cluster_probability"] = hdbscan_results["probabilities"]
    cluster_df["umap_x"] = reduced_2d[:, 0]
    cluster_df["umap_y"] = reduced_2d[:, 1]

    # Sort: clusters first (by cluster id), then noise at end
    cluster_df["sort_key"] = cluster_df["cluster"].apply(lambda x: 9999 if x == -1 else x)
    cluster_df = cluster_df.sort_values(["sort_key", "cluster_probability"], ascending=[True, False])
    cluster_df = cluster_df.drop(columns=["sort_key"])

    cluster_df.to_csv(output_dir / "hdbscan_clusters.csv", index=False)

    # Save summary
    with open(output_dir / "hdbscan_clusters_summary.txt", "w") as f:
        f.write("UMAP + HDBSCAN Clustering Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total clusters: {hdbscan_results['n_clusters']}\n")
        f.write(f"Noise points: {hdbscan_results['n_noise']} ({hdbscan_results['n_noise']/len(labels)*100:.1f}%)\n")
        f.write("-" * 80 + "\n")
        f.write("Quality Metrics:\n")
        if metrics['silhouette_score'] is not None:
            f.write(f"  Silhouette Score: {metrics['silhouette_score']:.3f}")
            sil = metrics['silhouette_score']
            if sil >= 0.7:
                f.write(" (Strong structure)\n")
            elif sil >= 0.5:
                f.write(" (Reasonable structure)\n")
            elif sil >= 0.25:
                f.write(" (Weak structure)\n")
            else:
                f.write(" (No clear structure)\n")
            f.write(f"  Calinski-Harabasz Index: {metrics['calinski_harabasz_score']:.1f}\n")
            f.write(f"  Davies-Bouldin Index: {metrics['davies_bouldin_score']:.3f}")
            if metrics['davies_bouldin_score'] < 1:
                f.write(" (Good)\n")
            else:
                f.write(" (Could be better)\n")
        else:
            f.write("  (Not enough clusters to compute metrics)\n")
        f.write("=" * 80 + "\n\n")

        # Clusters first
        for cluster_id in sorted(set(labels)):
            if cluster_id == -1:
                continue
            cluster_methods = cluster_df[cluster_df["cluster"] == cluster_id]
            f.write(f"\n{'='*80}\n")
            f.write(f"CLUSTER {cluster_id} ({len(cluster_methods)} methods)\n")
            f.write(f"{'='*80}\n\n")

            for _, row in cluster_methods.iterrows():
                prob = row['cluster_probability']
                f.write(f"  [{row['Index']}] {row['Method']} (prob: {prob:.2f})\n")

        # Noise points at end
        noise_methods = cluster_df[cluster_df["cluster"] == -1]
        if len(noise_methods) > 0:
            f.write(f"\n{'='*80}\n")
            f.write(f"NOISE / UNCLUSTERED ({len(noise_methods)} methods)\n")
            f.write(f"{'='*80}\n\n")
            for _, row in noise_methods.iterrows():
                f.write(f"  [{row['Index']}] {row['Method']}\n")

    logger.info(f"Saved HDBSCAN summary")

    # Save JSON (convert numpy types to native Python types)
    cluster_json = {
        "n_clusters": int(hdbscan_results["n_clusters"]),
        "n_noise": int(hdbscan_results["n_noise"]),
        "quality_metrics": metrics,
        "clusters": {}
    }

    for cluster_id in sorted(set(labels)):
        cluster_methods = cluster_df[cluster_df["cluster"] == cluster_id]
        key = f"cluster_{int(cluster_id)}" if cluster_id != -1 else "noise"
        # Convert DataFrame to records with native Python types
        methods_list = []
        for _, row in cluster_methods[["Index", "Method", "Description", "Source"]].iterrows():
            methods_list.append({
                "Index": int(row["Index"]),
                "Method": str(row["Method"]),
                "Description": str(row["Description"]),
                "Source": str(row["Source"])
            })
        cluster_json["clusters"][key] = {
            "methods": methods_list,
            "count": len(cluster_methods)
        }

    with open(output_dir / "hdbscan_clusters.json", "w") as f:
        json.dump(cluster_json, f, indent=2)

    return metrics


def try_hdbscan_parameters(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    output_dir: Path
):
    """Try different HDBSCAN parameters and report results."""

    # First do UMAP once
    logger.info("Running UMAP for parameter search...")
    reducer = umap.UMAP(
        n_components=5,
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )
    reduced = reducer.fit_transform(embeddings)

    # Try different parameter combinations
    param_results = []

    for min_cluster_size in [5, 8, 10, 15, 20, 25, 30]:
        for min_samples in [3, 5, 8, 10]:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean',
                cluster_selection_method='eom'
            )
            labels = clusterer.fit_predict(reduced)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = (labels == -1).sum()
            noise_pct = n_noise / len(labels) * 100

            # Calculate cluster sizes
            cluster_sizes = Counter(labels)
            if -1 in cluster_sizes:
                del cluster_sizes[-1]

            if cluster_sizes:
                min_size = min(cluster_sizes.values())
                max_size = max(cluster_sizes.values())
                avg_size = sum(cluster_sizes.values()) / len(cluster_sizes)
            else:
                min_size = max_size = avg_size = 0

            param_results.append({
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "noise_pct": noise_pct,
                "min_cluster": min_size,
                "max_cluster": max_size,
                "avg_cluster": avg_size
            })

    # Save parameter search results
    param_df = pd.DataFrame(param_results)
    param_df.to_csv(output_dir / "hdbscan_parameter_search.csv", index=False)

    # Print summary
    print("\nHDBSCAN Parameter Search Results:")
    print("=" * 100)
    print(f"{'min_cluster_size':>16} {'min_samples':>12} {'n_clusters':>10} {'n_noise':>8} {'noise%':>8} {'min':>6} {'max':>6} {'avg':>8}")
    print("-" * 100)
    for r in param_results:
        print(f"{r['min_cluster_size']:>16} {r['min_samples']:>12} {r['n_clusters']:>10} {r['n_noise']:>8} {r['noise_pct']:>7.1f}% {r['min_cluster']:>6} {r['max_cluster']:>6} {r['avg_cluster']:>8.1f}")

    return param_results, reduced


def main():
    """Main entry point."""
    logger.info("Starting advanced semantic clustering analysis")

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load data
    df, embeddings = load_data()

    # =========================================================================
    # 1. Hierarchical Clustering
    # =========================================================================
    print("\n" + "=" * 80)
    print("HIERARCHICAL CLUSTERING")
    print("=" * 80)

    hier_results = hierarchical_clustering(embeddings, method="ward")
    hier_metrics = save_hierarchical_results(df, embeddings, hier_results, OUTPUT_DIR)

    print(f"\nSaved dendrogram and cluster assignments for k = {list(hier_results['cluster_assignments'].keys())}")

    # Print hierarchical quality metrics summary
    print("\nHierarchical Clustering Quality Metrics:")
    print("-" * 70)
    print(f"{'k':>4} {'Silhouette':>12} {'Calinski-H':>12} {'Davies-B':>10}")
    print("-" * 70)
    for k in sorted(hier_metrics.keys()):
        m = hier_metrics[k]
        print(f"{k:>4} {m['silhouette_score']:>12.3f} {m['calinski_harabasz_score']:>12.1f} {m['davies_bouldin_score']:>10.3f}")

    # =========================================================================
    # 2. UMAP + HDBSCAN with parameter search
    # =========================================================================
    print("\n" + "=" * 80)
    print("UMAP + HDBSCAN CLUSTERING")
    print("=" * 80)

    # First do parameter search
    param_results, reduced = try_hdbscan_parameters(embeddings, df, OUTPUT_DIR)

    # Find a good parameter set (reasonable number of clusters, not too much noise)
    # Target: 10-20 clusters with <30% noise
    good_params = [
        r for r in param_results
        if 8 <= r["n_clusters"] <= 25 and r["noise_pct"] < 35
    ]

    if good_params:
        # Pick one with most clusters within constraints
        best = max(good_params, key=lambda x: x["n_clusters"])
        min_cluster_size = best["min_cluster_size"]
        min_samples = best["min_samples"]
        print(f"\nSelected parameters: min_cluster_size={min_cluster_size}, min_samples={min_samples}")
    else:
        # Default to reasonable values
        min_cluster_size = 15
        min_samples = 5
        print(f"\nUsing default parameters: min_cluster_size={min_cluster_size}, min_samples={min_samples}")

    # Run final HDBSCAN with selected parameters
    hdbscan_results = umap_hdbscan_clustering(
        embeddings,
        n_components=5,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples
    )
    hdbscan_metrics = save_hdbscan_results(df, embeddings, hdbscan_results, OUTPUT_DIR)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"\nHierarchical clustering: dendrogram.png + cluster files for k={list(hier_results['cluster_assignments'].keys())}")
    print(f"HDBSCAN: {hdbscan_results['n_clusters']} clusters, {hdbscan_results['n_noise']} noise points")

    # Print HDBSCAN quality metrics
    print("\nHDBSCAN Quality Metrics:")
    print("-" * 50)
    if hdbscan_metrics['silhouette_score'] is not None:
        sil = hdbscan_metrics['silhouette_score']
        print(f"  Silhouette Score:      {sil:.3f}", end="")
        if sil >= 0.7:
            print(" (Strong structure)")
        elif sil >= 0.5:
            print(" (Reasonable structure)")
        elif sil >= 0.25:
            print(" (Weak structure)")
        else:
            print(" (No clear structure)")
        print(f"  Calinski-Harabasz:     {hdbscan_metrics['calinski_harabasz_score']:.1f}")
        db = hdbscan_metrics['davies_bouldin_score']
        print(f"  Davies-Bouldin:        {db:.3f}", end="")
        if db < 1:
            print(" (Good)")
        else:
            print(" (Could be better)")
    else:
        print("  (Not enough clusters to compute metrics)")

    # Find best hierarchical k by silhouette
    best_k = max(hier_metrics.keys(), key=lambda k: hier_metrics[k]['silhouette_score'])
    best_sil = hier_metrics[best_k]['silhouette_score']
    print(f"\nBest Hierarchical k by Silhouette: k={best_k} (silhouette={best_sil:.3f})")

    print(f"\nRecommendation: Check the dendrogram.png to see natural cluster boundaries,")
    print(f"and compare with HDBSCAN results for different perspectives on the data.")
    print(f"\nQuality metrics saved to: hierarchical_quality_metrics.txt")


if __name__ == "__main__":
    main()
