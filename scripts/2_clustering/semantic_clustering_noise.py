#!/usr/bin/env python3
"""
Cluster the noise points from the initial HDBSCAN run.

These 180 methods didn't fit into any cluster in the full dataset.
We attempt to find structure among them when analyzed separately.
"""

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
EMBEDDINGS_PATH = BASE_DIR / "results_semantic_clustering" / "embeddings.npy"
HDBSCAN_RESULTS_PATH = BASE_DIR / "results_semantic_clustering_advanced" / "hdbscan_clusters.csv"
INPUT_PATH = BASE_DIR / "input" / "methods_deduplicated.csv"
OUTPUT_DIR = BASE_DIR / "results_semantic_clustering_noise"


def load_noise_data() -> tuple[pd.DataFrame, np.ndarray, list[int]]:
    """Load only the noise points from previous clustering."""

    # Load the HDBSCAN results to get noise point indices
    hdbscan_df = pd.read_csv(HDBSCAN_RESULTS_PATH)
    noise_indices = hdbscan_df[hdbscan_df["cluster"] == -1]["Index"].tolist()

    logger.info(f"Found {len(noise_indices)} noise points from previous clustering")

    # Load original data
    df = pd.read_csv(INPUT_PATH, delimiter="|")

    # Load all embeddings
    all_embeddings = np.load(EMBEDDINGS_PATH)

    # Filter to noise points only
    # The Index column is 1-based, embeddings are 0-based by row order
    noise_df = df[df["Index"].isin(noise_indices)].copy()

    # Get embeddings for noise points (using row position, not Index)
    noise_mask = df["Index"].isin(noise_indices)
    noise_embeddings = all_embeddings[noise_mask.values]

    logger.info(f"Loaded {len(noise_df)} noise methods with embeddings shape {noise_embeddings.shape}")

    return noise_df, noise_embeddings, noise_indices


def try_hdbscan_parameters(embeddings: np.ndarray) -> tuple[list, np.ndarray]:
    """Try different HDBSCAN parameters for the smaller noise dataset."""

    logger.info("Running UMAP for noise points...")
    reducer = umap.UMAP(
        n_components=5,
        n_neighbors=10,  # Smaller neighborhood for smaller dataset
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )
    reduced = reducer.fit_transform(embeddings)

    # Try different parameter combinations - smaller values for smaller dataset
    param_results = []

    for min_cluster_size in [3, 4, 5, 6, 8, 10]:
        for min_samples in [2, 3, 4, 5]:
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

    # Print summary
    print("\nHDBSCAN Parameter Search (Noise Points):")
    print("=" * 100)
    print(f"{'min_cluster_size':>16} {'min_samples':>12} {'n_clusters':>10} {'n_noise':>8} {'noise%':>8} {'min':>6} {'max':>6} {'avg':>8}")
    print("-" * 100)
    for r in param_results:
        print(f"{r['min_cluster_size']:>16} {r['min_samples']:>12} {r['n_clusters']:>10} {r['n_noise']:>8} {r['noise_pct']:>7.1f}% {r['min_cluster']:>6} {r['max_cluster']:>6} {r['avg_cluster']:>8.1f}")

    return param_results, reduced


def run_clustering(
    embeddings: np.ndarray,
    min_cluster_size: int,
    min_samples: int
) -> dict:
    """Run UMAP + HDBSCAN with specified parameters."""

    logger.info(f"Running UMAP (n_components=5, n_neighbors=10)")

    # UMAP for clustering
    reducer = umap.UMAP(
        n_components=5,
        n_neighbors=10,
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )
    reduced = reducer.fit_transform(embeddings)

    # UMAP for visualization
    reducer_2d = umap.UMAP(
        n_components=2,
        n_neighbors=10,
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )
    reduced_2d = reducer_2d.fit_transform(embeddings)

    logger.info(f"Running HDBSCAN (min_cluster_size={min_cluster_size}, min_samples={min_samples})")

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

    logger.info(f"Found {n_clusters} clusters, {n_noise} still noise")

    return {
        "reduced_5d": reduced,
        "reduced_2d": reduced_2d,
        "labels": labels,
        "probabilities": probabilities,
        "n_clusters": n_clusters,
        "n_noise": n_noise
    }


def save_results(
    df: pd.DataFrame,
    results: dict,
    output_dir: Path
):
    """Save clustering results."""
    output_dir.mkdir(exist_ok=True)

    labels = results["labels"]
    reduced_2d = results["reduced_2d"]

    # Visualization
    plt.figure(figsize=(14, 10))

    unique_labels = set(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for idx, label in enumerate(unique_labels):
        mask = labels == label
        if label == -1:
            plt.scatter(
                reduced_2d[mask, 0], reduced_2d[mask, 1],
                c='lightgray', alpha=0.5, s=20, label=f'Still Noise ({mask.sum()})'
            )
        else:
            plt.scatter(
                reduced_2d[mask, 0], reduced_2d[mask, 1],
                c=[colors[idx]], alpha=0.7, s=30, label=f'Cluster {label} ({mask.sum()})'
            )

    plt.title(f"Noise Points Clustering\n{results['n_clusters']} clusters found, {results['n_noise']} still unclustered")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "noise_clusters.png", dpi=150)
    plt.close()

    # Save cluster assignments
    cluster_df = df.copy()
    cluster_df["noise_cluster"] = labels
    cluster_df["cluster_probability"] = results["probabilities"]
    cluster_df["umap_x"] = reduced_2d[:, 0]
    cluster_df["umap_y"] = reduced_2d[:, 1]

    cluster_df["sort_key"] = cluster_df["noise_cluster"].apply(lambda x: 9999 if x == -1 else x)
    cluster_df = cluster_df.sort_values(["sort_key", "cluster_probability"], ascending=[True, False])
    cluster_df = cluster_df.drop(columns=["sort_key"])

    cluster_df.to_csv(output_dir / "noise_clusters.csv", index=False)

    # Summary file
    with open(output_dir / "noise_clusters_summary.txt", "w") as f:
        f.write("Clustering of Previously Unclustered (Noise) Methods\n")
        f.write("=" * 80 + "\n")
        f.write(f"Input: {len(df)} methods that were noise in initial HDBSCAN\n")
        f.write(f"New clusters found: {results['n_clusters']}\n")
        f.write(f"Still unclustered: {results['n_noise']}\n")
        f.write("=" * 80 + "\n\n")

        # Clusters
        for cluster_id in sorted(set(labels)):
            if cluster_id == -1:
                continue
            cluster_methods = cluster_df[cluster_df["noise_cluster"] == cluster_id]
            f.write(f"\n{'='*80}\n")
            f.write(f"NOISE CLUSTER {cluster_id} ({len(cluster_methods)} methods)\n")
            f.write(f"{'='*80}\n\n")

            for _, row in cluster_methods.iterrows():
                prob = row['cluster_probability']
                f.write(f"  [{row['Index']}] {row['Method']} (prob: {prob:.2f})\n")

        # Remaining noise
        still_noise = cluster_df[cluster_df["noise_cluster"] == -1]
        if len(still_noise) > 0:
            f.write(f"\n{'='*80}\n")
            f.write(f"STILL UNCLUSTERED ({len(still_noise)} methods)\n")
            f.write(f"{'='*80}\n\n")
            for _, row in still_noise.iterrows():
                f.write(f"  [{row['Index']}] {row['Method']}\n")

    # JSON output
    cluster_json = {
        "n_clusters": int(results["n_clusters"]),
        "n_still_noise": int(results["n_noise"]),
        "clusters": {}
    }

    for cluster_id in sorted(set(labels)):
        cluster_methods = cluster_df[cluster_df["noise_cluster"] == cluster_id]
        key = f"noise_cluster_{int(cluster_id)}" if cluster_id != -1 else "still_noise"
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

    with open(output_dir / "noise_clusters.json", "w") as f:
        json.dump(cluster_json, f, indent=2)

    logger.info(f"Saved results to {output_dir}")


def main():
    """Main entry point."""
    logger.info("Starting noise points clustering")

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load noise data
    noise_df, noise_embeddings, noise_indices = load_noise_data()

    print(f"\nAnalyzing {len(noise_df)} methods that were noise in initial clustering")
    print("=" * 80)

    # Parameter search
    param_results, _ = try_hdbscan_parameters(noise_embeddings)

    # Save parameter search
    param_df = pd.DataFrame(param_results)
    param_df.to_csv(OUTPUT_DIR / "parameter_search.csv", index=False)

    # Select best parameters
    # For noise data, we want more clusters with reasonable noise
    good_params = [
        r for r in param_results
        if r["n_clusters"] >= 5 and r["noise_pct"] < 50
    ]

    if good_params:
        best = max(good_params, key=lambda x: x["n_clusters"])
        min_cluster_size = best["min_cluster_size"]
        min_samples = best["min_samples"]
        print(f"\nSelected parameters: min_cluster_size={min_cluster_size}, min_samples={min_samples}")
    else:
        # Fallback to small values
        min_cluster_size = 5
        min_samples = 3
        print(f"\nUsing fallback parameters: min_cluster_size={min_cluster_size}, min_samples={min_samples}")

    # Run clustering
    results = run_clustering(noise_embeddings, min_cluster_size, min_samples)

    # Save results
    save_results(noise_df, results, OUTPUT_DIR)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nInput: {len(noise_df)} noise points from initial clustering")
    print(f"New clusters found: {results['n_clusters']}")
    print(f"Still unclustered: {results['n_noise']} ({results['n_noise']/len(noise_df)*100:.1f}%)")
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
