#!/usr/bin/env python3
"""
Identify weak clusters and extract their names from clustering results
"""

import re

def extract_weak_clusters(file_path, cluster_type="PRIMARY"):
    """Extract clusters with weak members (prob < 0.6) and their names"""

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern to match cluster headers with names
    if cluster_type == "PRIMARY":
        cluster_header_pattern = r'CLUSTER\s+(\d+)\s+\((\d+)\s+methods\)\n=+\n\n((?:.*?\(prob: [\d.]+\)\n)+)'
    else:  # NOISE
        cluster_header_pattern = r'SECONDARY CLUSTER\s+(\d+)\s+\((\d+)\s+methods\)\s+#\s+([^\n]+)\n=+\n\n((?:.*?\(prob: [\d.]+\)\n)+)'

    clusters = re.findall(cluster_header_pattern, content, re.MULTILINE)

    weak_clusters = []

    for match in clusters:
        if cluster_type == "PRIMARY":
            cluster_id, method_count, methods_text = match
            cluster_name = f"Cluster {cluster_id}"
        else:
            cluster_id, method_count, cluster_name, methods_text = match

        # Extract probabilities
        prob_pattern = r'\[(\d+)\]\s+([^\(]+)\s+\(prob: ([\d.]+)\)'
        methods_with_probs = re.findall(prob_pattern, methods_text)

        # Find weak members
        weak_members = []
        all_probs = []

        for method_id, method_name, prob_str in methods_with_probs:
            prob = float(prob_str)
            all_probs.append(prob)
            if prob < 0.6:
                weak_members.append({
                    'id': method_id,
                    'name': method_name.strip(),
                    'prob': prob
                })

        if weak_members:
            weak_clusters.append({
                'cluster_id': cluster_id,
                'cluster_name': cluster_name,
                'method_count': int(method_count),
                'weak_count': len(weak_members),
                'min_prob': min(all_probs),
                'mean_prob': sum(all_probs) / len(all_probs),
                'weak_members': weak_members
            })

    return weak_clusters


def print_weak_clusters_report(primary_weak, noise_weak):
    """Print formatted report of weak clusters"""

    print("="*80)
    print("WEAK CLUSTERS ANALYSIS")
    print("Clusters with methods having probability < 0.6")
    print("="*80)
    print()

    # PRIMARY CLUSTERING
    print("="*80)
    print("PRIMARY CLUSTERING (Advanced)")
    print("="*80)
    print()

    if primary_weak:
        for cluster in primary_weak:
            print(f"Cluster {cluster['cluster_id']}: {cluster['cluster_name']}")
            print(f"  Total methods: {cluster['method_count']}")
            print(f"  Weak members: {cluster['weak_count']}")
            print(f"  Mean probability: {cluster['mean_prob']:.3f}")
            print(f"  Min probability: {cluster['min_prob']:.3f}")
            print()
            print("  Weak members (prob < 0.6):")
            for member in cluster['weak_members']:
                print(f"    [{member['id']}] {member['name']} (prob: {member['prob']:.2f})")
            print()
            print("-"*80)
            print()
    else:
        print("  No clusters with weak members found!")
        print()

    # NOISE CLUSTERING
    print("="*80)
    print("NOISE CLUSTERING (Secondary)")
    print("="*80)
    print()

    if noise_weak:
        for cluster in noise_weak:
            print(f"Cluster {cluster['cluster_id']}: {cluster['cluster_name']}")
            print(f"  Total methods: {cluster['method_count']}")
            print(f"  Weak members: {cluster['weak_count']}")
            print(f"  Mean probability: {cluster['mean_prob']:.3f}")
            print(f"  Min probability: {cluster['min_prob']:.3f}")
            print()
            print("  Weak members (prob < 0.6):")
            for member in cluster['weak_members']:
                print(f"    [{member['id']}] {member['name']} (prob: {member['prob']:.2f})")
            print()
            print("-"*80)
            print()
    else:
        print("  No clusters with weak members found!")
        print()

    # SUMMARY
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()
    print(f"PRIMARY CLUSTERING:")
    print(f"  Clusters with weak members: {len(primary_weak)}")
    total_weak_primary = sum(c['weak_count'] for c in primary_weak)
    print(f"  Total weak methods: {total_weak_primary}")
    print()
    print(f"NOISE CLUSTERING:")
    print(f"  Clusters with weak members: {len(noise_weak)}")
    total_weak_noise = sum(c['weak_count'] for c in noise_weak)
    print(f"  Total weak methods: {total_weak_noise}")
    print()
    print(f"OVERALL:")
    print(f"  Total clusters with weak members: {len(primary_weak) + len(noise_weak)}")
    print(f"  Total weak method assignments: {total_weak_primary + total_weak_noise}")
    print()


if __name__ == '__main__':
    primary_file = 'results_semantic_clustering_advanced/hdbscan_clusters_summary.txt'
    noise_file = 'results_semantic_clustering_noise/noise_clusters_summary_named.txt'

    print("Analyzing weak clusters...")
    print()

    # Extract weak clusters
    primary_weak = extract_weak_clusters(primary_file, cluster_type="PRIMARY")
    noise_weak = extract_weak_clusters(noise_file, cluster_type="NOISE")

    # Print report
    print_weak_clusters_report(primary_weak, noise_weak)
