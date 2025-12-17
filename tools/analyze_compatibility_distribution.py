#!/usr/bin/env python3
"""Analyze the distribution of compatibility scores"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

def analyze_distribution(pkl_path):
    """Analyze compatibility score distribution"""
    print(f"Loading {pkl_path}...")

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    results = data.get('results', [])
    print(f"Total results: {len(results)}")

    # Extract scores
    scores = [r.get('compatibility_score', 0) for r in results]
    relationship_types = [r.get('relationship_type', 'unknown') for r in results]

    print(f"\nCompatibility Score Statistics:")
    print(f"  Mean: {np.mean(scores):.3f}")
    print(f"  Median: {np.median(scores):.3f}")
    print(f"  Std Dev: {np.std(scores):.3f}")
    print(f"  Min: {np.min(scores):.3f}")
    print(f"  Max: {np.max(scores):.3f}")

    print(f"\nPercentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  {p}th: {np.percentile(scores, p):.3f}")

    print(f"\nRelationship Type Distribution:")
    from collections import Counter
    type_counts = Counter(relationship_types)
    for rel_type, count in type_counts.most_common():
        pct = count / len(results) * 100
        print(f"  {rel_type}: {count} ({pct:.1f}%)")

    print(f"\nScore Distribution by Bins:")
    bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    for i in range(len(bins) - 1):
        count = sum(1 for s in scores if bins[i] <= s < bins[i+1])
        pct = count / len(scores) * 100
        print(f"  {bins[i]:.2f} - {bins[i+1]:.2f}: {count} ({pct:.1f}%)")

    # Count for >= 1.0
    count_max = sum(1 for s in scores if s >= 1.0)
    if count_max > 0:
        pct = count_max / len(scores) * 100
        print(f"  >= 1.00: {count_max} ({pct:.1f}%)")

    # Recommended thresholds
    print(f"\nRecommended Thresholds (based on distribution):")
    print(f"  Incompatible: score < 0.70 → {sum(1 for s in scores if s < 0.70)} pairs ({sum(1 for s in scores if s < 0.70)/len(scores)*100:.1f}%)")
    print(f"  Compatible: 0.70 <= score < 0.95 → {sum(1 for s in scores if 0.70 <= s < 0.95)} pairs ({sum(1 for s in scores if 0.70 <= s < 0.95)/len(scores)*100:.1f}%)")
    print(f"  Synergistic: score >= 0.95 → {sum(1 for s in scores if s >= 0.95)} pairs ({sum(1 for s in scores if s >= 0.95)/len(scores)*100:.1f}%)")

    # Plot distribution
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(scores, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Compatibility Score')
    plt.ylabel('Count')
    plt.title('Distribution of Compatibility Scores')
    plt.axvline(0.7, color='r', linestyle='--', label='Incompatible threshold (0.7)')
    plt.axvline(0.95, color='g', linestyle='--', label='Synergistic threshold (0.95)')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(scores, bins=50, edgecolor='black', alpha=0.7, cumulative=True, density=True)
    plt.xlabel('Compatibility Score')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution')
    plt.axvline(0.7, color='r', linestyle='--')
    plt.axvline(0.95, color='g', linestyle='--')
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/compatibility_distribution.png', dpi=150)
    print(f"\nSaved distribution plot to results/compatibility_distribution.png")

if __name__ == '__main__':
    analyze_distribution('results/compatibility_checkpoint.pkl')
