#!/usr/bin/env python3
"""
Analyze cluster membership probabilities from HDBSCAN clustering results
"""

import re
import pandas as pd
import numpy as np
from collections import defaultdict

def parse_cluster_file(file_path):
    """Parse the HDBSCAN cluster summary file and extract probability statistics"""

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract cluster sections (handle both "CLUSTER" and "SECONDARY CLUSTER" formats)
    cluster_pattern = r'(?:SECONDARY )?CLUSTER\s+(\d+) \((\d+) methods\)[^\n]*\n=+\n\n((?:.*?\(prob: [\d.]+\)\n)+)'
    clusters = re.findall(cluster_pattern, content, re.MULTILINE)

    cluster_stats = []
    all_probabilities = []

    for cluster_id, method_count, methods_text in clusters:
        # Extract probabilities from each cluster
        prob_pattern = r'\(prob: ([\d.]+)\)'
        probabilities = [float(p) for p in re.findall(prob_pattern, methods_text)]

        if probabilities:
            stats = {
                'Cluster': f'Cluster {cluster_id}',
                'Method_Count': int(method_count),
                'Min_Prob': np.min(probabilities),
                'Max_Prob': np.max(probabilities),
                'Mean_Prob': np.mean(probabilities),
                'Median_Prob': np.median(probabilities),
                'Std_Prob': np.std(probabilities),
                'Prob_Range': np.max(probabilities) - np.min(probabilities),
                'Methods_Below_0.6': sum(1 for p in probabilities if p < 0.6),
                'Methods_Below_0.8': sum(1 for p in probabilities if p < 0.8),
                'Methods_Below_0.9': sum(1 for p in probabilities if p < 0.9),
                'Methods_At_1.0': sum(1 for p in probabilities if p >= 0.995),
                'Pct_High_Confidence': (sum(1 for p in probabilities if p >= 0.9) / len(probabilities)) * 100
            }
            cluster_stats.append(stats)
            all_probabilities.extend(probabilities)

    # Create DataFrame
    df = pd.DataFrame(cluster_stats)

    # Calculate overall statistics
    overall_stats = {
        'Total_Clusters': len(clusters),
        'Total_Methods_Clustered': sum(int(count) for _, count, _ in clusters),
        'Overall_Mean_Prob': np.mean(all_probabilities),
        'Overall_Median_Prob': np.median(all_probabilities),
        'Overall_Std_Prob': np.std(all_probabilities),
        'Overall_Min_Prob': np.min(all_probabilities),
        'Overall_Max_Prob': np.max(all_probabilities)
    }

    return df, overall_stats, all_probabilities


def create_probability_distribution(probabilities, bins=10):
    """Create a distribution of probabilities"""
    hist, bin_edges = np.histogram(probabilities, bins=bins, range=(0, 1))
    distribution = []
    for i in range(len(hist)):
        distribution.append({
            'Range': f'{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}',
            'Count': hist[i],
            'Percentage': (hist[i] / len(probabilities)) * 100
        })
    return pd.DataFrame(distribution)


def print_summary(df, overall_stats, distribution_df):
    """Print formatted summary"""
    print("="*80)
    print("CLUSTER PROBABILITY STATISTICS")
    print("="*80)
    print()

    print("OVERALL STATISTICS:")
    print("-"*80)
    for key, value in overall_stats.items():
        if isinstance(value, float):
            print(f"  {key:30s}: {value:.4f}")
        else:
            print(f"  {key:30s}: {value}")
    print()

    print("="*80)
    print("PER-CLUSTER STATISTICS:")
    print("="*80)
    print()

    # Format the DataFrame for display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.3f}'.format)

    print(df.to_string(index=False))
    print()

    print("="*80)
    print("PROBABILITY DISTRIBUTION:")
    print("="*80)
    print()
    print(distribution_df.to_string(index=False))
    print()

    print("="*80)
    print("CLUSTER QUALITY INSIGHTS:")
    print("="*80)
    print()

    # Identify high-quality clusters (high mean probability, low std)
    high_quality = df[(df['Mean_Prob'] >= 0.9) & (df['Std_Prob'] <= 0.1)]
    print(f"High-quality clusters (mean ≥ 0.9, std ≤ 0.1): {len(high_quality)}")
    if len(high_quality) > 0:
        print(high_quality[['Cluster', 'Method_Count', 'Mean_Prob', 'Std_Prob']].to_string(index=False))
    print()

    # Identify problematic clusters (low mean probability or high variance)
    problematic = df[(df['Mean_Prob'] < 0.7) | (df['Std_Prob'] > 0.2)]
    print(f"Problematic clusters (mean < 0.7 or std > 0.2): {len(problematic)}")
    if len(problematic) > 0:
        print(problematic[['Cluster', 'Method_Count', 'Mean_Prob', 'Std_Prob', 'Min_Prob']].to_string(index=False))
    print()

    # Clusters with weak members
    weak_members = df[df['Methods_Below_0.6'] > 0]
    print(f"Clusters with weak members (prob < 0.6): {len(weak_members)}")
    if len(weak_members) > 0:
        print(weak_members[['Cluster', 'Method_Count', 'Methods_Below_0.6', 'Min_Prob']].to_string(index=False))
    print()


def save_to_excel(df, overall_stats, distribution_df, output_path):
    """Save statistics to Excel file"""
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Per-cluster statistics
        df.to_excel(writer, sheet_name='Cluster Statistics', index=False)

        # Overall statistics
        overall_df = pd.DataFrame([overall_stats])
        overall_df.to_excel(writer, sheet_name='Overall Statistics', index=False)

        # Distribution
        distribution_df.to_excel(writer, sheet_name='Probability Distribution', index=False)

    print(f"✓ Excel file saved: {output_path}")


if __name__ == '__main__':
    # Analyze both primary and noise clustering results
    analyses = [
        {
            'name': 'PRIMARY CLUSTERING (Advanced)',
            'input_file': 'results_semantic_clustering_advanced/hdbscan_clusters_summary.txt',
            'output_excel': 'results_semantic_clustering_advanced/cluster_probability_statistics.xlsx'
        },
        {
            'name': 'NOISE CLUSTERING',
            'input_file': 'results_semantic_clustering_noise/noise_clusters_summary_named.txt',
            'output_excel': 'results_semantic_clustering_noise/cluster_probability_statistics.xlsx'
        }
    ]

    all_results = {}

    for analysis in analyses:
        print("="*80)
        print(f"ANALYZING: {analysis['name']}")
        print("="*80)
        print()

        try:
            # Parse and analyze
            df, overall_stats, all_probabilities = parse_cluster_file(analysis['input_file'])
            distribution_df = create_probability_distribution(all_probabilities, bins=10)

            # Store results for comparison
            all_results[analysis['name']] = {
                'df': df,
                'overall_stats': overall_stats,
                'distribution_df': distribution_df
            }

            # Print summary
            print_summary(df, overall_stats, distribution_df)

            # Save to Excel
            save_to_excel(df, overall_stats, distribution_df, analysis['output_excel'])

            print()
            print()

        except FileNotFoundError:
            print(f"ERROR: File not found: {analysis['input_file']}")
            print()
            print()
        except Exception as e:
            print(f"ERROR: {e}")
            print()
            print()

    # Print comparative summary
    if len(all_results) == 2:
        print("="*80)
        print("COMPARATIVE SUMMARY")
        print("="*80)
        print()

        primary_name = analyses[0]['name']
        noise_name = analyses[1]['name']

        if primary_name in all_results and noise_name in all_results:
            primary_stats = all_results[primary_name]['overall_stats']
            noise_stats = all_results[noise_name]['overall_stats']

            comparison = pd.DataFrame({
                'Metric': [
                    'Total Clusters',
                    'Total Methods',
                    'Mean Probability',
                    'Median Probability',
                    'Std Probability',
                    'Min Probability',
                    'Max Probability'
                ],
                primary_name: [
                    primary_stats['Total_Clusters'],
                    primary_stats['Total_Methods_Clustered'],
                    f"{primary_stats['Overall_Mean_Prob']:.4f}",
                    f"{primary_stats['Overall_Median_Prob']:.4f}",
                    f"{primary_stats['Overall_Std_Prob']:.4f}",
                    f"{primary_stats['Overall_Min_Prob']:.4f}",
                    f"{primary_stats['Overall_Max_Prob']:.4f}"
                ],
                noise_name: [
                    noise_stats['Total_Clusters'],
                    noise_stats['Total_Methods_Clustered'],
                    f"{noise_stats['Overall_Mean_Prob']:.4f}",
                    f"{noise_stats['Overall_Median_Prob']:.4f}",
                    f"{noise_stats['Overall_Std_Prob']:.4f}",
                    f"{noise_stats['Overall_Min_Prob']:.4f}",
                    f"{noise_stats['Overall_Max_Prob']:.4f}"
                ]
            })

            print(comparison.to_string(index=False))
            print()

            # Calculate high-quality cluster percentages
            primary_df = all_results[primary_name]['df']
            noise_df = all_results[noise_name]['df']

            primary_high_quality = len(primary_df[(primary_df['Mean_Prob'] >= 0.9) & (primary_df['Std_Prob'] <= 0.1)])
            noise_high_quality = len(noise_df[(noise_df['Mean_Prob'] >= 0.9) & (noise_df['Std_Prob'] <= 0.1)])

            primary_pct = (primary_high_quality / len(primary_df)) * 100 if len(primary_df) > 0 else 0
            noise_pct = (noise_high_quality / len(noise_df)) * 100 if len(noise_df) > 0 else 0

            print(f"High-Quality Clusters (mean ≥ 0.9, std ≤ 0.1):")
            print(f"  {primary_name:30s}: {primary_high_quality:2d} / {len(primary_df):2d} ({primary_pct:5.1f}%)")
            print(f"  {noise_name:30s}: {noise_high_quality:2d} / {len(noise_df):2d} ({noise_pct:5.1f}%)")
            print()
