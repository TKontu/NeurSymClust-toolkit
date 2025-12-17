#!/usr/bin/env python3
"""
Value-Based Method Analysis
Analyzes methods based on different organizational value profiles and creates implementation roadmaps
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import argparse


# Phase 1: Define "Value" for Different Contexts
VALUE_PROFILES = {
    'startup': {
        'name': 'Startup/Scale-up',
        'needs': 'Fast impact, low resources, external focus',
        'formula': lambda df: (
            df['impact_potential'] * 0.25 +
            df['time_to_value'] * 0.25 +
            (100 - df['resources_required']) * 0.20 +
            df['purpose_orientation'] * 0.20 +
            df['ease_adoption'] * 0.10
        )
    },
    'enterprise': {
        'name': 'Large Enterprise',
        'needs': 'Scalable, process-driven, manageable change',
        'formula': lambda df: (
            df['applicability'] * 0.25 +
            df['scope'] * 0.20 +
            df['process_focus'] * 0.20 +
            df['impact_potential'] * 0.20 +
            (100 - df['change_management_difficulty']) * 0.15
        )
    },
    'transformation': {
        'name': 'Digital Transformation',
        'needs': 'Strategic, long-term, high impact',
        'formula': lambda df: (
            df['impact_potential'] * 0.30 +
            df['scope'] * 0.25 +
            df['temporality'] * 0.25 +
            df['purpose_orientation'] * 0.20
        )
    },
    'quick_wins': {
        'name': 'Quick Wins Needed',
        'needs': 'Fast, easy, visible impact',
        'formula': lambda df: (
            df['time_to_value'] * 0.35 +
            df['ease_adoption'] * 0.30 +
            df['impact_potential'] * 0.20 +
            (100 - df['implementation_difficulty']) * 0.15
        )
    }
}


def load_method_data(json_path: str) -> pd.DataFrame:
    """Load 12D analysis results from JSON"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data['methods'])
    print(f"✓ Loaded {len(df)} methods from {json_path}")
    return df


def calculate_value_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate value scores for all profiles"""

    print("\nCalculating value scores for different profiles...")

    for profile_id, profile in VALUE_PROFILES.items():
        score_col = f'value_{profile_id}'
        df[score_col] = profile['formula'](df)
        print(f"  ✓ {profile['name']}: {score_col}")

    return df


# Phase 2: Multi-Criteria Analysis Framework
def find_highest_value_methods(df: pd.DataFrame, top_n: int = 20) -> dict:
    """Find highest value methods across different criteria"""

    print(f"\nFinding top {top_n} methods by multiple criteria...")

    results = {}

    # 1. ABSOLUTE WINNERS - Top N by pure impact
    results['pure_impact'] = df.nlargest(top_n, 'impact_potential')[
        ['name', 'impact_potential', 'implementation_difficulty', 'time_to_value', 'scope']
    ].to_dict('records')
    print(f"  ✓ Pure Impact: {len(results['pure_impact'])} methods")

    # 2. EASY WINS - High impact, low difficulty
    df['easy_win_score'] = df['impact_potential'] - df['implementation_difficulty']
    results['easy_wins'] = df.nlargest(top_n, 'easy_win_score')[
        ['name', 'impact_potential', 'implementation_difficulty', 'easy_win_score', 'time_to_value']
    ].to_dict('records')
    print(f"  ✓ Easy Wins: {len(results['easy_wins'])} methods")

    # 3. QUICK STRATEGIC - Fast + Strategic
    df['quick_strategic'] = (
        df['time_to_value'] * 0.5 +
        df['scope'] * 0.3 +
        df['impact_potential'] * 0.2
    )
    results['quick_strategic'] = df.nlargest(top_n, 'quick_strategic')[
        ['name', 'quick_strategic', 'time_to_value', 'scope', 'impact_potential']
    ].to_dict('records')
    print(f"  ✓ Quick Strategic: {len(results['quick_strategic'])} methods")

    # 4. UNIVERSAL METHODS - Work everywhere
    universal = df[df['applicability'] > 70].nlargest(top_n, 'impact_potential')
    results['universal'] = universal[
        ['name', 'applicability', 'impact_potential', 'implementation_difficulty']
    ].to_dict('records')
    print(f"  ✓ Universal: {len(results['universal'])} methods")

    # 5. HIDDEN GEMS - Moderate difficulty but exceptional value
    hidden_gems = df[
        (df['implementation_difficulty'] < 60) &
        (df['impact_potential'] > 70) &
        (df['time_to_value'] > 50)
    ].nlargest(top_n, 'impact_potential')
    results['hidden_gems'] = hidden_gems[
        ['name', 'impact_potential', 'implementation_difficulty', 'time_to_value']
    ].to_dict('records')
    print(f"  ✓ Hidden Gems: {len(results['hidden_gems'])} methods")

    # 6. FOUNDATION BUILDERS - Universal, easy, applicable
    foundation = df[
        (df['applicability'] > 60) &
        (df['ease_adoption'] > 60) &
        (df['implementation_difficulty'] < 50)
    ].nlargest(top_n, 'impact_potential')
    results['foundation'] = foundation[
        ['name', 'applicability', 'ease_adoption', 'implementation_difficulty', 'impact_potential']
    ].to_dict('records')
    print(f"  ✓ Foundation Builders: {len(results['foundation'])} methods")

    return results


def create_implementation_roadmap(df: pd.DataFrame, top_n: int = 10) -> dict:
    """Create optimal implementation sequence in waves"""

    print(f"\nCreating implementation roadmap (top {top_n} per wave)...")

    roadmap = {
        'wave_1_foundation': [],
        'wave_2_quick_wins': [],
        'wave_3_scaling': [],
        'wave_4_transformation': []
    }

    used_indices = set()

    # Wave 1: Foundation (Easy + Universal)
    wave1 = df[
        (df['ease_adoption'] > 70) &
        (df['applicability'] > 60) &
        (df['implementation_difficulty'] < 40)
    ].nlargest(top_n, 'impact_potential')

    roadmap['wave_1_foundation'] = wave1[
        ['name', 'ease_adoption', 'applicability', 'implementation_difficulty', 'impact_potential']
    ].to_dict('records')
    used_indices.update(wave1.index)
    print(f"  ✓ Wave 1 - Foundation: {len(roadmap['wave_1_foundation'])} methods")

    # Wave 2: Quick Wins (Fast + Visible)
    wave2 = df[
        (df['time_to_value'] > 70) &
        (df['impact_potential'] > 60) &
        (~df.index.isin(used_indices))
    ].nlargest(top_n, 'impact_potential')

    roadmap['wave_2_quick_wins'] = wave2[
        ['name', 'time_to_value', 'impact_potential', 'implementation_difficulty']
    ].to_dict('records')
    used_indices.update(wave2.index)
    print(f"  ✓ Wave 2 - Quick Wins: {len(roadmap['wave_2_quick_wins'])} methods")

    # Wave 3: Scaling (Broader scope, more process)
    wave3 = df[
        (df['scope'] > 50) &
        (df['process_focus'] > 50) &
        (~df.index.isin(used_indices))
    ].nlargest(top_n, 'impact_potential')

    roadmap['wave_3_scaling'] = wave3[
        ['name', 'scope', 'process_focus', 'impact_potential', 'implementation_difficulty']
    ].to_dict('records')
    used_indices.update(wave3.index)
    print(f"  ✓ Wave 3 - Scaling: {len(roadmap['wave_3_scaling'])} methods")

    # Wave 4: Transformation (High impact, strategic)
    wave4 = df[
        (df['scope'] > 70) &
        (df['temporality'] > 70) &
        (df['impact_potential'] > 80)
    ].nlargest(top_n, 'impact_potential')

    roadmap['wave_4_transformation'] = wave4[
        ['name', 'scope', 'temporality', 'impact_potential', 'implementation_difficulty']
    ].to_dict('records')
    print(f"  ✓ Wave 4 - Transformation: {len(roadmap['wave_4_transformation'])} methods")

    return roadmap


def find_top_by_profile(df: pd.DataFrame, top_n: int = 20) -> dict:
    """Find top methods for each organizational profile"""

    print(f"\nFinding top {top_n} methods for each value profile...")

    profile_results = {}

    for profile_id, profile in VALUE_PROFILES.items():
        score_col = f'value_{profile_id}'
        top_methods = df.nlargest(top_n, score_col)

        profile_results[profile_id] = {
            'profile_name': profile['name'],
            'profile_needs': profile['needs'],
            'top_methods': top_methods[
                ['name', score_col, 'impact_potential', 'implementation_difficulty', 'scope', 'time_to_value']
            ].to_dict('records')
        }

        print(f"  ✓ {profile['name']}: {len(profile_results[profile_id]['top_methods'])} methods")

    return profile_results


def generate_summary_stats(df: pd.DataFrame, results: dict) -> dict:
    """Generate summary statistics for the analysis"""

    summary = {
        'total_methods': len(df),
        'profiles_analyzed': len(VALUE_PROFILES),
        'criteria_analyzed': len(results),
        'dimension_stats': {}
    }

    # Key dimension statistics
    key_dims = ['impact_potential', 'implementation_difficulty', 'time_to_value',
                'scope', 'applicability', 'ease_adoption']

    for dim in key_dims:
        if dim in df.columns:
            summary['dimension_stats'][dim] = {
                'mean': float(df[dim].mean()),
                'median': float(df[dim].median()),
                'std': float(df[dim].std()),
                'min': float(df[dim].min()),
                'max': float(df[dim].max())
            }

    return summary


def save_results(profile_results: dict, criteria_results: dict, roadmap: dict,
                summary: dict, output_dir: str = 'results'):
    """Save all results to JSON files"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save comprehensive results
    comprehensive_output = {
        'summary': summary,
        'value_profiles': profile_results,
        'multi_criteria_analysis': criteria_results,
        'implementation_roadmap': roadmap
    }

    output_file = output_path / 'value_analysis_results.json'
    with open(output_file, 'w') as f:
        json.dump(comprehensive_output, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")

    # Also save a simplified version for each profile
    for profile_id, data in profile_results.items():
        profile_file = output_path / f'value_profile_{profile_id}.json'
        with open(profile_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  ✓ {data['profile_name']} → {profile_file}")


def print_summary_report(profile_results: dict, criteria_results: dict, roadmap: dict):
    """Print a summary report to console"""

    print("\n" + "="*80)
    print("VALUE-BASED METHOD ANALYSIS SUMMARY")
    print("="*80)

    # Profile-based recommendations
    print("\n📊 TOP METHODS BY ORGANIZATIONAL PROFILE:")
    print("-" * 80)
    for profile_id, data in profile_results.items():
        print(f"\n{data['profile_name'].upper()}")
        print(f"Context: {data['profile_needs']}")
        print(f"Top 5 Methods:")
        for i, method in enumerate(data['top_methods'][:5], 1):
            score_col = f'value_{profile_id}'
            print(f"  {i}. {method['name'][:60]:60s} | Score: {method[score_col]:.1f}")

    # Multi-criteria analysis
    print("\n" + "="*80)
    print("🎯 MULTI-CRITERIA ANALYSIS:")
    print("-" * 80)

    for criterion, methods in criteria_results.items():
        criterion_name = criterion.replace('_', ' ').title()
        print(f"\n{criterion_name} (Top 5):")
        for i, method in enumerate(methods[:5], 1):
            print(f"  {i}. {method['name'][:70]}")

    # Implementation roadmap
    print("\n" + "="*80)
    print("🚀 IMPLEMENTATION ROADMAP:")
    print("-" * 80)

    waves = [
        ('wave_1_foundation', 'Wave 1: Foundation (Easy, Universal)'),
        ('wave_2_quick_wins', 'Wave 2: Quick Wins (Fast, Visible)'),
        ('wave_3_scaling', 'Wave 3: Scaling (Process, Scope)'),
        ('wave_4_transformation', 'Wave 4: Transformation (Strategic)')
    ]

    for wave_key, wave_name in waves:
        print(f"\n{wave_name}:")
        methods = roadmap[wave_key]
        if methods:
            for i, method in enumerate(methods[:5], 1):
                print(f"  {i}. {method['name'][:70]}")
        else:
            print("  (No methods match criteria)")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze methods based on organizational value profiles'
    )
    parser.add_argument(
        '--input',
        default='results/method_scores_12d_deduplicated.json',
        help='Input JSON file with 12D analysis results'
    )
    parser.add_argument(
        '--output',
        default='results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=20,
        help='Number of top methods to return per criterion (default: 20)'
    )
    parser.add_argument(
        '--roadmap-size',
        type=int,
        default=10,
        help='Number of methods per roadmap wave (default: 10)'
    )

    args = parser.parse_args()

    print("="*80)
    print("VALUE-BASED METHOD ANALYSIS")
    print("="*80)
    print(f"\nInput:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Top-N:  {args.top_n}")
    print(f"Roadmap Wave Size: {args.roadmap_size}")

    # Load data
    df = load_method_data(args.input)

    # Calculate value scores for all profiles
    df = calculate_value_scores(df)

    # Find top methods by profile
    profile_results = find_top_by_profile(df, args.top_n)

    # Multi-criteria analysis
    criteria_results = find_highest_value_methods(df, args.top_n)

    # Create implementation roadmap
    roadmap = create_implementation_roadmap(df, args.roadmap_size)

    # Generate summary statistics
    summary = generate_summary_stats(df, criteria_results)

    # Save results
    save_results(profile_results, criteria_results, roadmap, summary, args.output)

    # Print summary report
    print_summary_report(profile_results, criteria_results, roadmap)

    print("\n" + "="*80)
    print("✅ Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
