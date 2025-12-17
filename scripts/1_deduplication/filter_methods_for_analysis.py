#!/usr/bin/env python3
"""
Filter methods to remove duplicates BEFORE 9D analysis.
This saves time and LLM cost by analyzing only unique methods.

Uses transitive closure on embedding similarity > 0.8 to find duplicate groups,
then uses LLM to select best name and synthesize unified description.
"""
import json
import csv
import yaml
import asyncio
import networkx as nx
from pathlib import Path
from typing import Dict, Set, List
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.duplicate_synthesizer import DuplicateSynthesizer


def load_duplicates(duplicates_path: str) -> List[Dict]:
    """Load duplicate pairs from JSON"""
    with open(duplicates_path, 'r') as f:
        data = json.load(f)
    return data['all_duplicate_pairs']


def load_methods_csv(csv_path: str) -> Dict[int, Dict]:
    """Load methods from CSV (pipe-delimited)"""
    methods = {}

    with open(csv_path, 'r', encoding='utf-8') as f:
        # Use pipe delimiter
        reader = csv.DictReader(f, delimiter='|')
        for row in reader:
            idx = int(row['Index'])  # Capital I
            methods[idx] = {
                'index': idx,
                'name': row['Method'].strip(),
                'description': row['Description'].strip(),
                'source': row['Source'].strip()
            }

    return methods


def build_duplicate_graph(pairs: List[Dict], threshold: float = 0.8) -> nx.Graph:
    """
    Build graph where edges exist if embedding similarity > threshold.
    """
    G = nx.Graph()

    for pair in pairs:
        sim = pair['embedding_similarity']

        if sim > threshold:
            idx1 = pair['method1_index']
            idx2 = pair['method2_index']
            name1 = pair['method1_name']
            name2 = pair['method2_name']

            # Add nodes with names
            if not G.has_node(idx1):
                G.add_node(idx1, name=name1)
            if not G.has_node(idx2):
                G.add_node(idx2, name=name2)

            # Add edge
            G.add_edge(idx1, idx2, similarity=sim)

    return G


def find_duplicate_groups(G: nx.Graph) -> Dict[int, Set[int]]:
    """Find connected components (transitive closure)"""
    groups = {}
    for i, component in enumerate(nx.connected_components(G)):
        groups[i] = component
    return groups


def build_groups_with_methods(groups: Dict[int, Set[int]],
                             all_methods: Dict[int, Dict],
                             G: nx.Graph) -> Dict[int, List[Dict]]:
    """
    Build groups with full method information for LLM synthesis.

    Returns:
        Dict mapping group_id to list of method dicts
    """
    groups_with_methods = {}

    for group_id, indices in groups.items():
        methods_in_group = []

        for idx in indices:
            if idx in all_methods:
                methods_in_group.append(all_methods[idx])
            elif G.has_node(idx):
                # Fallback: use name from graph
                methods_in_group.append({
                    'index': idx,
                    'name': G.nodes[idx]['name'],
                    'description': '',
                    'source': 'unknown'
                })

        if methods_in_group:
            groups_with_methods[group_id] = methods_in_group

    return groups_with_methods


async def synthesize_duplicate_groups(groups_with_methods: Dict[int, List[Dict]],
                                      config: dict) -> Dict[int, Dict]:
    """
    Use LLM to synthesize duplicate groups into canonical representations.

    Returns:
        Dict mapping group_id to synthesized method
    """
    synthesizer = DuplicateSynthesizer(config)

    # Synthesize all groups with aggressive batching
    # batch_size=10 means 10 groups per LLM call (60 groups = only 6 API calls)
    synthesized = await synthesizer.batch_synthesize_groups(
        groups_with_methods,
        batch_size=10,
        max_concurrent=4
    )

    return synthesized


def save_filtered_csv(methods: Dict[int, Dict],
                      kept_indices: Set[int],
                      output_path: str):
    """Save filtered methods to CSV (pipe-delimited)"""

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        # Use pipe delimiter and capital column names
        fieldnames = ['Index', 'Method', 'Description', 'Source']
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='|')
        writer.writeheader()

        for idx in sorted(kept_indices):
            if idx in methods:
                method = methods[idx]
                writer.writerow({
                    'Index': method['index'],
                    'Method': method['name'],
                    'Description': method['description'],
                    'Source': method['source']
                })


def save_filter_metadata(original_count: int,
                        filtered_count: int,
                        removed_count: int,
                        duplicate_groups: int,
                        removed_methods: List[Dict],
                        synthesis_stats: Dict,
                        output_path: str):
    """Save metadata about filtering"""

    metadata = {
        'original_count': original_count,
        'filtered_count': filtered_count,
        'removed_count': removed_count,
        'duplicate_groups': duplicate_groups,
        'similarity_threshold': 0.8,
        'selection_strategy': 'llm_synthesis',
        'synthesis': synthesis_stats,
        'removed_methods': removed_methods
    }

    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)


async def main():
    print("="*80)
    print("FILTER METHODS FOR 9D ANALYSIS")
    print("="*80)
    print("\nThis creates a deduplicated dataset for faster 9D analysis.")
    print("Using transitive closure on embedding similarity > 0.8\n")

    # Paths
    duplicates_path = "results/duplicates.json"
    methods_csv_path = "input/methods.csv"
    output_csv_path = "input/methods_deduplicated.csv"
    metadata_path = "results/filter_metadata.json"

    # Check if files exist
    if not Path(duplicates_path).exists():
        print(f"❌ Error: {duplicates_path} not found")
        print("   Please run the full analysis pipeline first to generate duplicates.json")
        return

    if not Path(methods_csv_path).exists():
        print(f"❌ Error: {methods_csv_path} not found")
        return

    # Load methods
    print(f"Loading methods from {methods_csv_path}...")
    methods = load_methods_csv(methods_csv_path)
    print(f"✓ Loaded {len(methods)} methods")

    # Load duplicate pairs
    print(f"\nLoading duplicate pairs from {duplicates_path}...")
    pairs = load_duplicates(duplicates_path)
    print(f"✓ Loaded {len(pairs)} duplicate pairs")

    # Build graph with threshold
    print("\nBuilding duplicate graph (similarity > 0.8)...")
    G = build_duplicate_graph(pairs, threshold=0.8)
    print(f"✓ Graph has {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Find connected components
    print("\nFinding duplicate groups (transitive closure)...")
    groups = find_duplicate_groups(G)
    print(f"✓ Found {len(groups)} duplicate groups")

    # Show largest groups
    if groups:
        print("\nLargest duplicate groups:")
        sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
        for i, (group_id, indices) in enumerate(sorted_groups[:5], 1):
            names = []
            for idx in list(indices)[:3]:
                if idx in methods:
                    names.append(methods[idx]['name'])
                elif G.has_node(idx):
                    names.append(G.nodes[idx]['name'])

            print(f"  {i}. Group {group_id}: {len(indices)} methods")
            for name in names:
                print(f"     - {name}")
            if len(indices) > 3:
                print(f"     ... and {len(indices)-3} more")

    # Load config for LLM synthesis
    print("\nLoading configuration...")
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print("✓ Config loaded")

    # Build groups with full method information
    print("\nBuilding duplicate groups with full method information...")
    groups_with_methods = build_groups_with_methods(groups, methods, G)
    print(f"✓ Built {len(groups_with_methods)} groups with method details")

    # Synthesize duplicates using LLM
    print("\nSynthesizing duplicate groups with LLM...")
    print("  (Selecting best names and creating unified descriptions)")
    synthesized = await synthesize_duplicate_groups(groups_with_methods, config)
    print(f"✓ Synthesized {len(synthesized)} groups")

    # Update methods with synthesized results and select representatives
    print("\nUpdating methods with synthesized versions...")
    representatives = set()
    synthesis_stats = {
        'groups_synthesized': 0,
        'methods_updated': 0
    }

    for group_id, synth_result in synthesized.items():
        group_indices = list(groups[group_id])

        # Select first index as representative
        rep_idx = group_indices[0]
        representatives.add(rep_idx)

        # Update the representative with synthesized name and description
        if rep_idx in methods:
            methods[rep_idx]['name'] = synth_result['name']
            methods[rep_idx]['description'] = synth_result['description']
            methods[rep_idx]['source'] = synth_result['source']
            methods[rep_idx]['synthesized'] = True
            methods[rep_idx]['original_count'] = synth_result.get('original_count', len(group_indices))

            synthesis_stats['groups_synthesized'] += 1
            synthesis_stats['methods_updated'] += 1

    print(f"✓ Updated {synthesis_stats['methods_updated']} methods with synthesized versions")

    # Build set of all indices to keep
    # = representatives + methods not in any duplicate group
    all_duplicate_indices = set(G.nodes())
    non_duplicate_indices = set(methods.keys()) - all_duplicate_indices
    kept_indices = representatives | non_duplicate_indices

    removed_indices = set(methods.keys()) - kept_indices
    removed_methods = [
        {
            'index': idx,
            'name': methods[idx]['name'],
            'reason': 'duplicate'
        }
        for idx in sorted(removed_indices) if idx in methods
    ]

    # Save filtered CSV
    print(f"\nSaving filtered methods to {output_csv_path}...")
    save_filtered_csv(methods, kept_indices, output_csv_path)
    print(f"✓ Saved {len(kept_indices)} unique methods")

    # Save metadata
    print(f"Saving filter metadata to {metadata_path}...")
    save_filter_metadata(
        len(methods),
        len(kept_indices),
        len(removed_indices),
        len(groups),
        removed_methods,
        synthesis_stats,
        metadata_path
    )
    print(f"✓ Saved metadata")

    # Print summary
    print("\n" + "="*80)
    print("FILTERING SUMMARY")
    print("="*80)
    print(f"\nOriginal methods:      {len(methods)}")
    print(f"Duplicate groups:      {len(groups)}")
    print(f"Removed duplicates:    {len(removed_indices)}")
    print(f"Unique methods kept:   {len(kept_indices)}")

    reduction_pct = (len(removed_indices) / len(methods)) * 100
    print(f"\nReduction:             {reduction_pct:.1f}%")
    print(f"LLM synthesized:       {synthesis_stats['groups_synthesized']} groups")

    time_saved = (len(removed_indices) / len(methods)) * 45  # Assuming 45 min for full analysis
    print(f"Estimated time saved:  ~{time_saved:.0f} minutes")

    # Show examples of synthesized methods
    synthesized_methods = [m for m in methods.values() if m.get('synthesized', False)]
    if synthesized_methods:
        print(f"\nExample synthesized methods (first 3):")
        for i, method in enumerate(synthesized_methods[:3], 1):
            print(f"\n  {i}. {method['name']} (merged {method['original_count']} duplicates)")
            desc_preview = method['description'][:100] + "..." if len(method['description']) > 100 else method['description']
            print(f"     {desc_preview}")

    if len(removed_indices) > 10:
        print(f"\nFirst 10 removed methods:")
        for method in removed_methods[:10]:
            print(f"  - [{method['index']}] {method['name']}")
        if len(removed_indices) > 10:
            print(f"  ... and {len(removed_indices)-10} more")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Run 9D analysis on deduplicated dataset:")
    print(f"   python analyze_9d_comprehensive.py --input {output_csv_path}")
    print(f"\n   This will analyze {len(kept_indices)} unique methods instead of {len(methods)}")
    print(f"   Estimated time: ~{45 * len(kept_indices) / len(methods):.0f} minutes (vs ~45 min for all)")
    print("\n2. Or run on full dataset (includes duplicates):")
    print("   python analyze_9d_comprehensive.py")


if __name__ == "__main__":
    asyncio.run(main())
