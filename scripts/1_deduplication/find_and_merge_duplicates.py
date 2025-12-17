#!/usr/bin/env python3
"""
Find exact name duplicates and create merged dataset with LLM-unified descriptions.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path
import anthropic
import os

def find_exact_duplicates(csv_path):
    """Find methods with identical names."""
    methods_by_name = defaultdict(list)

    with open(csv_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 3 and parts[0].isdigit():
                idx = int(parts[0])
                name = parts[1].strip()
                description = parts[2].strip() if len(parts) > 2 else ''
                source = parts[3].strip() if len(parts) > 3 else ''

                methods_by_name[name].append({
                    'index': idx,
                    'name': name,
                    'description': description,
                    'source': source
                })

    # Find duplicates
    duplicates = {name: methods for name, methods in methods_by_name.items()
                  if len(methods) > 1}

    return duplicates

def merge_descriptions_with_llm(name, descriptions, sources):
    """Use Claude to merge multiple descriptions into one comprehensive version."""
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    prompt = f"""You are merging duplicate entries for a product development method.

Method Name: {name}

The following are {len(descriptions)} different descriptions from various sources:

"""
    for i, (desc, source) in enumerate(zip(descriptions, sources), 1):
        prompt += f"\n{i}. Source: {source}\n   Description: {desc}\n"

    prompt += """
Task: Create a single, comprehensive, and clear description that:
1. Captures all unique information from all versions
2. Removes redundancy and repetition
3. Maintains professional, encyclopedic tone
4. Is 2-3 sentences long
5. Focuses on the essence and key benefits of the method

Return ONLY the merged description, nothing else."""

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )

    return message.content[0].text.strip()

def main():
    csv_path = Path('input/methods.csv')

    print("Finding exact name duplicates...")
    duplicates = find_exact_duplicates(csv_path)

    print(f"\nFound {len(duplicates)} methods with exact name duplicates")
    print(f"Total duplicate entries: {sum(len(methods) for methods in duplicates.values())}")
    print()

    # Display duplicates
    for name, methods in sorted(duplicates.items()):
        print(f"\n{'='*80}")
        print(f"Method: {name}")
        print(f"Occurrences: {len(methods)}")
        print(f"Indices: {[m['index'] for m in methods]}")
        print()
        for m in methods:
            print(f"  [{m['index']}] Source: {m['source']}")
            print(f"       Description: {m['description'][:100]}...")

    # Ask user for confirmation
    print(f"\n{'='*80}")
    response = input("\nMerge these duplicates using LLM? (yes/no): ").strip().lower()

    if response != 'yes':
        print("Aborted.")
        return

    # Create merged dataset
    print("\nMerging duplicates with Claude...")

    # Read all original methods
    all_methods = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 3 and parts[0].isdigit():
                all_methods.append({
                    'index': int(parts[0]),
                    'name': parts[1].strip(),
                    'description': parts[2].strip(),
                    'source': parts[3].strip() if len(parts) > 3 else ''
                })

    # Track which indices to remove
    indices_to_remove = set()
    merged_methods = {}

    # Process each duplicate group
    for name, methods in duplicates.items():
        print(f"\nMerging: {name} ({len(methods)} entries)...")

        descriptions = [m['description'] for m in methods]
        sources = [m['source'] for m in methods]

        # Merge descriptions with LLM
        merged_desc = merge_descriptions_with_llm(name, descriptions, sources)

        # Keep the first occurrence, remove others
        keep_index = methods[0]['index']
        remove_indices = [m['index'] for m in methods[1:]]

        merged_methods[keep_index] = {
            'index': keep_index,
            'name': name,
            'description': merged_desc,
            'source': methods[0]['source'],  # Keep first source
            'merged_from': remove_indices
        }

        indices_to_remove.update(remove_indices)

        print(f"  ✓ Kept index {keep_index}, removed {remove_indices}")
        print(f"  Merged description: {merged_desc[:100]}...")

    # Create new dataset
    output_methods = []
    for method in all_methods:
        if method['index'] in indices_to_remove:
            continue  # Skip duplicates
        elif method['index'] in merged_methods:
            # Use merged version
            output_methods.append(merged_methods[method['index']])
        else:
            # Keep original
            output_methods.append(method)

    # Write merged dataset
    output_path = Path('input/methods_dedup.csv')
    with open(output_path, 'w', encoding='utf-8') as f:
        for m in output_methods:
            f.write(f"{m['index']}|{m['name']}|{m['description']}|{m['source']}\n")

    # Write merge report
    report = {
        'original_count': len(all_methods),
        'deduplicated_count': len(output_methods),
        'removed_count': len(indices_to_remove),
        'duplicate_groups': len(duplicates),
        'merges': [
            {
                'name': name,
                'kept_index': methods[0]['index'],
                'removed_indices': [m['index'] for m in methods[1:]],
                'merged_description': merged_methods[methods[0]['index']]['description']
            }
            for name, methods in duplicates.items()
        ]
    }

    report_path = Path('results/deduplication_report.json')
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*80}")
    print("DEDUPLICATION COMPLETE")
    print(f"{'='*80}")
    print(f"Original methods: {len(all_methods)}")
    print(f"Deduplicated methods: {len(output_methods)}")
    print(f"Removed duplicates: {len(indices_to_remove)}")
    print(f"\nOutput file: {output_path}")
    print(f"Report: {report_path}")

if __name__ == '__main__':
    main()
