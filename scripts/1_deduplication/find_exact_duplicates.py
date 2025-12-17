#!/usr/bin/env python3
"""Find exact name duplicates in the methods dataset."""
import csv
from collections import defaultdict
from pathlib import Path

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

def main():
    csv_path = Path('input/methods.csv')

    print("Finding exact name duplicates...")
    duplicates = find_exact_duplicates(csv_path)

    total_duplicates = sum(len(methods) for methods in duplicates.values())
    reduction = total_duplicates - len(duplicates)

    print(f"\n{'='*80}")
    print(f"EXACT NAME DUPLICATE ANALYSIS")
    print(f"{'='*80}")
    print(f"Unique method names with duplicates: {len(duplicates)}")
    print(f"Total duplicate entries: {total_duplicates}")
    print(f"Can be reduced by: {reduction} methods")
    print()

    # Display duplicates
    for name, methods in sorted(duplicates.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"\n{'─'*80}")
        print(f"Method: '{name}'")
        print(f"Occurrences: {len(methods)}")
        print(f"Indices: {[m['index'] for m in methods]}")
        print()
        for i, m in enumerate(methods, 1):
            print(f"  {i}. Index [{m['index']}] - Source: {m['source']}")
            print(f"     {m['description'][:150]}{'...' if len(m['description']) > 150 else ''}")
            print()

    print(f"{'='*80}")

if __name__ == '__main__':
    main()
