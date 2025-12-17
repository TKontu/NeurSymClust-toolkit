#!/usr/bin/env python3
"""Inspect the compatibility checkpoint structure"""

import pickle
import json

def inspect_checkpoint(pkl_path):
    """Inspect the checkpoint structure"""
    print(f"Loading {pkl_path}...")

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    print(f"\nTop-level keys: {list(data.keys())}")

    for key in data.keys():
        print(f"\n{key}:")
        print(f"  Type: {type(data[key])}")

        if isinstance(data[key], dict):
            print(f"  Length: {len(data[key])}")
            print(f"  Keys (first 5): {list(data[key].keys())[:5]}")

            # Show a sample entry
            if len(data[key]) > 0:
                sample_key = list(data[key].keys())[0]
                sample_value = data[key][sample_key]
                print(f"\n  Sample entry:")
                print(f"    Key: {sample_key}")
                print(f"    Value type: {type(sample_value)}")
                print(f"    Value: {sample_value}")

        elif isinstance(data[key], list):
            print(f"  Length: {len(data[key])}")
            if len(data[key]) > 0:
                print(f"  First item type: {type(data[key][0])}")
                print(f"  First item: {data[key][0]}")
        elif isinstance(data[key], set):
            print(f"  Length: {len(data[key])}")
            print(f"  First 3 items: {list(data[key])[:3]}")
        else:
            print(f"  Value: {data[key]}")

if __name__ == '__main__':
    inspect_checkpoint('results/compatibility_checkpoint.pkl')
