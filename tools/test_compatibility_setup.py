#!/usr/bin/env python3
"""
Test compatibility analysis setup and dependencies.
"""
import sys
from pathlib import Path

def check_dependencies():
    """Check if all required packages are installed."""
    print("="*80)
    print("CHECKING DEPENDENCIES")
    print("="*80)

    required = [
        'numpy',
        'pandas',
        'networkx',
        'aiohttp',
        'tenacity',
        'tqdm',
        'yaml',
        'plotly'
    ]

    missing = []
    for package in required:
        try:
            if package == 'yaml':
                __import__('yaml')
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing.append(package)

    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("   Install with: pip install " + " ".join(missing))
        return False
    else:
        print("\n✅ All dependencies installed")
        return True


def check_files():
    """Check if required files exist."""
    print("\n" + "="*80)
    print("CHECKING FILES")
    print("="*80)

    required_files = [
        'config.yaml',
        'input/methods_deduplicated.csv',
        'analyze_compatibility.py',
        'visualize_compatibility.py'
    ]

    optional_files = [
        'results/method_scores_12d_deduplicated.json'
    ]

    all_ok = True

    for file in required_files:
        if Path(file).exists():
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - MISSING")
            all_ok = False

    print("\nOptional files:")
    for file in optional_files:
        if Path(file).exists():
            print(f"✓ {file}")
        else:
            print(f"  {file} - not found (ok)")

    if all_ok:
        print("\n✅ All required files present")
    else:
        print("\n❌ Some required files missing")

    return all_ok


def check_config():
    """Check configuration file."""
    print("\n" + "="*80)
    print("CHECKING CONFIGURATION")
    print("="*80)

    try:
        import yaml

        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        # Check LLM settings
        if 'llm' in config:
            llm = config['llm']
            print(f"✓ LLM provider: {llm.get('provider', 'unknown')}")
            print(f"✓ LLM model: {llm.get('model', 'unknown')}")
            print(f"✓ Base URL: {llm.get('base_url', 'unknown')}")
            print(f"✓ Temperature: {llm.get('temperature', 'unknown')}")
            print(f"✓ Max concurrent: {llm.get('max_concurrent', 'unknown')}")

            # Validate settings
            warnings = []
            if llm.get('temperature', 0) > 0.3:
                warnings.append("Temperature > 0.3 may cause inconsistent JSON output")

            if llm.get('max_concurrent', 0) > 50:
                warnings.append("High concurrency may overload LLM server")

            if warnings:
                print("\n⚠️  Configuration warnings:")
                for w in warnings:
                    print(f"   - {w}")
            else:
                print("\n✅ Configuration looks good")

            return True
        else:
            print("❌ No 'llm' section in config.yaml")
            return False

    except FileNotFoundError:
        print("❌ config.yaml not found")
        return False
    except Exception as e:
        print(f"❌ Error reading config: {e}")
        return False


def check_data_format():
    """Check if input data is in correct format."""
    print("\n" + "="*80)
    print("CHECKING DATA FORMAT")
    print("="*80)

    try:
        import pandas as pd

        df = pd.read_csv('input/methods_deduplicated.csv', sep='|')

        required_columns = ['Method', 'Description', 'Source']

        print(f"✓ CSV loaded: {len(df)} rows")
        print(f"✓ Columns: {list(df.columns)}")

        missing_cols = [c for c in required_columns if c not in df.columns]

        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return False

        # Check for empty values
        for col in required_columns:
            empty_count = df[col].isna().sum()
            if empty_count > 0:
                print(f"⚠️  {empty_count} empty values in column '{col}'")

        # Show sample
        print("\nSample row:")
        print(f"  Method: {df.iloc[0]['Method']}")
        print(f"  Description: {df.iloc[0]['Description'][:80]}...")
        print(f"  Source: {df.iloc[0]['Source']}")

        print("\n✅ Data format is correct")
        return True

    except FileNotFoundError:
        print("❌ input/methods_deduplicated.csv not found")
        return False
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return False


def estimate_runtime():
    """Estimate runtime for analysis."""
    print("\n" + "="*80)
    print("RUNTIME ESTIMATION")
    print("="*80)

    try:
        import yaml

        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        concurrent = config['llm'].get('max_concurrent', 25)

        scenarios = [
            (200, "Small test"),
            (500, "Recommended"),
            (1000, "Comprehensive")
        ]

        print(f"Assuming {concurrent} concurrent requests:\n")

        for n_pairs, label in scenarios:
            # Each pair = 2 LLM calls (overlap + compatibility)
            total_calls = n_pairs * 2

            # Estimate: ~2 seconds per call with concurrency
            time_seconds = (total_calls / concurrent) * 2

            minutes = int(time_seconds / 60)
            seconds = int(time_seconds % 60)

            print(f"  {label:15s} ({n_pairs:4d} pairs): ~{minutes:2d}m {seconds:2d}s")

        print("\n💡 Tip: Start with 200 pairs for testing")

    except Exception as e:
        print(f"⚠️  Could not estimate runtime: {e}")


def main():
    print("\n" + "="*80)
    print("COMPATIBILITY ANALYSIS SETUP CHECK")
    print("="*80 + "\n")

    checks = [
        ("Dependencies", check_dependencies),
        ("Files", check_files),
        ("Configuration", check_config),
        ("Data Format", check_data_format)
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Error in {name} check: {e}")
            results.append((name, False))

    # Estimate runtime
    estimate_runtime()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    all_passed = all(result for _, result in results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:8s} {name}")

    if all_passed:
        print("\n✅ All checks passed! Ready to run compatibility analysis.")
        print("\nNext steps:")
        print("  1. Run small test:  python analyze_compatibility.py --max-pairs 200")
        print("  2. Review results:  cat results/compatibility_analysis.json | head -50")
        print("  3. Visualize:       python visualize_compatibility.py")
        return 0
    else:
        print("\n❌ Some checks failed. Please fix issues before running analysis.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
