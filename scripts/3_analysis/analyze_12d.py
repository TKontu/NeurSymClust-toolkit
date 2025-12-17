#!/usr/bin/env python3
"""
Comprehensive 12-dimensional analysis of methods using ranking approach.
This replaces direct scoring with relative ranking on each dimension.

Dimensions analyzed:
1. Scope (Tactical → Strategic)
2. Temporality (Immediate → Evolutionary)
3. Ease of Adoption
4. Resources Required
5. Technical Complexity
6. Change Management Difficulty
7. Impact Potential
8. Time to Value
9. Applicability (Narrow → Universal)
10. People Focus (Technical → Human)
11. Process Focus (Ad-hoc → Systematic)
12. Purpose Orientation (Internal → External)

Time estimate: ~75-100 minutes for 595 methods (with 5-pass validation, 20 parallel chunks)
"""
import asyncio
import json
import yaml
import time
import numpy as np
from pathlib import Path
import sys
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data import load_methods
from src.ranking_analyzer import RankingLLMAnalyzer

# Set up logging to show progress
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)


def save_12d_results(results: dict, methods: list, output_path: str):
    """Save comprehensive 12D results to JSON"""

    # Build output structure
    output = {
        'metadata': {
            'total_methods': len(methods),
            'dimensions': [
                'scope',
                'temporality',
                'ease_adoption',
                'resources_required',
                'technical_complexity',
                'change_management_difficulty',
                'impact_potential',
                'time_to_value',
                'applicability',
                'implementation_difficulty',
                'people_focus',
                'process_focus',
                'purpose_orientation'
            ],
            'dimension_descriptions': {
                'scope': {
                    'description': 'Scope (Tactical → Strategic)',
                    'scale': '0-100',
                    'low': 'Tactical - Individual/team level, specific tasks, narrow focus',
                    'medium': 'Mixed - Multi-team/department, cross-functional coordination',
                    'high': 'Strategic - Enterprise-wide, organizational, systemic change',
                    'examples_low': 'Daily standup, Code review, Task estimation',
                    'examples_high': 'Digital transformation, Enterprise architecture, Innovation framework'
                },
                'temporality': {
                    'description': 'Temporality (Immediate → Evolutionary)',
                    'scale': '0-100',
                    'low': 'Immediate - Hours/days impact, operational, quick fixes',
                    'medium': 'Mixed - Weeks/months impact, iterative improvement',
                    'high': 'Evolutionary - Years impact, transformational, cultural change',
                    'examples_low': 'Bug triage, Daily deployment, Hotfix process',
                    'examples_high': 'Learning organization, Continuous improvement culture, Innovation mindset'
                },
                'ease_adoption': {
                    'description': 'Ease of Adoption',
                    'scale': '0-100',
                    'low': 'Hard to adopt - Significant learning curve, organizational resistance',
                    'medium': 'Moderate - Some training needed, manageable change',
                    'high': 'Easy to adopt - Minimal training, quick to implement',
                    'note': 'Higher score = easier to adopt'
                },
                'resources_required': {
                    'description': 'Resources Required',
                    'scale': '0-100',
                    'low': 'Low resources - Minimal investment, leverages existing capabilities',
                    'medium': 'Moderate resources - Some dedicated resources needed',
                    'high': 'High resources - Significant investment in people, tools, infrastructure',
                    'note': 'Higher score = more resources required (inverted from ranking for intuitive interpretation)'
                },
                'technical_complexity': {
                    'description': 'Technical Complexity',
                    'scale': '0-100',
                    'low': 'Simple - Low technical barriers, straightforward implementation',
                    'medium': 'Moderate - Some technical expertise required',
                    'high': 'Complex - High technical sophistication, specialized knowledge needed',
                    'note': 'Higher score = more complex (inverted from ranking for intuitive interpretation)'
                },
                'change_management_difficulty': {
                    'description': 'Change Management Difficulty',
                    'scale': '0-100',
                    'low': 'Easy change - Minimal organizational resistance, natural fit',
                    'medium': 'Moderate change - Some adaptation required',
                    'high': 'Difficult change - Significant organizational transformation, cultural shift needed',
                    'note': 'Higher score = more difficult (inverted from ranking for intuitive interpretation)'
                },
                'impact_potential': {
                    'description': 'Impact Potential',
                    'scale': '0-100',
                    'low': 'Low impact - Incremental improvements, localized benefits',
                    'medium': 'Moderate impact - Noticeable improvements across teams',
                    'high': 'High impact - Transformative results, significant competitive advantage',
                    'note': 'Higher score = greater potential impact on outcomes'
                },
                'time_to_value': {
                    'description': 'Time to Value',
                    'scale': '0-100',
                    'low': 'Slow - Months/years to realize benefits',
                    'medium': 'Moderate - Weeks/months to see results',
                    'high': 'Fast - Days/weeks to achieve value',
                    'note': 'Higher score = faster time to value'
                },
                'applicability': {
                    'description': 'Applicability (Context Dependency → Universality)',
                    'scale': '0-100',
                    'low': 'Narrow/Niche - Works only in specific contexts, industries, or situations',
                    'medium': 'Moderately general - Works in many but not all contexts',
                    'high': 'Universal - Applicable across almost all contexts, industries, team sizes',
                    'examples_low': 'Embedded systems testing, FDA regulatory compliance',
                    'examples_high': 'Clear communication, Problem-solving, Goal setting'
                },
                'implementation_difficulty': {
                    'description': 'Implementation Difficulty (Derived Metric)',
                    'scale': '0-100',
                    'calculation': 'Average of (100 - ease_adoption, resources_required, technical_complexity, change_management_difficulty)',
                    'note': 'Composite score combining ease, resources, complexity, and change difficulty. Higher score = more difficult to implement.'
                },
                'people_focus': {
                    'description': 'People Focus (Technical/System → Human)',
                    'scale': '0-100',
                    'low': 'Pure technical/system - Minimal human element, automation-focused, tool-driven',
                    'medium': 'Balanced - Mix of human and system elements, collaborative tools',
                    'high': 'Entirely about people - Relationships, behavior, culture, leadership',
                    'examples_low': 'Automated testing, CI/CD pipeline, Static code analysis',
                    'examples_high': 'Team building, Coaching, Psychological safety, Servant leadership',
                    'framework': 'People × Process × Purpose'
                },
                'process_focus': {
                    'description': 'Process Focus (Ad-hoc → Systematic)',
                    'scale': '0-100',
                    'low': 'Ad-hoc/informal - Flexible, no structure, situational, informal practices',
                    'medium': 'Semi-structured - Some procedures with flexibility, light framework',
                    'high': 'Rigid process - Fully procedural, standardized, measured, documented',
                    'examples_low': 'Open innovation, Exploratory testing, Ad-hoc collaboration',
                    'examples_high': 'CMMI, ISO 9001, Six Sigma, Formal verification',
                    'framework': 'People × Process × Purpose'
                },
                'purpose_orientation': {
                    'description': 'Purpose Orientation (Internal → External)',
                    'scale': '0-100',
                    'low': 'Pure internal - Cost cutting, productivity, efficiency, internal processes',
                    'medium': 'Balanced - Mix of internal efficiency and external value',
                    'high': 'Pure external - Customer value, innovation, market-facing outcomes',
                    'examples_low': 'Technical debt reduction, Code refactoring, Build optimization',
                    'examples_high': 'Customer discovery, User research, A/B testing, Value stream mapping',
                    'framework': 'People × Process × Purpose'
                }
            },
            'analysis_type': 'ranking_based_12d',
            'note': 'All scores on 0-100 scale. Implementation difficulty is average of 4 sub-dimensions. People × Process × Purpose framework adds 3 new dimensions.'
        },
        'summary': {
            'dimensions': {}
        },
        'methods': []
    }

    # Calculate summary statistics for each dimension
    dimensions = output['metadata']['dimensions']
    for dim in dimensions:
        values = [results[method.index][dim] for method in methods if method.index in results]

        output['summary']['dimensions'][dim] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values))
        }

    # Add method details
    for method in methods:
        if method.index not in results:
            continue

        result = results[method.index]

        method_data = {
            'index': method.index,
            'name': method.name,
            'source': method.source,
            'description': method.description[:200] + '...' if len(method.description) > 200 else method.description,

            # All 12 dimensions + derived metric (0-100 scale)
            'scope': result['scope'],
            'temporality': result['temporality'],
            'ease_adoption': result['ease_adoption'],
            'resources_required': result['resources_required'],
            'technical_complexity': result['technical_complexity'],
            'change_management_difficulty': result['change_management_difficulty'],
            'impact_potential': result['impact_potential'],
            'time_to_value': result['time_to_value'],
            'applicability': result['applicability'],
            'implementation_difficulty': result['implementation_difficulty'],
            'people_focus': result['people_focus'],
            'process_focus': result['process_focus'],
            'purpose_orientation': result['purpose_orientation'],

            # For backward compatibility with 2D visualization
            'scope_score': result['scope_score'],
            'temporality_score': result['temporality_score'],

            'reasoning': result['reasoning']
        }

        output['methods'].append(method_data)

    # Save to file
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    return output


async def main():
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Comprehensive 12-dimensional method analysis with People × Process × Purpose'
    )
    parser.add_argument(
        '--input',
        default='input/methods.csv',
        help='Input CSV file with methods (default: input/methods.csv)'
    )
    args = parser.parse_args()

    print("="*80)
    print("COMPREHENSIVE 12-DIMENSIONAL METHOD ANALYSIS")
    print("="*80)
    print("\nThis uses relative ranking (not direct scoring) on 12 dimensions")
    print("to avoid LLM clustering and ensure even distribution.")
    print("Includes People × Process × Purpose framework.\n")

    # Load configuration
    config_path = Path("config.yaml")
    if not config_path.exists():
        print("❌ Error: config.yaml not found")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load methods
    input_file = args.input
    if not Path(input_file).exists():
        print(f"❌ Error: {input_file} not found")
        return

    print(f"Loading methods from {input_file}...")
    methods = load_methods(input_file)
    print(f"✓ Loaded {len(methods)} methods")

    # Check if this is deduplicated dataset
    if 'deduplicated' in input_file:
        print("✓ Using deduplicated dataset (faster analysis)")
        metadata_path = Path("results/filter_metadata.json")
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                meta = json.load(f)
            print(f"  Original: {meta['original_count']} methods")
            print(f"  Removed:  {meta['removed_count']} duplicates")
            print(f"  Analyzing: {len(methods)} unique methods")
    print()

    # Initialize ranking analyzer
    print("Initializing 12D ranking analyzer...")
    ranking_analyzer = RankingLLMAnalyzer(config)
    print(f"✓ Using model: {config['llm']['model']}")
    print(f"✓ Chunk size: {ranking_analyzer.ranking_config.chunk_size}")
    print(f"✓ Overlap size: {ranking_analyzer.ranking_config.overlap_size}")
    print(f"✓ Parallel chunks: {ranking_analyzer.ranking_config.parallel_chunks}")

    # Check for multi-pass validation
    ranking_rounds = ranking_analyzer.ranking_config.ranking_rounds if hasattr(ranking_analyzer.ranking_config, 'ranking_rounds') else 1
    print(f"✓ Ranking passes: {ranking_rounds} (for validation)")
    print()

    # Estimate time (accounting for parallel processing)
    num_chunks = (len(methods) + ranking_analyzer.ranking_config.chunk_size - 1) // ranking_analyzer.ranking_config.chunk_size
    parallel_chunks = ranking_analyzer.ranking_config.parallel_chunks
    batches_per_pass = (num_chunks + parallel_chunks - 1) // parallel_chunks  # Sequential batches with parallelism
    time_per_batch = 15  # ~15 seconds per batch of parallel chunks
    time_per_dimension = batches_per_pass * time_per_batch * ranking_rounds  # Account for multiple passes
    total_time = time_per_dimension * 12 / 60  # 12 dimensions
    print(f"Estimated time: ~{total_time:.0f} minutes ({total_time/60:.1f} hours)")
    print(f"  ({num_chunks} chunks in {batches_per_pass} batches × 12 dimensions × {ranking_rounds} passes)")
    print(f"  (Parallel: {parallel_chunks} chunks at a time)")
    print()

    input("Press Enter to start analysis (or Ctrl+C to cancel)...")

    # Perform comprehensive 12D ranking analysis
    start_time = time.time()
    print("\nStarting comprehensive 12D ranking analysis...")
    print("="*80)

    results = await ranking_analyzer.batch_analyze_12d_ranked(methods)

    elapsed = time.time() - start_time
    print(f"\n✅ Analysis completed in {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)")

    # Save results
    print("\nSaving results...")

    # Choose output filename based on input
    if 'deduplicated' in args.input:
        output_path = Path("results/method_scores_12d_deduplicated.json")
    else:
        output_path = Path("results/method_scores_12d.json")

    output_data = save_12d_results(results, methods, str(output_path))

    print(f"✓ Results saved to {output_path}")

    # Print summary
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)

    print(f"\nTotal methods analyzed: {len(results)}")
    print("\nDimension statistics (0-100 scale):")
    print(f"{'Dimension':<35} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-"*80)

    for dim, stats in output_data['summary']['dimensions'].items():
        print(f"{dim.replace('_', ' ').title():<35} "
              f"{stats['mean']:>8.1f} "
              f"{stats['std']:>8.1f} "
              f"{stats['min']:>8.1f} "
              f"{stats['max']:>8.1f}")

    # Show sample methods
    print("\n" + "="*80)
    print("SAMPLE METHODS (Top 5 by Impact Potential)")
    print("="*80)

    sorted_methods = sorted(output_data['methods'], key=lambda x: x['impact_potential'], reverse=True)
    for i, method in enumerate(sorted_methods[:5], 1):
        print(f"\n{i}. {method['name']}")
        print(f"   Source: {method['source']}")
        print(f"   Scope: {method['scope']:.0f}  Temporality: {method['temporality']:.0f}  Impact: {method['impact_potential']:.0f}")
        print(f"   Implementation Difficulty: {method['implementation_difficulty']:.0f}")
        print(f"   Time to Value: {method['time_to_value']:.0f}  Applicability: {method['applicability']:.0f}")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Assign intelligent categories:")
    print("   python assign_intelligent_categories.py --input results/method_scores_12d_deduplicated.json")
    print("\n2. Visualize results:")
    print("   python plot_9d_interactive.py --input results/method_scores_12d_deduplicated_categorized.json")
    print("   python create_9d_visualizations.py  # Comprehensive dashboard")
    print("\n3. Analyze People × Process × Purpose:")
    print("   python analyze_3p_framework.py  # Specialized 3P analysis")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        sys.exit(1)
