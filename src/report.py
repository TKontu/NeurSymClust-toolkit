"""
Report generation with JSON/HTML outputs and visualizations.
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import networkx as nx
import numpy as np

from src.data import Method

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates analysis reports in multiple formats."""

    def __init__(self, output_dir: str = "./results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_duplicate_report(self, duplicate_results: List[Dict], methods: List[Method]) -> Path:
        """
        Generate duplicate analysis report.
        Returns path to output file.
        """
        logger.info("Generating duplicate analysis report...")

        # Find confirmed duplicates
        duplicates = [r for r in duplicate_results if r['is_duplicate'] and r['confidence'] >= 0.7]

        # Group duplicates (transitive closure)
        duplicate_groups = self._group_duplicates(duplicates)

        report = {
            "summary": {
                "total_pairs_analyzed": len(duplicate_results),
                "duplicates_found": len(duplicates),
                "duplicate_groups": len(duplicate_groups),
                "reduction_potential": len([m for group in duplicate_groups.values() for m in group]) - len(duplicate_groups)
            },
            "duplicate_groups": [
                {
                    "group_id": gid,
                    "methods": list(group),
                    "count": len(group)
                }
                for gid, group in duplicate_groups.items()
            ],
            "all_duplicate_pairs": duplicates
        }

        output_path = self.output_dir / "duplicates.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Duplicate report saved to {output_path}")
        return output_path

    def generate_compatibility_report(self, compatibility_results: List[Dict]) -> Path:
        """Generate compatibility analysis report."""
        logger.info("Generating compatibility analysis report...")

        # Build compatibility matrix data
        high_compat = [r for r in compatibility_results if r['compatibility_score'] >= 0.7]
        low_compat = [r for r in compatibility_results if r['compatibility_score'] < 0.4]

        report = {
            "summary": {
                "total_pairs_analyzed": len(compatibility_results),
                "high_compatibility_pairs": len(high_compat),
                "low_compatibility_pairs": len(low_compat),
                "average_compatibility": sum(r['compatibility_score'] for r in compatibility_results) / len(compatibility_results) if compatibility_results else 0
            },
            "high_compatibility_pairs": high_compat[:100],  # Top 100
            "low_compatibility_pairs": low_compat[:50],  # Worst 50
            "all_results": compatibility_results
        }

        output_path = self.output_dir / "compatibility.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Compatibility report saved to {output_path}")
        return output_path

    def generate_abstraction_report(self, abstraction_results: Dict[int, Dict], methods: List[Method]) -> Path:
        """Generate abstraction level analysis report."""
        logger.info("Generating abstraction analysis report...")

        # Count by level
        by_level = {"high": [], "medium": [], "low": []}
        for method in methods:
            if method.index in abstraction_results:
                level = abstraction_results[method.index]['level']
                by_level[level].append({
                    "index": method.index,
                    "name": method.name,
                    "source": method.source,
                    "reasoning": abstraction_results[method.index]['reasoning']
                })

        report = {
            "summary": {
                "total_methods": len(methods),
                "high_level_count": len(by_level['high']),
                "medium_level_count": len(by_level['medium']),
                "low_level_count": len(by_level['low'])
            },
            "by_level": by_level
        }

        output_path = self.output_dir / "abstraction.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Abstraction report saved to {output_path}")
        return output_path

    def generate_category_report(self, category_results: Dict[int, Dict], methods: List[Method]) -> Path:
        """Generate method categorization report."""
        logger.info("Generating categorization report...")

        # Group by category
        by_category = defaultdict(list)
        for method in methods:
            if method.index in category_results:
                cat_id = category_results[method.index]['category_id']
                cat_name = category_results[method.index]['category_name']
                by_category[cat_id].append({
                    "index": method.index,
                    "name": method.name,
                    "source": method.source,
                    "category_name": cat_name,
                    "reasoning": category_results[method.index]['reasoning']
                })

        # Calculate summary
        summary = {
            "total_methods": len(methods),
            "categories_count": len(by_category),
            "distribution": {cat_id: len(methods) for cat_id, methods in by_category.items()}
        }

        report = {
            "summary": summary,
            "by_category": dict(by_category)
        }

        output_path = self.output_dir / "categories.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Category report saved to {output_path}")
        return output_path

    def generate_scope_temporality_report(
        self,
        scope_temp_results: Dict[int, Dict],
        methods: List[Method]
    ) -> Path:
        """Generate scope/temporality positioning report."""
        logger.info("Generating scope/temporality report...")

        # Build positioning data
        positions = []
        for method in methods:
            if method.index in scope_temp_results:
                result = scope_temp_results[method.index]
                positions.append({
                    "index": method.index,
                    "name": method.name,
                    "source": method.source,
                    "scope_score": result['scope_score'],  # 0.0-1.0 for plotting
                    "temporality_score": result['temporality_score'],  # 0.0-1.0 for plotting
                    "scope_score_100": int(result['scope_score'] * 100),  # 0-100 for readability
                    "temporality_score_100": int(result['temporality_score'] * 100),  # 0-100 for readability
                    "reasoning": result['reasoning']
                })

        # Calculate statistics
        if positions:
            scope_scores = [p['scope_score'] for p in positions]
            temp_scores = [p['temporality_score'] for p in positions]

            summary = {
                "total_methods": len(positions),
                "avg_scope": float(np.mean(scope_scores)),
                "avg_temporality": float(np.mean(temp_scores)),
                "scope_std": float(np.std(scope_scores)),
                "temporality_std": float(np.std(temp_scores)),
                "scope_min": float(np.min(scope_scores)),
                "scope_max": float(np.max(scope_scores)),
                "temporality_min": float(np.min(temp_scores)),
                "temporality_max": float(np.max(temp_scores))
            }
        else:
            summary = {
                "total_methods": 0,
                "avg_scope": 0.0,
                "avg_temporality": 0.0,
                "scope_std": 0.0,
                "temporality_std": 0.0,
                "scope_min": 0.0,
                "scope_max": 0.0,
                "temporality_min": 0.0,
                "temporality_max": 0.0
            }

        report = {
            "summary": summary,
            "positions": positions
        }

        output_path = self.output_dir / "scope_temporality.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Scope/temporality report saved to {output_path}")
        return output_path

    def _build_duplicate_map(self, duplicate_results: List[Dict], threshold: float = 0.7) -> Dict[int, Set[int]]:
        """
        Build a mapping of method index to set of its duplicates with transitive closure.

        If A is a duplicate of B, and B is a duplicate of C, then A, B, and C
        are all considered duplicates of each other (connected components).

        Args:
            duplicate_results: Duplicate analysis results
            threshold: Confidence threshold for considering methods as duplicates

        Returns:
            Dict mapping method index to set of duplicate method indices (including transitive)
        """
        # First build direct duplicate relationships
        duplicate_map = {}

        for result in duplicate_results:
            if result.get('is_duplicate', False) and result.get('confidence', 0) >= threshold:
                idx1 = result['method1_index']
                idx2 = result['method2_index']

                # Add bidirectional duplicates
                if idx1 not in duplicate_map:
                    duplicate_map[idx1] = set()
                if idx2 not in duplicate_map:
                    duplicate_map[idx2] = set()

                duplicate_map[idx1].add(idx2)
                duplicate_map[idx2].add(idx1)

        # Compute transitive closure using iterative expansion
        # Keep expanding until no new duplicates are found
        changed = True
        iterations = 0
        while changed and iterations < 10:  # Safety limit
            changed = False
            iterations += 1

            for method_idx in list(duplicate_map.keys()):
                original_size = len(duplicate_map[method_idx])

                # For each duplicate of this method, add all of ITS duplicates
                for dup_idx in list(duplicate_map[method_idx]):
                    if dup_idx in duplicate_map:
                        duplicate_map[method_idx].update(duplicate_map[dup_idx])

                # Remove self-reference if it got added
                duplicate_map[method_idx].discard(method_idx)

                if len(duplicate_map[method_idx]) > original_size:
                    changed = True

        logger.info(f"Built duplicate map with transitive closure: {len(duplicate_map)} methods have duplicates (converged in {iterations} iterations)")
        return duplicate_map

    def generate_toolkit_recommendations(
        self,
        methods: List[Method],
        compatibility_results: List[Dict],
        abstraction_results: Dict[int, Dict],
        duplicate_results: List[Dict],
        category_results: Dict[int, Dict] = None,
        toolkit_constraints: Dict = None
    ) -> Path:
        """
        Generate recommended method toolkits based on compatibility analysis.
        """
        logger.info("Generating toolkit recommendations...")

        # Set default constraints if not provided
        if toolkit_constraints is None:
            toolkit_constraints = {
                "min_size": 5,
                "max_size": 10,
                "require_category_diversity": True,
                "max_methods_per_category": 3,
                "min_compatibility_for_same_category": 0.7
            }

        # Build compatibility graph
        G = self._build_compatibility_graph(methods, compatibility_results)

        # Build duplicate map for filtering
        duplicate_map = self._build_duplicate_map(duplicate_results, threshold=0.7)

        # Find diverse toolkits with constraints
        toolkits = self._find_constrained_toolkits(
            G, methods, abstraction_results, category_results, toolkit_constraints, duplicate_map
        )

        report = {
            "summary": {
                "total_toolkits": len(toolkits),
                "description": "Recommended method combinations based on compatibility and abstraction diversity"
            },
            "toolkits": toolkits
        }

        output_path = self.output_dir / "toolkits.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Toolkit recommendations saved to {output_path}")
        return output_path

    def generate_html_summary(
        self,
        methods: List[Method],
        duplicate_report: Dict,
        compatibility_report: Dict,
        abstraction_report: Dict,
        toolkit_report: Dict
    ) -> Path:
        """Generate HTML summary report with visualizations."""
        logger.info("Generating HTML summary report...")

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Methods Analysis Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .section {{
            background-color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px 10px 0;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }}
        .metric-label {{
            font-size: 0.9em;
            color: #7f8c8d;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #34495e;
            color: white;
        }}
        .high {{ color: #27ae60; font-weight: bold; }}
        .medium {{ color: #f39c12; }}
        .low {{ color: #e74c3c; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Product Development Methods Analysis</h1>
        <p>Comprehensive analysis of {len(methods)} methods</p>
    </div>

    <div class="section">
        <h2>Overview</h2>
        <div class="metric">
            <div class="metric-value">{len(methods)}</div>
            <div class="metric-label">Total Methods</div>
        </div>
        <div class="metric">
            <div class="metric-value">{duplicate_report['summary']['duplicate_groups']}</div>
            <div class="metric-label">Duplicate Groups</div>
        </div>
        <div class="metric">
            <div class="metric-value">{toolkit_report['summary']['total_toolkits']}</div>
            <div class="metric-label">Recommended Toolkits</div>
        </div>
        <div class="metric">
            <div class="metric-value">{compatibility_report['summary']['high_compatibility_pairs']}</div>
            <div class="metric-label">High Compatibility Pairs</div>
        </div>
    </div>

    <div class="section">
        <h2>Duplicate Analysis</h2>
        <p><strong>{duplicate_report['summary']['duplicates_found']}</strong> duplicate pairs found across
           <strong>{duplicate_report['summary']['duplicate_groups']}</strong> groups.</p>
        <p>Reduction potential: <strong>{duplicate_report['summary']['reduction_potential']}</strong> methods
           could be consolidated.</p>

        <h3>Top Duplicate Groups</h3>
        <table>
            <tr>
                <th>Group</th>
                <th>Count</th>
                <th>Methods</th>
            </tr>
            {self._generate_duplicate_table_rows(duplicate_report['duplicate_groups'][:10])}
        </table>
    </div>

    <div class="section">
        <h2>Abstraction Levels</h2>
        <div class="metric">
            <div class="metric-value high">{abstraction_report['summary']['high_level_count']}</div>
            <div class="metric-label">High (Principles)</div>
        </div>
        <div class="metric">
            <div class="metric-value medium">{abstraction_report['summary']['medium_level_count']}</div>
            <div class="metric-label">Medium (Frameworks)</div>
        </div>
        <div class="metric">
            <div class="metric-value low">{abstraction_report['summary']['low_level_count']}</div>
            <div class="metric-label">Low (Techniques)</div>
        </div>
    </div>

    <div class="section">
        <h2>Toolkit Recommendations</h2>
        <p>{toolkit_report['summary']['description']}</p>
        <p>Generated <strong>{toolkit_report['summary']['total_toolkits']}</strong> toolkit recommendations.</p>

        <h3>Top Toolkits</h3>
        {self._generate_toolkit_html(toolkit_report['toolkits'][:5])}
    </div>

    <div class="section">
        <h2>Compatibility Insights</h2>
        <p>Average compatibility score: <strong>{compatibility_report['summary']['average_compatibility']:.2f}</strong></p>
        <p>High compatibility pairs: <strong>{compatibility_report['summary']['high_compatibility_pairs']}</strong></p>
        <p>Low compatibility pairs: <strong>{compatibility_report['summary']['low_compatibility_pairs']}</strong></p>
    </div>
</body>
</html>
        """

        output_path = self.output_dir / "report.html"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"HTML report saved to {output_path}")
        return output_path

    def _group_duplicates(self, duplicates: List[Dict]) -> Dict[int, Set[str]]:
        """Group duplicates using transitive closure."""
        # Build graph
        G = nx.Graph()
        for dup in duplicates:
            G.add_edge(dup['method1_name'], dup['method2_name'])

        # Find connected components
        groups = {}
        for i, component in enumerate(nx.connected_components(G)):
            groups[i] = component

        return groups

    def _build_compatibility_graph(self, methods: List[Method], compatibility_results: List[Dict]) -> nx.Graph:
        """Build graph where edges represent compatible methods."""
        G = nx.Graph()

        # Add all methods as nodes
        for method in methods:
            G.add_node(method.index, name=method.name)

        # Add edges for compatible pairs
        for result in compatibility_results:
            if result['compatibility_score'] >= 0.7:  # High compatibility threshold
                G.add_edge(
                    result['method1_index'],
                    result['method2_index'],
                    weight=result['compatibility_score']
                )

        logger.info(f"Built compatibility graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    def _find_constrained_toolkits(
        self,
        G: nx.Graph,
        methods: List[Method],
        abstraction_results: Dict,
        category_results: Dict,
        constraints: Dict,
        duplicate_map: Dict[int, Set[int]]
    ) -> List[Dict]:
        """Find toolkit recommendations with size and category constraints."""
        toolkits = []

        # Check if graph has edges
        if G.number_of_edges() == 0:
            logger.warning("Compatibility graph has no edges - cannot generate toolkits.")
            return []

        min_size = constraints.get('min_size', 5)
        max_size = constraints.get('max_size', 10)
        max_per_category = constraints.get('max_methods_per_category', 3)
        min_compat_same_cat = constraints.get('min_compatibility_for_same_category', 0.7)

        # Try to find diverse toolkits using Monte Carlo sampling
        try:
            import random

            # Get nodes with good connectivity (top 25%)
            node_degrees = dict(G.degree())
            degree_threshold = sorted(node_degrees.values(), reverse=True)[len(node_degrees)//4] if node_degrees else 0
            candidate_seeds = [n for n, d in node_degrees.items() if d >= degree_threshold]

            # Group by category for diversity
            seeds_by_category = defaultdict(list)
            for node in candidate_seeds:
                if node in category_results:
                    cat_id = category_results[node]['category_id']
                    seeds_by_category[cat_id].append(node)

            # Select diverse seeds (one from each category, then random from top candidates)
            diverse_seeds = []
            for cat_id, nodes in seeds_by_category.items():
                if nodes:
                    diverse_seeds.append(random.choice(nodes))

            # Add more random candidates
            while len(diverse_seeds) < 50:
                if candidate_seeds:
                    seed = random.choice(candidate_seeds)
                    if seed not in diverse_seeds:
                        diverse_seeds.append(seed)
                else:
                    break

            candidate_toolkits = []

            # Generate candidate toolkits from diverse seeds
            logger.info(f"Testing {min(len(diverse_seeds), 50)} diverse seed points...")
            for seed_node in diverse_seeds[:50]:  # Try up to 50 different starting points
                toolkit = self._build_single_toolkit(
                    G, methods, seed_node, category_results,
                    min_size, max_size, max_per_category, set(), min_compat_same_cat,
                    duplicate_map, abstraction_results
                )

                if toolkit and len(toolkit) >= min_size:
                    # Calculate metrics
                    toolkit_methods = []
                    abstraction_mix = {"high": 0, "medium": 0, "low": 0}
                    category_counts = defaultdict(int)
                    category_abstraction_dist = defaultdict(lambda: {'high': 0, 'medium': 0, 'low': 0})

                    for method_idx in toolkit:
                        method = next((m for m in methods if m.index == method_idx), None)
                        if method:
                            toolkit_methods.append({
                                "index": method.index,
                                "name": method.name,
                                "source": method.source,
                                "category": category_results[method.index]['category_name'] if category_results and method.index in category_results else "Unknown"
                            })

                            if method.index in abstraction_results:
                                level = abstraction_results[method.index]['level']
                                abstraction_mix[level] += 1
                                if method.index in category_results:
                                    cat_id = category_results[method.index]['category_id']
                                    category_abstraction_dist[cat_id][level] += 1

                            if category_results and method.index in category_results:
                                cat_id = category_results[method.index]['category_id']
                                category_counts[cat_id] += 1

                    # Enhanced scoring: reward abstraction complementarity within categories
                    diversity_score = len([v for v in abstraction_mix.values() if v > 0]) / 3.0
                    category_diversity = len(category_counts) / 13  # 13 categories now
                    size_score = len(toolkit_methods) / max_size

                    # Bonus for complementary abstraction levels within categories
                    complementarity_bonus = 0
                    for cat_id, abs_dist in category_abstraction_dist.items():
                        levels_present = sum(1 for v in abs_dist.values() if v > 0)
                        if levels_present > 1:  # Multiple abstraction levels in same category
                            complementarity_bonus += 0.1

                    toolkit_score = (
                        diversity_score * 0.25 +
                        category_diversity * 0.45 +
                        size_score * 0.15 +
                        complementarity_bonus * 0.15
                    )

                    candidate_toolkits.append({
                        "methods": toolkit_methods,
                        "size": len(toolkit_methods),
                        "abstraction_mix": abstraction_mix,
                        "category_distribution": dict(category_counts),
                        "score": toolkit_score,
                        "method_indices": set(toolkit)
                    })

            logger.info(f"Generated {len(candidate_toolkits)} candidate toolkits, selecting top 5...")

            # Sort by score and select top 5 non-overlapping toolkits
            candidate_toolkits.sort(key=lambda x: x['score'], reverse=True)

            toolkits = []
            used_methods = set()
            for candidate in candidate_toolkits:
                # Check overlap with already selected toolkits
                overlap = len(candidate['method_indices'] & used_methods)
                if overlap < len(candidate['method_indices']) * 0.3:  # Allow max 30% overlap
                    candidate['toolkit_id'] = len(toolkits)
                    del candidate['method_indices']  # Remove temp field
                    toolkits.append(candidate)
                    used_methods.update([m['index'] for m in candidate['methods']])

                    if len(toolkits) >= 5:
                        break

        except Exception as e:
            logger.warning(f"Failed to find toolkits: {e}")
            import traceback
            logger.warning(traceback.format_exc())

        logger.info(f"Generated {len(toolkits)} diverse toolkits")
        return toolkits

    def _check_abstraction_constraints(
        self,
        category_id: str,
        new_level: str,
        category_abstraction_counts: Dict
    ) -> bool:
        """
        Check if adding a method with new_level to category respects hierarchical constraints.

        Rules per category:
        - Max 1 high (abstract)
        - If high exists: max 1 medium OR max 2 low (not both)
        - If only medium: max 2 low
        - If only low: max 3 low
        """
        counts = category_abstraction_counts.get(category_id, {'high': 0, 'medium': 0, 'low': 0})

        if new_level == 'high':
            # Only 1 abstract principle per category
            return counts['high'] == 0

        elif new_level == 'medium':
            # If there's an abstract principle, only 1 framework allowed
            if counts['high'] > 0:
                return counts['medium'] == 0
            # Otherwise, max 2 frameworks
            return counts['medium'] < 2

        elif new_level == 'low':
            # If there's an abstract principle, no frameworks, max 2 concrete
            if counts['high'] > 0 and counts['medium'] == 0:
                return counts['low'] < 2
            # If there's an abstract + framework, no concrete tools
            if counts['high'] > 0 and counts['medium'] > 0:
                return False
            # If only framework(s), max 2 concrete
            if counts['high'] == 0 and counts['medium'] > 0:
                return counts['low'] < 2
            # If no abstract or framework, max 3 concrete
            return counts['low'] < 3

        return False

    def _build_single_toolkit(
        self,
        G: nx.Graph,
        methods: List[Method],
        seed_node: int,
        category_results: Dict,
        min_size: int,
        max_size: int,
        max_per_category: int,
        exclude_methods: set,
        min_compatibility: float = 0.7,
        duplicate_map: Dict[int, Set[int]] = None,
        abstraction_results: Dict = None
    ) -> list:
        """
        Build a single toolkit with abstraction-aware hierarchical constraints.

        Prioritizes diversity and complementary abstraction levels over raw compatibility.
        """
        toolkit = [seed_node]
        category_counts = defaultdict(int)
        category_abstraction_counts = defaultdict(lambda: {'high': 0, 'medium': 0, 'low': 0})

        # Initialize with seed
        if category_results and seed_node in category_results:
            cat_id = category_results[seed_node]['category_id']
            category_counts[cat_id] += 1
            if abstraction_results and seed_node in abstraction_results:
                level = abstraction_results[seed_node]['level']
                category_abstraction_counts[cat_id][level] += 1

        # Get all compatible neighbors
        all_neighbors = []
        for neighbor in G.neighbors(seed_node):
            if neighbor not in exclude_methods and neighbor != seed_node:
                weight = G[seed_node][neighbor].get('weight', 0.5)
                all_neighbors.append((neighbor, weight))

        # Phase 1: Prioritize category diversity (one method per category first)
        covered_categories = set(category_counts.keys())
        for neighbor, weight in sorted(all_neighbors, key=lambda x: x[1], reverse=True):
            if len(toolkit) >= max_size:
                break

            if neighbor not in category_results:
                continue

            cat_id = category_results[neighbor]['category_id']

            # Skip if category already covered in phase 1
            if cat_id in covered_categories and len(covered_categories) < 10:
                continue

            # Check duplicate
            if duplicate_map and neighbor in duplicate_map:
                if any(existing in duplicate_map[neighbor] for existing in toolkit):
                    continue

            # Check abstraction constraints
            if abstraction_results and neighbor in abstraction_results:
                level = abstraction_results[neighbor]['level']
                if not self._check_abstraction_constraints(cat_id, level, category_abstraction_counts):
                    continue

            # Check basic compatibility (relaxed to 0.5)
            if weight < 0.5:
                continue

            # Add method
            toolkit.append(neighbor)
            category_counts[cat_id] += 1
            covered_categories.add(cat_id)
            if abstraction_results and neighbor in abstraction_results:
                level = abstraction_results[neighbor]['level']
                category_abstraction_counts[cat_id][level] += 1

        # Phase 2: Add complementary methods to existing categories
        for neighbor, weight in sorted(all_neighbors, key=lambda x: x[1], reverse=True):
            if len(toolkit) >= max_size:
                break

            if neighbor in toolkit:
                continue

            if neighbor not in category_results:
                continue

            cat_id = category_results[neighbor]['category_id']

            # Check duplicate
            if duplicate_map and neighbor in duplicate_map:
                if any(existing in duplicate_map[neighbor] for existing in toolkit):
                    continue

            # Check abstraction constraints (hierarchical)
            if abstraction_results and neighbor in abstraction_results:
                level = abstraction_results[neighbor]['level']
                if not self._check_abstraction_constraints(cat_id, level, category_abstraction_counts):
                    continue

            # Check compatibility with same-category methods
            same_cat_methods = [m for m in toolkit
                              if category_results.get(m, {}).get('category_id') == cat_id]

            if same_cat_methods:
                # Require compatibility with at least half of same-category methods
                compat_count = sum(1 for m in same_cat_methods
                                 if G.has_edge(neighbor, m) and G[neighbor][m].get('weight', 0) >= 0.6)
                if compat_count < len(same_cat_methods) / 2:
                    continue

            # Add method
            toolkit.append(neighbor)
            category_counts[cat_id] += 1
            if abstraction_results and neighbor in abstraction_results:
                level = abstraction_results[neighbor]['level']
                category_abstraction_counts[cat_id][level] += 1

        return toolkit if len(toolkit) >= min_size else []

    def _find_toolkits(self, G: nx.Graph, methods: List[Method], abstraction_results: Dict) -> List[Dict]:
        """Legacy method - kept for backward compatibility."""
        toolkits = []

        # Check if graph has edges (need edges for community detection)
        if G.number_of_edges() == 0:
            logger.warning("Compatibility graph has no edges - cannot generate toolkits. "
                         "Try lowering compatibility threshold or analyzing more pairs.")
            return []

        # Find communities using greedy modularity
        try:
            communities = nx.community.greedy_modularity_communities(G)

            for i, community in enumerate(communities):
                if len(community) >= 3:  # Minimum toolkit size
                    toolkit_methods = []
                    abstraction_mix = {"high": 0, "medium": 0, "low": 0}

                    for method_idx in community:
                        method = next((m for m in methods if m.index == method_idx), None)
                        if method:
                            toolkit_methods.append({
                                "index": method.index,
                                "name": method.name,
                                "source": method.source
                            })

                            if method.index in abstraction_results:
                                level = abstraction_results[method.index]['level']
                                abstraction_mix[level] += 1

                    # Score toolkit (prefer diverse abstraction levels)
                    diversity_score = len([v for v in abstraction_mix.values() if v > 0]) / 3.0
                    size_score = min(len(toolkit_methods) / 10.0, 1.0)
                    toolkit_score = (diversity_score + size_score) / 2.0

                    toolkits.append({
                        "toolkit_id": i,
                        "methods": toolkit_methods,
                        "size": len(toolkit_methods),
                        "abstraction_mix": abstraction_mix,
                        "score": toolkit_score
                    })

        except Exception as e:
            logger.warning(f"Failed to find communities: {e}")

        # Sort by score
        toolkits.sort(key=lambda x: x['score'], reverse=True)

        return toolkits

    def _generate_duplicate_table_rows(self, groups: List[Dict]) -> str:
        """Generate HTML table rows for duplicate groups."""
        if not groups:
            return "<tr><td colspan='3' style='text-align: center; font-style: italic;'>No duplicate groups found</td></tr>"

        rows = []
        for group in groups:
            methods_str = ", ".join(list(group['methods'])[:5])
            if len(group['methods']) > 5:
                methods_str += f" ... (+{len(group['methods']) - 5} more)"

            rows.append(f"""
            <tr>
                <td>Group {group['group_id']}</td>
                <td>{group['count']}</td>
                <td>{methods_str}</td>
            </tr>
            """)

        return "\n".join(rows)

    def _generate_toolkit_html(self, toolkits: List[Dict]) -> str:
        """Generate HTML for toolkit recommendations."""
        if not toolkits:
            return "<p style='font-style: italic;'>No toolkits could be generated. This may be due to insufficient compatible pairs.</p>"

        html_parts = []

        for toolkit in toolkits:
            methods_list = "<br>".join([f"- {m['name']}" for m in toolkit['methods'][:10]])
            if len(toolkit['methods']) > 10:
                methods_list += f"<br>... +{len(toolkit['methods']) - 10} more"

            html_parts.append(f"""
            <div style="margin-bottom: 20px; padding: 15px; background-color: #ecf0f1; border-radius: 5px;">
                <h4>Toolkit #{toolkit['toolkit_id']} (Score: {toolkit['score']:.2f})</h4>
                <p><strong>Size:</strong> {toolkit['size']} methods</p>
                <p><strong>Abstraction Mix:</strong> High: {toolkit['abstraction_mix']['high']},
                   Medium: {toolkit['abstraction_mix']['medium']},
                   Low: {toolkit['abstraction_mix']['low']}</p>
                <p><strong>Methods:</strong><br>{methods_list}</p>
            </div>
            """)

        return "\n".join(html_parts)
