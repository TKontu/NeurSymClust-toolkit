#!/usr/bin/env python3
"""
Comprehensive Compatibility Analysis for Methods
Analyzes method compatibility using structured LLM prompts and graph-based analysis.
"""
import asyncio
import json
import logging
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
import yaml
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Phase 1: Structured Compatibility Framework
# ============================================================================

COMPATIBILITY_DIMENSIONS = {
    'resource_conflict': {
        'description': 'Do these methods compete for same resources (time, people, budget)?',
        'scale': 'high_conflict (-1) to complementary (+1)',
        'examples': {
            'conflict': 'Both require dedicated scrum master full-time',
            'complementary': 'One needs morning time, other async work'
        }
    },

    'conceptual_overlap': {
        'description': 'Do they solve the same problem or address different needs?',
        'scale': 'redundant (-1) to distinct (+1)',
        'examples': {
            'redundant': 'Both are retrospective techniques',
            'distinct': 'One for planning, other for review'
        }
    },

    'philosophical_alignment': {
        'description': 'Do their underlying principles align or conflict?',
        'scale': 'contradictory (-1) to reinforcing (+1)',
        'examples': {
            'contradictory': 'Command-control vs self-organization',
            'reinforcing': 'Both emphasize continuous learning'
        }
    },

    'implementation_sequence': {
        'description': 'Can they be implemented simultaneously or need sequencing?',
        'scale': 'prerequisite (-1) to independent (0) to synergistic (+1)',
        'examples': {
            'prerequisite': 'Need A before B can work',
            'synergistic': 'Work better when combined'
        }
    },

    'cognitive_load': {
        'description': 'Combined mental burden on team',
        'scale': 'overwhelming (-1) to manageable (+1)',
        'examples': {
            'overwhelming': 'Too many new concepts at once',
            'manageable': 'Builds on familiar concepts'
        }
    }
}

OVERLAP_DETECTION_PROMPT = """
Analyze overlap between these methods:

Method A: {method_a_name}
{method_a_description}

Method B: {method_b_name}
{method_b_description}

YES/NO for each:
1. Same core problem?
2. Same role executes both?
3. Same timing/ceremony?
4. Same output/artifact?
5. Confusing to use both?

Classification:
- 4-5 YES = redundant
- 2-3 YES = conflicting
- 1 YES = partial
- 0 YES = none

JSON only:
{{
  "same_problem": true/false,
  "same_role": true/false,
  "same_timing": true/false,
  "same_output": true/false,
  "causes_confusion": true/false,
  "overlap_type": "none|partial|conflicting|redundant"
}}
"""

COMPATIBILITY_ANALYSIS_PROMPT = """
Rate compatibility (0.0-1.0):

Method A: {method_a}
{description_a}

Method B: {method_b}
{description_b}

Calibration scale:
0.95 = Daily Standup + Sprint Planning
0.85 = TDD + CI
0.60 = Scrum + Kanban
0.35 = User Stories + Use Cases
0.20 = Sprint Review + Sprint Demo (redundant)
0.10 = Waterfall + Agile (incompatible)

Consider:
- Resource conflict?
- Purpose overlap?
- Philosophy alignment?
- Can coexist?

Use full 0-1 scale. Redundant methods <0.3, incompatible <0.15.

JSON only:
{{
  "compatibility_score": [0.0-1.0],
  "relationship_type": "synergistic|compatible|neutral|problematic|incompatible",
  "recommendation": "combine|choose_one|sequence|avoid",
  "key_concern": "[brief issue or 'none']"
}}
"""


# ============================================================================
# LLM Interface
# ============================================================================

class CompatibilityAnalyzer:
    """Handles LLM-based compatibility analysis."""

    def __init__(self, config: dict):
        self.config = config['llm']
        self.base_url = self.config['base_url']
        self.api_key = self.config['api_key']
        self.model = self.config['model']
        self.temperature = self.config['temperature']
        self.timeout = self.config['timeout']
        self.max_concurrent = self.config.get('max_concurrent', 25)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60))
    async def _call_llm(self, prompt: str, max_tokens: int = 4000) -> str:
        """Call LLM API with retry logic."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": self.temperature,
                        "max_tokens": max_tokens
                    },
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"LLM API error: {response.status} - {error_text}")

                    data = await response.json()
                    content = data['choices'][0]['message']['content']

                    # Handle None responses
                    if content is None:
                        raise Exception("LLM returned None content")

                    return content.strip()

        except asyncio.TimeoutError:
            logger.error(f"LLM call timeout after {self.timeout}s")
            raise
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    def _extract_json(self, response: str) -> dict:
        """Extract JSON from LLM response, with fallbacks for truncated JSON."""
        # Remove markdown code blocks if present
        response = response.replace('```json', '').replace('```', '').strip()

        # Try to find JSON in response
        start = response.find('{')
        end = response.rfind('}') + 1

        if start >= 0 and end > start:
            json_str = response[start:end]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                # Try to fix common truncation issues
                logger.warning(f"Attempting to repair truncated JSON...")

                # If missing closing brace, try to complete it
                if json_str.count('{') > json_str.count('}'):
                    # Try adding missing closing braces
                    missing = json_str.count('{') - json_str.count('}')
                    json_str += '}' * missing
                    try:
                        return json.loads(json_str)
                    except:
                        pass

                # If still fails, try to extract what we can
                logger.error(f"JSON decode error: {e}\nPartial response: {response[:200]}...")
                raise ValueError(f"Cannot parse JSON: {response[:200]}...")

        raise ValueError(f"No JSON found in response: {response[:200]}...")

    async def analyze_overlap(self, method_a: Dict, method_b: Dict) -> Dict:
        """Analyze if two methods have problematic overlap."""
        # Truncate descriptions if too long
        desc_a = method_a['Description'][:500] if len(method_a['Description']) > 500 else method_a['Description']
        desc_b = method_b['Description'][:500] if len(method_b['Description']) > 500 else method_b['Description']

        prompt = OVERLAP_DETECTION_PROMPT.format(
            method_a_name=method_a['Method'],
            method_a_description=desc_a,
            method_b_name=method_b['Method'],
            method_b_description=desc_b
        )

        try:
            response = await self._call_llm(prompt, max_tokens=4000)
            result = self._extract_json(response)

            # Derive has_problematic_overlap from overlap_type (for backward compatibility)
            if 'has_problematic_overlap' not in result:
                overlap_type = result.get('overlap_type', 'none')
                result['has_problematic_overlap'] = overlap_type in ['redundant', 'conflicting']

            return result
        except Exception as e:
            logger.error(f"Failed to analyze overlap for {method_a['Method']} vs {method_b['Method']}: {str(e)[:100]}")
            return {
                "same_problem": False,
                "same_role": False,
                "same_timing": False,
                "same_output": False,
                "causes_confusion": False,
                "has_problematic_overlap": False,
                "overlap_type": "error"
            }

    async def analyze_compatibility(self, method_a: Dict, method_b: Dict) -> Dict:
        """Analyze compatibility between two methods."""
        # Truncate descriptions if too long
        desc_a = method_a['Description'][:500] if len(method_a['Description']) > 500 else method_a['Description']
        desc_b = method_b['Description'][:500] if len(method_b['Description']) > 500 else method_b['Description']

        prompt = COMPATIBILITY_ANALYSIS_PROMPT.format(
            method_a=method_a['Method'],
            description_a=desc_a,
            method_b=method_b['Method'],
            description_b=desc_b
        )

        try:
            response = await self._call_llm(prompt, max_tokens=4000)
            result = self._extract_json(response)

            # Normalize field names (support both old and new prompts)
            if 'key_concern' in result and 'specific_concern' not in result:
                result['specific_concern'] = result.pop('key_concern')

            # Add method names for reference
            result['method_a'] = method_a['Method']
            result['method_b'] = method_b['Method']

            return result
        except Exception as e:
            logger.error(f"Failed to analyze compatibility for {method_a['Method']} vs {method_b['Method']}: {str(e)[:100]}")
            return {
                "method_a": method_a['Method'],
                "method_b": method_b['Method'],
                "compatibility_score": 0.5,
                "relationship_type": "error",
                "recommendation": "unknown",
                "specific_concern": f"Analysis error"
            }


# ============================================================================
# Phase 2: Robust Comparison Strategy
# ============================================================================

def generate_all_remaining_pairs(methods_df: pd.DataFrame,
                                 max_pairs: int = 10000,
                                 exclude_pairs: set = None) -> List[Tuple[int, int]]:
    """
    Generate all remaining pairs exhaustively (not strategically sampled).

    This is used when you want to systematically cover all pairs,
    especially useful after strategic sampling has covered high-value pairs.

    Args:
        methods_df: DataFrame with methods
        max_pairs: Maximum number of pairs to return
        exclude_pairs: Set of (idx1, idx2) tuples already analyzed

    Returns:
        List of (idx1, idx2) tuples for the next batch of pairs
    """
    if exclude_pairs is None:
        exclude_pairs = set()

    n_methods = len(methods_df)
    total_possible = n_methods * (n_methods - 1) // 2
    remaining_count = total_possible - len(exclude_pairs)

    logger.info("Generating remaining pairs exhaustively...")
    logger.info(f"  Total possible pairs: {total_possible:,}")
    logger.info(f"  Already analyzed: {len(exclude_pairs):,}")
    logger.info(f"  Remaining: {remaining_count:,}")
    logger.info(f"  Requesting: {max_pairs:,}")

    pairs = []

    # Generate pairs in order until we hit max_pairs or run out
    for i in range(n_methods):
        if len(pairs) >= max_pairs:
            break
        for j in range(i + 1, n_methods):
            pair = (i, j)
            if pair not in exclude_pairs:
                pairs.append(pair)
                if len(pairs) >= max_pairs:
                    break

    logger.info(f"  Generated: {len(pairs):,} new pairs")
    return pairs


def sample_method_pairs_strategic(methods_df: pd.DataFrame,
                                  scores_df: pd.DataFrame = None,
                                  max_pairs: int = 2000,
                                  exclude_pairs: set = None) -> List[Tuple[int, int]]:
    """
    Sample method pairs strategically for compatibility analysis.

    Strategies:
    1. High-impact methods (top ranked)
    2. Same-source pairs (likely similar)
    3. Cross-source pairs (likely different)
    4. Random baseline pairs

    Args:
        exclude_pairs: Set of (idx1, idx2) tuples already analyzed (will be skipped)
    """
    if exclude_pairs is None:
        exclude_pairs = set()

    pairs = []
    methods_list = methods_df.to_dict('records')
    n_methods = len(methods_list)

    logger.info("Sampling method pairs strategically...")
    logger.info(f"  Total possible pairs: {n_methods * (n_methods - 1) // 2}")
    logger.info(f"  Already analyzed: {len(exclude_pairs)}")
    logger.info(f"  Target new pairs: {max_pairs} ({max_pairs / (n_methods * (n_methods - 1) // 2) * 100:.2f}% coverage)")

    # Strategy 1: High-impact method pairs (if we have scores)
    if scores_df is not None and 'impact_potential' in scores_df.columns:
        logger.info("  Strategy 1: High-impact method pairs (all combinations)")
        # Get indices of top methods by impact - scale with max_pairs
        top_n = min(50, int(np.sqrt(max_pairs * 2)))  # Dynamic: more pairs = more top methods
        top_indices = scores_df.nlargest(top_n, 'impact_potential').index.tolist()
        high_impact_pairs = [(top_indices[i], top_indices[j])
                            for i in range(len(top_indices))
                            for j in range(i+1, len(top_indices))
                            if (min(top_indices[i], top_indices[j]), max(top_indices[i], top_indices[j])) not in exclude_pairs]
        logger.info(f"    Top {top_n} methods = {len(high_impact_pairs)} new pairs")
        pairs.extend(high_impact_pairs)

    # Strategy 2: Same-source pairs (sample rate scales with max_pairs)
    logger.info("  Strategy 2: Same-source pairs (likely overlaps)")
    source_groups = methods_df.groupby('Source').groups

    # Calculate sample rate based on max_pairs budget
    remaining_budget = max_pairs - len(pairs)
    total_same_source_pairs = sum(len(indices) * (len(indices) - 1) // 2
                                  for indices in source_groups.values())

    same_source_rate = min(0.3, remaining_budget * 0.3 / max(1, total_same_source_pairs))
    logger.info(f"    Sample rate: {same_source_rate * 100:.1f}%")

    for source, indices in source_groups.items():
        indices = list(indices)
        if len(indices) >= 2:
            n_pairs = max(1, int(len(indices) * (len(indices) - 1) / 2 * same_source_rate))
            for _ in range(n_pairs):
                if len(pairs) >= max_pairs * 0.5:  # Cap at 50% of budget
                    break
                idx1, idx2 = np.random.choice(indices, 2, replace=False)
                pair = (min(idx1, idx2), max(idx1, idx2))
                if pair not in pairs and pair not in exclude_pairs:
                    pairs.append(pair)

    # Strategy 3: Cross-source pairs (discover complementary approaches)
    logger.info("  Strategy 3: Cross-source pairs (complementary)")
    sources = list(source_groups.keys())
    remaining_budget = max_pairs - len(pairs)

    # Sample more cross-source pairs with larger budgets
    pairs_per_source_combo = max(3, remaining_budget // (len(sources) * (len(sources) - 1) // 2) // 2)
    logger.info(f"    ~{pairs_per_source_combo} pairs per source combination")

    for i, source1 in enumerate(sources):
        for source2 in sources[i+1:]:
            if len(pairs) >= max_pairs * 0.8:  # Cap at 80% of budget
                break
            indices1 = list(source_groups[source1])
            indices2 = list(source_groups[source2])

            n_sample = min(pairs_per_source_combo, len(indices1), len(indices2))
            for _ in range(n_sample):
                idx1 = np.random.choice(indices1)
                idx2 = np.random.choice(indices2)
                pair = (min(idx1, idx2), max(idx1, idx2))
                if pair not in pairs and pair not in exclude_pairs:
                    pairs.append(pair)

    # Strategy 4: Random baseline pairs (unbiased sample)
    logger.info("  Strategy 4: Random baseline pairs")
    remaining_quota = max_pairs - len(pairs)
    random_quota = min(remaining_quota, max(100, int(max_pairs * 0.1)))  # At least 10% random
    logger.info(f"    Adding {random_quota} random pairs")

    attempts = 0
    max_attempts = random_quota * 10
    while len(pairs) < max_pairs and attempts < max_attempts:
        idx1, idx2 = np.random.choice(n_methods, 2, replace=False)
        pair = (min(idx1, idx2), max(idx1, idx2))
        if pair not in pairs and pair not in exclude_pairs:
            pairs.append(pair)
        attempts += 1

    # Deduplicate and limit
    pairs = list(set(pairs))[:max_pairs]

    logger.info(f"  Total pairs selected: {len(pairs)}")
    return pairs


async def filter_pairs_by_embedding_similarity(
    methods_df: pd.DataFrame,
    candidate_pairs: List[Tuple[int, int]],
    config: dict,
    max_similarity: float = 0.95   # Redundancy threshold
) -> Tuple[List[Tuple[int, int]], List[Dict]]:
    """
    Filter pairs using embedding similarity to focus LLM on meaningful cases.

    Logic:
    - Too similar (>0.95): Likely redundant/duplicate, auto-classify without LLM
    - All others: USE LLM for detailed analysis

    Note: Low similarity does NOT mean incompatible - different methods can be
    highly complementary or synergistic. Only filter obvious duplicates.

    Returns:
        - filtered_pairs: Pairs to send to LLM
        - auto_classified: Pre-classified pairs (without LLM)
    """
    logger.info("Pre-filtering pairs by embedding similarity...")

    # Use existing embedding infrastructure
    from src.embeddings import EmbeddingService
    from src.data import Method

    # Generate embeddings using configured API
    logger.info("  Generating embeddings via API...")
    embedding_service = EmbeddingService(config)

    # Convert DataFrame to Method objects
    methods = [
        Method(
            index=idx,
            name=row['Method'],
            description=row['Description'],
            source=row.get('Source', 'Unknown')
        )
        for idx, row in methods_df.iterrows()
    ]

    embeddings_dict = await embedding_service.generate_embeddings(methods)

    # Build embedding matrix
    embeddings = np.array([embeddings_dict[idx] for idx in range(len(methods))])

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / (norms + 1e-8)

    filtered_pairs = []
    auto_classified = []
    stats = {'redundant': 0, 'kept': 0}

    logger.info("  Filtering pairs...")
    for idx1, idx2 in tqdm(candidate_pairs, desc="Filtering"):
        # Compute cosine similarity
        similarity = float(np.dot(normalized_embeddings[idx1], normalized_embeddings[idx2]))

        method_a = methods_df.iloc[idx1]['Method']
        method_b = methods_df.iloc[idx2]['Method']

        if similarity > max_similarity:
            # Too similar - likely redundant/duplicate
            stats['redundant'] += 1
            auto_classified.append({
                'method_a': method_a,
                'method_b': method_b,
                'method_a_index': idx1,
                'method_b_index': idx2,
                'compatibility_score': 0.25,  # Low - redundant
                'relationship_type': 'redundant',
                'resource_conflict': -0.8,
                'conceptual_overlap': -0.9,  # Very high overlap
                'philosophical_alignment': 0.8,
                'implementation_fit': -0.5,
                'recommendation': 'choose_one',
                'specific_concern': f'Very high embedding similarity ({similarity:.3f}) suggests redundancy',
                'overlap_analysis': {
                    'same_problem': True,
                    'same_role': True,
                    'same_timing': False,
                    'same_output': True,
                    'causes_confusion': True,
                    'has_problematic_overlap': True,
                    'overlap_type': 'redundant'
                },
                'auto_classified': True,
                'embedding_similarity': float(similarity)
            })
        else:
            # Send to LLM for detailed analysis (may be compatible, synergistic, or incompatible)
            filtered_pairs.append((idx1, idx2))
            stats['kept'] += 1

    reduction_pct = (stats['redundant'] / len(candidate_pairs)) * 100
    logger.info(f"  ✓ Filtered: {stats['kept']}/{len(candidate_pairs)} pairs kept ({reduction_pct:.1f}% reduction)")
    logger.info(f"    - Auto-classified {stats['redundant']} redundant pairs (>{max_similarity})")
    logger.info(f"  → Sending {len(filtered_pairs)} pairs to LLM")
    logger.info(f"  → Auto-classified {len(auto_classified)} pairs without LLM")

    return filtered_pairs, auto_classified


async def analyze_compatibility_robust(methods_df: pd.DataFrame,
                                       analyzer: CompatibilityAnalyzer,
                                       config: dict,
                                       scores_df: pd.DataFrame = None,
                                       max_pairs: int = 500,
                                       use_embedding_filter: bool = True,
                                       exhaustive: bool = False,
                                       exclude_pairs: set = None) -> List[Dict]:
    """
    Analyze compatibility using sampling and structured analysis.

    Args:
        exhaustive: If True, use exhaustive pair generation instead of strategic sampling
        exclude_pairs: Set of (idx1, idx2) tuples already analyzed (will be skipped)
    """

    # Generate pairs: exhaustive or strategic
    if exhaustive:
        pairs_to_analyze = generate_all_remaining_pairs(methods_df, max_pairs, exclude_pairs)
    else:
        pairs_to_analyze = sample_method_pairs_strategic(methods_df, scores_df, max_pairs, exclude_pairs)

    logger.info(f"Sampled {len(pairs_to_analyze)} method pairs for compatibility analysis...")

    # Optional: Pre-filter using embeddings to auto-classify obvious redundancies
    auto_classified = []
    if use_embedding_filter:
        pairs_to_analyze, auto_classified = await filter_pairs_by_embedding_similarity(
            methods_df,
            pairs_to_analyze,
            config,
            max_similarity=0.95
        )
        logger.info(f"After embedding filter: {len(pairs_to_analyze)} pairs need LLM analysis")

    # Analyze remaining pairs with LLM
    logger.info(f"Analyzing {len(pairs_to_analyze)} pairs with LLM...")
    results = []
    semaphore = asyncio.Semaphore(analyzer.max_concurrent)

    async def analyze_pair_with_semaphore(idx1: int, idx2: int):
        async with semaphore:
            method_a = methods_df.iloc[idx1].to_dict()
            method_b = methods_df.iloc[idx2].to_dict()

            # First check for overlap
            overlap_result = await analyzer.analyze_overlap(method_a, method_b)

            # Then analyze compatibility
            compat_result = await analyzer.analyze_compatibility(method_a, method_b)

            # Combine results
            combined = {
                **compat_result,
                'overlap_analysis': overlap_result,
                'method_a_index': idx1,
                'method_b_index': idx2
            }

            return combined

    # Create tasks
    tasks = [analyze_pair_with_semaphore(idx1, idx2) for idx1, idx2 in pairs_to_analyze]

    # Execute with progress bar
    llm_results = await tqdm_asyncio.gather(*tasks, desc="Analyzing pairs")

    # Combine LLM results with auto-classified results
    all_results = auto_classified + llm_results

    logger.info(f"Total results: {len(all_results)} ({len(auto_classified)} auto-classified + {len(llm_results)} LLM-analyzed)")

    return all_results


# ============================================================================
# Phase 3: Validation & Consistency Checks
# ============================================================================

def validate_compatibility_assessment(method_a: Dict, method_b: Dict,
                                      llm_result: Dict) -> Dict:
    """
    Validate compatibility assessment using multiple checks.
    """

    validations = {
        'source_check': 'ok',
        'overlap_check': 'ok',
        'score_consistency': 'ok',
        'warnings': []
    }

    # 1. Overlap consistency check
    overlap_data = llm_result.get('overlap_analysis', {})
    if overlap_data.get('has_problematic_overlap', False):
        if llm_result.get('compatibility_score', 0.5) > 0.6:
            validations['overlap_check'] = 'warning'
            validations['warnings'].append(
                'Problematic overlap detected but high compatibility score'
            )

    # 2. Score consistency check (only for results with dimension fields)
    if 'resource_conflict' in llm_result:
        dim_scores = [
            llm_result.get('resource_conflict', 0),
            llm_result.get('conceptual_overlap', 0),
            llm_result.get('philosophical_alignment', 0),
            llm_result.get('implementation_fit', 0)
        ]
        avg_dim = np.mean(dim_scores)

        # Expected compatibility should be roughly (avg_dim + 1) / 2  (normalize -1,1 to 0,1)
        expected_compat = (avg_dim + 1) / 2
        actual_compat = llm_result.get('compatibility_score', 0.5)

        if abs(expected_compat - actual_compat) > 0.3:
            validations['score_consistency'] = 'warning'
            validations['warnings'].append(
                f'Dimension scores ({avg_dim:.2f}) inconsistent with compatibility ({actual_compat:.2f})'
            )

    return validations


# ============================================================================
# Phase 4: Building Compatible Sets
# ============================================================================

def reclassify_relationship(result: Dict) -> str:
    """
    Reclassify relationship based on refined rules.

    Rules:
    1. score < 0.7 → incompatible
    2. score >= 0.95 AND overlap_type != 'conflicting' → synergistic
    3. same_problem=false AND same_output=false AND overlap_type='none' → nonrelated
    4. Everything else → compatible (implicit, not stored in sparse graph)
    """
    score = result.get('compatibility_score', 0.5)
    overlap = result.get('overlap_analysis', {})

    # Rule 1: Incompatible
    if score < 0.7:
        return 'incompatible'

    # Rule 2: Synergistic (but not if conflicting)
    overlap_type = overlap.get('overlap_type', 'none')
    if score >= 0.95 and overlap_type != 'conflicting':
        return 'synergistic'

    # Rule 3: Non-related
    same_problem = overlap.get('same_problem', False)
    same_output = overlap.get('same_output', False)
    if not same_problem and not same_output and overlap_type == 'none':
        return 'nonrelated'

    # Rule 4: Compatible (implicit)
    return 'compatible'


def build_sparse_compatibility_graph(
    methods_df: pd.DataFrame,
    compatibility_results: List[Dict]
) -> Dict:
    """
    Build sparse graph storing only actionable relationships.

    Edge types stored:
    - incompatible: score < 0.7 (conflicts to avoid)
    - synergistic: score >= 0.95 (combinations to seek)

    Implicit (not stored):
    - compatible: 0.7 <= score < 0.95
    - nonrelated: different purposes, no overlap (no constraints)

    Returns dict with sparse graph data and metadata.
    """
    logger.info("Building sparse compatibility graph (actionable edges only)...")

    # Reclassify all relationships
    reclassified = []
    for result in compatibility_results:
        result_copy = result.copy()
        result_copy['refined_relationship'] = reclassify_relationship(result)
        reclassified.append(result_copy)

    # Separate by refined type
    incompatible = []
    synergistic = []
    nonrelated = []
    compatible_implicit = []

    for result in reclassified:
        rel_type = result['refined_relationship']

        if rel_type == 'incompatible':
            incompatible.append(result)
        elif rel_type == 'synergistic':
            synergistic.append(result)
        elif rel_type == 'nonrelated':
            nonrelated.append(result)
        else:  # compatible
            compatible_implicit.append(result)

    # Build edge list (ONLY incompatible and synergistic)
    edges = []

    for result in incompatible:
        score = result.get('compatibility_score', 0)
        edges.append({
            'source': int(result['method_a_index']),  # Convert to native Python int
            'target': int(result['method_b_index']),  # Convert to native Python int
            'type': 'incompatible',
            'score': float(score),  # Convert to native Python float
            'strength': float(1.0 - score),  # Repulsion strength
            'concern': result.get('specific_concern', 'Unknown'),
            'recommendation': result.get('recommendation', 'avoid')
        })

    for result in synergistic:
        score = result.get('compatibility_score', 0)
        edges.append({
            'source': int(result['method_a_index']),  # Convert to native Python int
            'target': int(result['method_b_index']),  # Convert to native Python int
            'type': 'synergistic',
            'score': float(score),  # Convert to native Python float
            'strength': float(score),  # Attraction strength
            'relationship': result.get('relationship_type', 'synergistic')
        })

    # Calculate statistics
    total_pairs = len(reclassified)
    implicit_count = len(compatible_implicit) + len(nonrelated)  # Both are implicit

    logger.info(f"  Sparse graph classification:")
    logger.info(f"    Incompatible: {len(incompatible)} edges (STORED - avoid)")
    logger.info(f"    Synergistic: {len(synergistic)} edges (STORED - combine)")
    logger.info(f"    Non-related: {len(nonrelated)} pairs (implicit - no constraints)")
    logger.info(f"    Compatible: {len(compatible_implicit)} pairs (implicit - work fine)")
    logger.info(f"  Total edges stored: {len(edges)} (vs {total_pairs} total pairs)")
    logger.info(f"  Graph sparsity: {(1 - len(edges)/total_pairs)*100:.1f}% reduction")

    # Prepare sparse graph data
    sparse_graph = {
        'nodes': [
            {
                'id': int(idx),
                'name': row['Method'],
                'source': row['Source']
            }
            for idx, row in methods_df.iterrows()
        ],
        'edges': edges,
        'metadata': {
            'total_methods': int(len(methods_df)),
            'total_pairs_analyzed': int(total_pairs),
            'edges_stored': int(len(edges)),
            'sparsity_reduction': f"{(1 - len(edges)/total_pairs)*100:.1f}%",
            'implicit_assumption': 'No edge = compatible, non-related, or no constraints',
            'implicit_pairs': int(implicit_count),
            'implicit_message': 'Pairs without edges can coexist - no conflicts or special synergies',
            'classification_rules': {
                'incompatible': 'score < 0.7 (STORED - avoid these pairs)',
                'synergistic': 'score >= 0.95 AND overlap_type != conflicting (STORED - combine these)',
                'compatible': '0.7 <= score < 0.95 (implicit - work fine together)',
                'nonrelated': 'different purposes, no overlap (implicit - no constraints)'
            }
        },
        'edge_categories': {
            'incompatible': [e for e in edges if e['type'] == 'incompatible'],
            'synergistic': [e for e in edges if e['type'] == 'synergistic']
        },
        'statistics': {
            'incompatible_count': int(len(incompatible)),
            'synergistic_count': int(len(synergistic)),
            'nonrelated_count': int(len(nonrelated)),
            'compatible_count': int(len(compatible_implicit)),
            'implicit_total': int(implicit_count),
            'avg_incompatible_score': float(np.mean([r['compatibility_score'] for r in incompatible])) if incompatible else 0.0,
            'avg_synergistic_score': float(np.mean([r['compatibility_score'] for r in synergistic])) if synergistic else 0.0
        }
    }

    return sparse_graph


def build_compatibility_graph(methods_df: pd.DataFrame,
                              compatibility_results: List[Dict],
                              min_compatibility: float = 0.6) -> Tuple[nx.Graph, List[Dict]]:
    """
    Build network graph of method compatibility and find compatible sets.

    LEGACY: This builds the full dense graph (75K edges).
    Use build_sparse_compatibility_graph() for optimized version.
    """

    logger.info("Building compatibility graph (LEGACY - full graph)...")

    G = nx.Graph()

    # Add nodes (methods)
    for idx, row in methods_df.iterrows():
        G.add_node(idx,
                   name=row['Method'],
                   source=row['Source'])

    # Add edges (compatibility)
    for result in compatibility_results:
        score = result.get('compatibility_score', 0)
        if score >= min_compatibility:
            m1_idx = result['method_a_index']
            m2_idx = result['method_b_index']

            G.add_edge(m1_idx, m2_idx,
                      weight=score,
                      relationship=result.get('relationship_type', 'compatible'))

    logger.info(f"  Graph: {len(G.nodes)} nodes, {len(G.edges)} edges")

    # Find cliques (fully compatible sets)
    logger.info("Finding compatible method sets...")
    cliques = list(nx.find_cliques(G))

    # Analyze cliques
    compatible_sets = []
    for clique in cliques:
        if len(clique) >= 3:  # At least 3 methods
            # Calculate average compatibility within clique
            edges_in_clique = []
            for i, m1 in enumerate(clique):
                for m2 in clique[i+1:]:
                    if G.has_edge(m1, m2):
                        edges_in_clique.append(G.edges[m1, m2]['weight'])

            if edges_in_clique:
                compatible_sets.append({
                    'methods': [methods_df.iloc[idx]['Method'] for idx in clique],
                    'method_indices': clique,
                    'size': len(clique),
                    'avg_compatibility': np.mean(edges_in_clique),
                    'min_compatibility': np.min(edges_in_clique),
                    'sources': [methods_df.iloc[idx]['Source'] for idx in clique]
                })

    # Sort by size and compatibility
    compatible_sets.sort(key=lambda x: (x['size'], x['avg_compatibility']), reverse=True)

    logger.info(f"  Found {len(compatible_sets)} compatible sets (size >= 3)")

    return G, compatible_sets


def analyze_incompatibilities(compatibility_results: List[Dict],
                              methods_df: pd.DataFrame,
                              max_incompatibility: float = 0.7) -> List[Dict]:
    """
    Analyze and report incompatible method pairs.

    Uses refined rule: score < 0.7 → incompatible
    """

    logger.info("Analyzing incompatibilities...")

    incompatibilities = []
    for result in compatibility_results:
        score = result.get('compatibility_score', 0.5)
        if score < max_incompatibility:
            incompat_entry = {
                'method_a': result['method_a'],
                'method_b': result['method_b'],
                'compatibility_score': score,
                'relationship_type': result.get('relationship_type', 'unknown'),
                'concern': result.get('specific_concern', 'Unknown'),
                'recommendation': result.get('recommendation', 'unknown')
            }

            # Include dimension fields if present (backward compatibility)
            if 'resource_conflict' in result:
                incompat_entry['resource_conflict'] = result['resource_conflict']
                incompat_entry['conceptual_overlap'] = result['conceptual_overlap']
                incompat_entry['philosophical_alignment'] = result['philosophical_alignment']

            # Include overlap_analysis if present
            if 'overlap_analysis' in result:
                incompat_entry['overlap_analysis'] = result['overlap_analysis']

            incompatibilities.append(incompat_entry)

    # Sort by compatibility score (lowest first)
    incompatibilities.sort(key=lambda x: x['compatibility_score'])

    logger.info(f"  Found {len(incompatibilities)} incompatible pairs")

    return incompatibilities


def analyze_synergies(compatibility_results: List[Dict],
                     methods_df: pd.DataFrame,
                     min_synergy: float = 0.95) -> List[Dict]:
    """
    Analyze and report synergistic method pairs.

    Uses refined rule: score >= 0.95 AND overlap_type != 'conflicting' → synergistic

    Args:
        compatibility_results: All compatibility analysis results
        methods_df: DataFrame with method information
        min_synergy: Minimum compatibility score to be considered synergistic

    Returns:
        List of synergistic pairs with full overlap_analysis data
    """

    logger.info("Analyzing synergies...")

    synergies = []
    for result in compatibility_results:
        score = result.get('compatibility_score', 0.5)
        overlap = result.get('overlap_analysis', {})
        overlap_type = overlap.get('overlap_type', 'none')

        # Apply refined rule: score >= 0.95 AND overlap_type != 'conflicting'
        if score >= min_synergy and overlap_type != 'conflicting':
            synergy_entry = {
                'method_a': result['method_a'],
                'method_b': result['method_b'],
                'compatibility_score': score,
                'relationship_type': result.get('relationship_type', 'synergistic'),
                'recommendation': result.get('recommendation', 'combine'),
                'concern': result.get('specific_concern', 'none')
            }

            # Include dimension fields if present
            if 'resource_conflict' in result:
                synergy_entry['resource_conflict'] = result['resource_conflict']
                synergy_entry['conceptual_overlap'] = result['conceptual_overlap']
                synergy_entry['philosophical_alignment'] = result['philosophical_alignment']

            # Include overlap_analysis (critical for synergy understanding)
            if 'overlap_analysis' in result:
                synergy_entry['overlap_analysis'] = result['overlap_analysis']

            synergies.append(synergy_entry)

    # Sort by compatibility score (highest first)
    synergies.sort(key=lambda x: x['compatibility_score'], reverse=True)

    logger.info(f"  Found {len(synergies)} synergistic pairs")

    return synergies


# ============================================================================
# Reporting
# ============================================================================

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def generate_lightweight_compatibility_report(methods_df: pd.DataFrame,
                                              compatibility_results: List[Dict],
                                              sparse_graph: Dict,
                                              incompatibilities: List[Dict],
                                              output_dir: Path,
                                              synergies: List[Dict] = None):
    """
    Generate compatibility report using sparse graph (no legacy graph needed).

    Args:
        methods_df: DataFrame with method information
        compatibility_results: All compatibility results
        sparse_graph: Sparse graph representation
        incompatibilities: List of incompatible pairs
        output_dir: Output directory
        synergies: List of synergistic pairs (optional)
    """

    logger.info("Generating lightweight compatibility report...")

    report = {
        'metadata': {
            'total_methods': len(methods_df),
            'pairs_analyzed': len(compatibility_results),
            'incompatibilities_found': len(incompatibilities),
            'synergies_found': len(synergies) if synergies else 0,
            'graph_type': 'sparse',
            'edges_stored': len(sparse_graph['edges'])
        },
        'statistics': {
            'avg_compatibility': float(np.mean([r['compatibility_score'] for r in compatibility_results])),
            'std_compatibility': float(np.std([r['compatibility_score'] for r in compatibility_results])),
            'high_compatibility_pairs': int(sum(1 for r in compatibility_results if r['compatibility_score'] >= 0.7)),
            'low_compatibility_pairs': int(sum(1 for r in compatibility_results if r['compatibility_score'] <= 0.3)),
            'incompatible_count': int(sparse_graph['statistics']['incompatible_count']),
            'synergistic_count': int(sparse_graph['statistics']['synergistic_count']),
            'avg_incompatible_score': float(sparse_graph['statistics']['avg_incompatible_score']),
            'avg_synergistic_score': float(sparse_graph['statistics']['avg_synergistic_score'])
        },
        'top_incompatibilities': incompatibilities[:20],  # Top 20
        'sparse_graph_summary': {
            'total_edges': len(sparse_graph['edges']),
            'incompatible_edges': len(sparse_graph['edge_categories']['incompatible']),
            'synergistic_edges': len(sparse_graph['edge_categories']['synergistic']),
            'sparsity_reduction': sparse_graph['metadata']['sparsity_reduction']
        }
    }

    # Add synergies if provided
    if synergies:
        report['top_synergies'] = synergies[:50]  # Top 50 synergistic pairs

    # Save report
    report_file = output_dir / 'compatibility_analysis.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)

    logger.info(f"✅ Report saved to: {report_file}")

    return report

def generate_compatibility_report(methods_df: pd.DataFrame,
                                  compatibility_results: List[Dict],
                                  graph: nx.Graph,
                                  compatible_sets: List[Dict],
                                  incompatibilities: List[Dict],
                                  output_dir: Path):
    """
    Generate comprehensive compatibility analysis report.
    """

    logger.info("Generating compatibility report...")

    report = {
        'metadata': {
            'total_methods': len(methods_df),
            'pairs_analyzed': len(compatibility_results),
            'compatible_sets_found': len(compatible_sets),
            'incompatibilities_found': len(incompatibilities)
        },
        'statistics': {
            'avg_compatibility': np.mean([r['compatibility_score'] for r in compatibility_results]),
            'std_compatibility': np.std([r['compatibility_score'] for r in compatibility_results]),
            'high_compatibility_pairs': sum(1 for r in compatibility_results if r['compatibility_score'] >= 0.7),
            'low_compatibility_pairs': sum(1 for r in compatibility_results if r['compatibility_score'] <= 0.3),
        },
        'top_compatible_sets': compatible_sets[:10],  # Top 10
        'top_incompatibilities': incompatibilities[:20],  # Top 20
        'all_compatibility_results': compatibility_results
    }

    # Add dimension statistics only if results have dimension fields (backward compatibility)
    if any('resource_conflict' in r for r in compatibility_results):
        report['dimension_statistics'] = {
            'resource_conflict': {
                'mean': np.mean([r.get('resource_conflict', 0) for r in compatibility_results]),
                'std': np.std([r.get('resource_conflict', 0) for r in compatibility_results])
            },
            'conceptual_overlap': {
                'mean': np.mean([r.get('conceptual_overlap', 0) for r in compatibility_results]),
                'std': np.std([r.get('conceptual_overlap', 0) for r in compatibility_results])
            },
            'philosophical_alignment': {
                'mean': np.mean([r.get('philosophical_alignment', 0) for r in compatibility_results]),
                'std': np.std([r.get('philosophical_alignment', 0) for r in compatibility_results])
            },
            'implementation_fit': {
                'mean': np.mean([r.get('implementation_fit', 0) for r in compatibility_results]),
                'std': np.std([r.get('implementation_fit', 0) for r in compatibility_results])
            }
        }

    # Save report
    output_file = output_dir / 'compatibility_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)

    logger.info(f"  Report saved to: {output_file}")

    # Generate summary
    print("\n" + "="*80)
    print("COMPATIBILITY ANALYSIS SUMMARY")
    print("="*80)
    print(f"\nMethods analyzed: {len(methods_df)}")
    print(f"Method pairs analyzed: {len(compatibility_results)}")
    print(f"\nCompatibility statistics:")
    print(f"  Average compatibility: {report['statistics']['avg_compatibility']:.3f}")
    print(f"  Std deviation: {report['statistics']['std_compatibility']:.3f}")
    print(f"  High compatibility pairs (>0.7): {report['statistics']['high_compatibility_pairs']}")
    print(f"  Low compatibility pairs (<0.3): {report['statistics']['low_compatibility_pairs']}")

    if 'dimension_statistics' in report:
        print(f"\nDimension averages:")
        for dim, stats in report['dimension_statistics'].items():
            print(f"  {dim}: {stats['mean']:.3f} (±{stats['std']:.3f})")

    print(f"\nCompatible sets found: {len(compatible_sets)}")
    if compatible_sets:
        print(f"\nTop 5 compatible sets:")
        for i, cset in enumerate(compatible_sets[:5], 1):
            print(f"  {i}. {cset['size']} methods (avg compat: {cset['avg_compatibility']:.3f})")
            for method in cset['methods'][:3]:
                print(f"     - {method}")
            if len(cset['methods']) > 3:
                print(f"     ... and {len(cset['methods'])-3} more")

    print(f"\nIncompatibilities found: {len(incompatibilities)}")
    if incompatibilities:
        print(f"\nTop 5 incompatible pairs:")
        for i, incomp in enumerate(incompatibilities[:5], 1):
            print(f"  {i}. {incomp['method_a']} ⚔️  {incomp['method_b']}")
            print(f"     Compatibility: {incomp['compatibility_score']:.3f}")
            print(f"     Concern: {incomp['concern']}")

    print("\n" + "="*80)

    return report


# ============================================================================
# Main
# ============================================================================

async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Analyze method compatibility using structured framework'
    )
    parser.add_argument(
        '--input',
        default='input/methods_deduplicated.csv',
        help='Input CSV file with methods'
    )
    parser.add_argument(
        '--scores',
        default='results/method_scores_12d_deduplicated.json',
        help='Optional: 12D scores JSON for impact-based sampling'
    )
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Configuration file'
    )
    parser.add_argument(
        '--max-pairs',
        type=int,
        default=500,
        help='Maximum number of pairs to analyze'
    )
    parser.add_argument(
        '--output-dir',
        default='results',
        help='Output directory'
    )
    parser.add_argument(
        '--no-embedding-filter',
        action='store_true',
        help='Disable embedding-based pre-filtering (analyze all pairs with LLM)'
    )
    parser.add_argument(
        '--exhaustive',
        action='store_true',
        help='Use exhaustive pair generation (systematic coverage) instead of strategic sampling'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-analysis even if checkpoint exists'
    )
    parser.add_argument(
        '--report-only',
        action='store_true',
        help='Skip analysis and only regenerate reports from checkpoint'
    )
    parser.add_argument(
        '--build-graph',
        action='store_true',
        help='Build compatibility graph and generate reports (expensive, use after analysis complete)'
    )

    args = parser.parse_args()

    # Load configuration
    logger.info("Loading configuration...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load methods
    logger.info(f"Loading methods from {args.input}...")
    methods_df = pd.read_csv(args.input, sep='|')
    logger.info(f"  Loaded {len(methods_df)} methods")

    # Load scores if available
    scores_df = None
    if Path(args.scores).exists():
        logger.info(f"Loading scores from {args.scores}...")
        with open(args.scores, 'r') as f:
            scores_data = json.load(f)

        # Extract methods with scores
        if 'methods' in scores_data:
            scores_list = []
            for method in scores_data['methods']:
                scores_list.append({
                    'name': method['name'],
                    'impact_potential': method.get('impact_potential', 50)
                })
            scores_df = pd.DataFrame(scores_list)
            logger.info(f"  Loaded scores for {len(scores_df)} methods")

    # Initialize analyzer
    analyzer = CompatibilityAnalyzer(config)

    # Check for checkpoint
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = output_dir / 'compatibility_checkpoint.pkl'

    existing_results = []
    analyzed_pairs = set()

    if checkpoint_file.exists() and not args.force:
        logger.info(f"Loading checkpoint from: {checkpoint_file}")
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
            raw_results = checkpoint.get('results', [])
            raw_pairs = checkpoint.get('analyzed_pairs', set())

            # Deduplicate results (keep first occurrence of each normalized pair)
            seen_pairs = set()
            existing_results = []
            for result in raw_results:
                idx_a = result['method_a_index']
                idx_b = result['method_b_index']
                normalized_pair = (min(idx_a, idx_b), max(idx_a, idx_b))
                if normalized_pair not in seen_pairs:
                    existing_results.append(result)
                    seen_pairs.add(normalized_pair)

            # Normalize all pairs to (min, max) for consistency
            analyzed_pairs = {(min(a, b), max(a, b)) for a, b in raw_pairs}

        if len(raw_results) > len(existing_results):
            logger.info(f"  Removed {len(raw_results) - len(existing_results)} duplicate results from checkpoint")
        logger.info(f"  Loaded {len(existing_results)} unique results")
        logger.info(f"  Already analyzed {len(analyzed_pairs)} unique pairs")

    # Skip analysis if report-only mode
    if args.report_only:
        if not existing_results:
            logger.error("No existing results found. Cannot generate report in --report-only mode.")
            logger.error(f"Run without --report-only to perform analysis first.")
            return
        logger.info("Report-only mode: skipping analysis, using existing results")
        compatibility_results = existing_results
        new_results = []
    else:
        # Analyze NEW compatibility pairs (excluding already-analyzed ones)
        logger.info(f"Analyzing {args.max_pairs} NEW pairs...")
        new_results = await analyze_compatibility_robust(
            methods_df,
            analyzer,
            config,
            scores_df=scores_df,
            max_pairs=args.max_pairs,
            use_embedding_filter=not args.no_embedding_filter,  # Use embedding filter by default
            exhaustive=args.exhaustive,  # Use exhaustive pair generation if requested
            exclude_pairs=analyzed_pairs
        )

        # Merge results
        compatibility_results = existing_results + new_results
        logger.info(f"Total results after merge: {len(compatibility_results)}")

        # Update analyzed pairs (normalize to min, max order)
        for result in new_results:
            idx_a = result['method_a_index']
            idx_b = result['method_b_index']
            pair = (min(idx_a, idx_b), max(idx_a, idx_b))
            analyzed_pairs.add(pair)

        # Save checkpoint immediately
        logger.info(f"Saving checkpoint to: {checkpoint_file}")
        with open(checkpoint_file, 'wb') as f:
            pickle.dump({
                'results': compatibility_results,
                'analyzed_pairs': analyzed_pairs
            }, f)
        logger.info("  Checkpoint saved!")

    # Validate new results only (skip already-validated ones)
    if new_results:
        logger.info("Validating new results...")
        for result in new_results:
            idx_a = result['method_a_index']
            idx_b = result['method_b_index']
            method_a = methods_df.iloc[idx_a].to_dict()
            method_b = methods_df.iloc[idx_b].to_dict()

            validation = validate_compatibility_assessment(method_a, method_b, result)
            result['validation'] = validation

    # Determine if we should build the graph
    should_build_graph = args.build_graph or args.report_only

    if should_build_graph:
        logger.info("\n" + "="*80)
        logger.info("BUILDING COMPATIBILITY GRAPHS & REPORTS")
        logger.info("="*80)

        # Build SPARSE graph (optimized, refined classification)
        logger.info("\n1. Building sparse graph (refined classification)...")
        sparse_graph = build_sparse_compatibility_graph(
            methods_df,
            compatibility_results
        )

        # Save sparse graph
        sparse_graph_file = output_dir / 'compatibility_graph_sparse.json'
        with open(sparse_graph_file, 'w') as f:
            json.dump(sparse_graph, f, indent=2)
        logger.info(f"✅ Sparse graph saved to: {sparse_graph_file}")
        logger.info(f"   Edges: {len(sparse_graph['edges'])} (vs {len(compatibility_results)} total pairs)")
        logger.info(f"   Reduction: {sparse_graph['metadata']['sparsity_reduction']}")

        # Analyze incompatibilities (using refined rule: < 0.7)
        logger.info("\n2. Analyzing incompatibilities...")
        incompatibilities = analyze_incompatibilities(
            compatibility_results,
            methods_df,
            max_incompatibility=0.7
        )

        # Analyze synergies (using refined rule: >= 0.95 and not conflicting)
        logger.info("\n3. Analyzing synergies...")
        synergies = analyze_synergies(
            compatibility_results,
            methods_df,
            min_synergy=0.95
        )

        # Generate lightweight report (without legacy graph)
        logger.info("\n4. Generating compatibility report...")
        report = generate_lightweight_compatibility_report(
            methods_df,
            compatibility_results,
            sparse_graph,
            incompatibilities,
            output_dir,
            synergies=synergies
        )

        logger.info("\n✅ Compatibility analysis complete!")
        logger.info("\n📊 Output files:")
        logger.info(f"  🎯 {sparse_graph_file} - Sparse graph (optimized for visualization)")
        logger.info(f"  📊 {output_dir / 'compatibility_analysis.json'} - Analysis report with statistics")

    else:
        logger.info("\n" + "="*80)
        logger.info("ANALYSIS COMPLETE - Graph building skipped")
        logger.info("="*80)
        logger.info(f"✅ Saved {len(compatibility_results)} results to: {checkpoint_file}")
        logger.info(f"   Analyzed pairs: {len(analyzed_pairs)}")
        logger.info("\nTo build compatibility graph and reports, run:")
        logger.info(f"  python3 {Path(__file__).name} --report-only --build-graph")
        logger.info("="*80)


if __name__ == "__main__":
    asyncio.run(main())
