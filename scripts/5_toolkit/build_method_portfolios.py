#!/usr/bin/env python3
"""
Enhanced Situational Toolkit Generator

Builds optimized method toolkits for specific organizational contexts using:
- 12D scoring data
- Pairwise compatibility matrices
- Category synergy patterns
- Organizational context profiles
"""
import json
import logging
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Compatibility Matrix Loader
# ============================================================================

class CompatibilityMatrix:
    """
    Manages pre-computed pairwise compatibility scores and overlap analysis.
    Provides O(1) lookup for method pair compatibility and relationship data.
    """

    def __init__(self, compatibility_data: Dict):
        """
        Initialize from compatibility checkpoint data.

        Args:
            compatibility_data: Dict with 'results' list from compatibility analysis
        """
        logger.info("Building compatibility matrix...")

        self.results = compatibility_data.get('results', [])
        self.method_names = set()
        self.method_to_idx = {}
        self.idx_to_method = {}

        # Extract all unique methods and build index mapping
        for result in self.results:
            self.method_names.add(result['method_a'])
            self.method_names.add(result['method_b'])

        self.method_names = sorted(list(self.method_names))
        self.n_methods = len(self.method_names)

        for idx, name in enumerate(self.method_names):
            self.method_to_idx[name] = idx
            self.idx_to_method[idx] = name

        # Build N×N compatibility matrix (scores only)
        self.matrix = np.ones((self.n_methods, self.n_methods))  # Default 1.0 (self-compatibility)

        # Build lookup dict for full result data (includes overlap_analysis)
        self.pair_data = {}

        # Fill matrix and lookup dict
        for result in self.results:
            method_a = result['method_a']
            method_b = result['method_b']
            score = result.get('compatibility_score', 0.5)

            idx_a = self.method_to_idx[method_a]
            idx_b = self.method_to_idx[method_b]

            # Symmetric matrix (scores only)
            self.matrix[idx_a][idx_b] = score
            self.matrix[idx_b][idx_a] = score

            # Store full result data both ways for easy lookup
            key_ab = (method_a, method_b)
            key_ba = (method_b, method_a)
            self.pair_data[key_ab] = result
            self.pair_data[key_ba] = result

        logger.info(f"  Matrix built: {self.n_methods}×{self.n_methods}")
        logger.info(f"  Pairs loaded: {len(self.results)}")
        logger.info(f"  Avg compatibility: {self.get_avg_compatibility():.3f}")

    def get_compatibility(self, method_a: str, method_b: str) -> float:
        """Get compatibility score between two methods."""
        if method_a not in self.method_to_idx or method_b not in self.method_to_idx:
            logger.warning(f"Method not found in matrix: {method_a} or {method_b}")
            return 0.5  # Default neutral compatibility

        idx_a = self.method_to_idx[method_a]
        idx_b = self.method_to_idx[method_b]

        return self.matrix[idx_a][idx_b]

    def get_compatibility_by_idx(self, idx_a: int, idx_b: int) -> float:
        """Get compatibility by index (faster)."""
        return self.matrix[idx_a][idx_b]

    def get_compatibilities_with(self, method_name: str) -> np.ndarray:
        """Get all compatibility scores for a method (returns array)."""
        if method_name not in self.method_to_idx:
            return np.zeros(self.n_methods)

        idx = self.method_to_idx[method_name]
        return self.matrix[idx]

    def get_avg_compatibility(self) -> float:
        """Get average compatibility across all pairs."""
        # Exclude diagonal (self-compatibility)
        mask = ~np.eye(self.n_methods, dtype=bool)
        return self.matrix[mask].mean()

    def get_method_idx(self, method_name: str) -> Optional[int]:
        """Get index for method name."""
        return self.method_to_idx.get(method_name)

    def get_method_name(self, idx: int) -> Optional[str]:
        """Get method name for index."""
        return self.idx_to_method.get(idx)

    def get_pair_data(self, method_a: str, method_b: str) -> Optional[Dict]:
        """
        Get full compatibility data for a method pair including overlap_analysis.

        Returns:
            Dict with compatibility_score, relationship_type, overlap_analysis, etc.
            None if pair not found.
        """
        return self.pair_data.get((method_a, method_b))

    def get_overlap_analysis(self, method_a: str, method_b: str) -> Optional[Dict]:
        """
        Get overlap analysis for a method pair.

        Returns:
            Dict with same_problem, same_role, overlap_type, etc.
            None if pair not found or no overlap_analysis available.
        """
        pair_data = self.get_pair_data(method_a, method_b)
        if pair_data:
            return pair_data.get('overlap_analysis')
        return None

    def is_synergistic(self, method_a: str, method_b: str) -> bool:
        """
        Check if two methods are synergistic using refined rules:
        score >= 0.95 AND overlap_type != 'conflicting'
        """
        pair_data = self.get_pair_data(method_a, method_b)
        if not pair_data:
            return False

        score = pair_data.get('compatibility_score', 0)
        overlap = pair_data.get('overlap_analysis', {})
        overlap_type = overlap.get('overlap_type', 'none')

        return score >= 0.95 and overlap_type != 'conflicting'

    def is_incompatible(self, method_a: str, method_b: str) -> bool:
        """
        Check if two methods are incompatible using refined rules:
        score < 0.7
        """
        pair_data = self.get_pair_data(method_a, method_b)
        if not pair_data:
            return False

        score = pair_data.get('compatibility_score', 0)
        return score < 0.7

    def has_problematic_overlap(self, method_a: str, method_b: str) -> bool:
        """Check if two methods have problematic overlap."""
        overlap = self.get_overlap_analysis(method_a, method_b)
        if overlap:
            return overlap.get('has_problematic_overlap', False)
        return False


def load_compatibility_matrix(pkl_file: str) -> CompatibilityMatrix:
    """Load compatibility matrix from pickle file."""
    logger.info(f"Loading compatibility data from {pkl_file}...")

    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    return CompatibilityMatrix(data)


# ============================================================================
# Category Synergy Definitions
# ============================================================================

def load_category_synergies() -> Dict:
    """
    Load category synergy definitions from dendrogram_categories.json.

    These are generated by extract_dendrogram_synergies.py using UMAP 5D
    reduced cluster centroids with Ward linkage hierarchical clustering.
    """
    categories_path = Path(__file__).parent / "results_semantic_clustering_combined" / "dendrogram_categories.json"

    if not categories_path.exists():
        logger.warning(f"Category synergies file not found: {categories_path}")
        logger.warning("Run extract_dendrogram_synergies.py to generate it")
        return {}

    with open(categories_path, 'r') as f:
        data = json.load(f)

    # Convert list format to dict format expected by the code
    synergies = {}
    for i, cat in enumerate(data.get('categories', [])):
        # Create a key from the category name (lowercase, underscores)
        key = cat['name'].lower().replace(' ', '_').replace('&', 'and').replace('-', '_')
        key = ''.join(c for c in key if c.isalnum() or c == '_')

        synergies[key] = {
            'display_name': cat['name'],
            'categories': cat['clusters'],
            'cluster_names': cat.get('cluster_names', []),
            'reason': f"Semantically related clusters (UMAP 5D based)",
            'strength': cat.get('strength', 'medium'),
            'bonus': cat.get('bonus', 1.15)
        }

    logger.info(f"Loaded {len(synergies)} category synergies from {categories_path}")
    return synergies


# Load synergies at module level (will be populated when module is imported)
CATEGORY_SYNERGIES = load_category_synergies()


# ============================================================================
# Context Profile Definitions
# ============================================================================

def get_context_profiles() -> Dict:
    """
    Define organizational context profiles with multi-dimensional requirements.

    Each profile includes:
    - constraints: Organizational characteristics
    - dimension_weights: How to prioritize 12D dimensions (positive = higher is better, negative = lower is better)
    - preferred_categories: Categories to prioritize
    - avoid_categories: Categories to deprioritize
    - ppp_profile: People-Process-Purpose balance (optional)
    - toolkit_size: Target number of methods
    - min_compatibility: Minimum compatibility threshold for pairs
    - max_per_category: Maximum methods from same category
    """

    return {
        'startup_mvp': {
            'name': 'Startup - MVP Phase',
            'description': 'Small team building initial product, need speed and customer validation',
            'constraints': {
                'team_size': 'small',
                'maturity': 'low',
                'primary_challenge': 'speed',
                'resource_level': 'minimal'
            },
            'dimension_weights': {
                'time_to_value': 0.35,
                'ease_adoption': 0.25,
                'resources_required': -0.20,  # Negative = lower is better
                'impact_potential': 0.15,
                'applicability': 0.05
            },
            'preferred_categories': [],  # Will be loaded from category definitions
            'avoid_categories': [],
            'ppp_profile': {'people': 70, 'process': 20, 'purpose': 85},
            'toolkit_size': 15,
            'min_compatibility': 0.7,
            'max_per_category': 3
        },

        'startup_scaling': {
            'name': 'Startup - Scaling Phase',
            'description': 'Growing team, need to balance quality with speed, building processes',
            'constraints': {
                'team_size': 'medium',
                'maturity': 'medium',
                'primary_challenge': 'quality_speed_balance',
                'resource_level': 'moderate'
            },
            'dimension_weights': {
                'scope': 0.20,
                'impact_potential': 0.25,
                'process_focus': 0.15,
                'time_to_value': 0.20,
                'technical_complexity': -0.10,
                'change_management_difficulty': -0.10
            },
            'preferred_categories': [],  # Will be loaded from category definitions
            'avoid_categories': [],
            'ppp_profile': {'people': 60, 'process': 50, 'purpose': 70},
            'toolkit_size': 15,
            'min_compatibility': 0.65,
            'max_per_category': 3
        },

        'enterprise_transformation': {
            'name': 'Enterprise - Digital Transformation',
            'description': 'Large organization, driving systemic change and innovation',
            'constraints': {
                'team_size': 'large',
                'maturity': 'high',
                'primary_challenge': 'innovation',
                'resource_level': 'high'
            },
            'dimension_weights': {
                'scope': 0.30,
                'temporality': 0.25,
                'impact_potential': 0.25,
                'change_management_difficulty': -0.20
            },
            'preferred_categories': [],  # Will be loaded from category definitions
            'avoid_categories': [],
            'ppp_profile': {'people': 50, 'process': 70, 'purpose': 60},
            'toolkit_size': 15,
            'min_compatibility': 0.6,
            'max_per_category': 3
        },

        'regulated_industry': {
            'name': 'Regulated Industry - Compliance Focus',
            'description': 'Operating under strict regulations, need balance of innovation and compliance',
            'constraints': {
                'industry': 'regulated',
                'primary_challenge': 'compliance_innovation_balance',
                'risk_tolerance': 'low'
            },
            'dimension_weights': {
                'process_focus': 0.30,
                'technical_complexity': 0.10,
                'risk_decision_making': 0.25,
                'applicability': 0.20,
                'ease_adoption': 0.15
            },
            'preferred_categories': [],  # Will be loaded from category definitions
            'avoid_categories': [],
            'ppp_profile': {'people': 40, 'process': 85, 'purpose': 50},
            'toolkit_size': 15,
            'min_compatibility': 0.75,
            'max_per_category': 3
        },

        'hardware_product': {
            'name': 'Hardware Product Development',
            'description': 'Long development cycles, high cost of change, physical constraints',
            'constraints': {
                'domain': 'hardware',
                'primary_challenge': 'long_cycles',
                'change_cost': 'high'
            },
            'dimension_weights': {
                'planning_adaptation': 0.25,
                'risk_decision_making': 0.25,
                'design_development': 0.20,
                'time_to_value': -0.10,  # Long cycles expected
                'impact_potential': 0.20
            },
            'preferred_categories': [],  # Will be loaded from category definitions
            'avoid_categories': [],
            'ppp_profile': {'people': 55, 'process': 70, 'purpose': 65},
            'toolkit_size': 15,
            'min_compatibility': 0.7,
            'max_per_category': 3
        }
    }


# ============================================================================
# Situational Toolkit Builder
# ============================================================================

class SituationalToolkitBuilder:
    """
    Builds optimized toolkits for specific organizational contexts using:
    - 12D scoring data
    - Pairwise compatibility matrices
    - Category synergy patterns
    - Organizational context profiles
    """

    def __init__(self, methods_df: pd.DataFrame, compatibility_matrix: CompatibilityMatrix, category_definitions: Dict = None):
        """
        Initialize toolkit builder.

        Args:
            methods_df: DataFrame with 12D scores and categories
            compatibility_matrix: Pre-computed compatibility matrix
            category_definitions: Category definitions from input/method_categories.json
        """
        self.methods = methods_df.copy()
        self.compatibility = compatibility_matrix
        self.context_profiles = get_context_profiles()
        self.category_definitions = category_definitions or {}

        # Disable category coverage requirement - select best fitting methods only
        # (Previously would add one method from each of 47 categories)
        self.required_categories = []
        logger.info(f"Category coverage requirement: DISABLED (selecting best-fit methods only)")

        # Add method indices to dataframe for quick lookup
        self.methods['method_idx'] = range(len(self.methods))

        # Create name to index mapping
        self.method_name_to_df_idx = {row['name']: idx for idx, row in self.methods.iterrows()}

        # Add category if missing (auto-categorize based on characteristics)
        if 'category' not in self.methods.columns:
            logger.warning("Category column missing - auto-assigning categories based on method characteristics")
            self.methods['category'] = self._auto_categorize()

        logger.info(f"SituationalToolkitBuilder initialized with {len(self.methods)} methods")
        logger.info(f"Context profiles available: {list(self.context_profiles.keys())}")

    def _auto_categorize(self) -> pd.Series:
        """Auto-assign categories based on 12D scores when category is missing."""
        categories = []

        for _, method in self.methods.iterrows():
            # Simple heuristic-based categorization
            people = method.get('people_focus', 50)
            process = method.get('process_focus', 50)
            purpose = method.get('purpose_orientation', 50)
            scope = method.get('scope', 50)
            impact = method.get('impact_potential', 50)

            # Categorization logic
            if people > 70:
                if purpose > 70:
                    cat = 'customer_centric' if purpose > 80 else 'team_collaboration'
                else:
                    cat = 'team_collaboration'
            elif process > 70:
                if scope > 70:
                    cat = 'systems_thinking'
                else:
                    cat = 'flow_optimization'
            elif purpose > 70:
                cat = 'customer_centric'
            elif impact > 70:
                if scope > 60:
                    cat = 'organizational_culture'
                else:
                    cat = 'continuous_improvement'
            else:
                cat = 'general_practices'

            categories.append(cat)

        return pd.Series(categories, index=self.methods.index)

    def _calculate_context_fitness(self, profile: Dict) -> pd.Series:
        """
        Calculate fitness scores for all methods given a context profile.

        Uses dimension_weights from profile to score each method.
        Higher score = better fit for context.

        Args:
            profile: Context profile dict with dimension_weights

        Returns:
            Series with fitness scores (index = df index)
        """
        weights = profile.get('dimension_weights', {})

        fitness = pd.Series(0.0, index=self.methods.index)

        for dimension, weight in weights.items():
            if dimension in self.methods.columns:
                values = self.methods[dimension]

                if weight > 0:
                    # Positive weight: higher dimension value is better
                    fitness += values * weight
                else:
                    # Negative weight: lower dimension value is better
                    fitness += (100 - values) * abs(weight)

        # Normalize to 0-100 scale
        if fitness.max() > 0:
            fitness = (fitness - fitness.min()) / (fitness.max() - fitness.min()) * 100

        return fitness

    def _get_compatible_candidates(
        self,
        current_toolkit: List[int],
        profile: Dict,
        fitness_scores: pd.Series
    ) -> pd.DataFrame:
        """
        Get methods compatible with ALL current toolkit members.

        Args:
            current_toolkit: List of df indices already in toolkit
            profile: Context profile with constraints
            fitness_scores: Pre-computed fitness scores

        Returns:
            DataFrame of compatible candidate methods
        """
        candidates = self.methods.copy()

        # Remove already selected methods
        candidates = candidates[~candidates.index.isin(current_toolkit)]

        # Filter by minimum compatibility with EVERY toolkit member
        min_compatibility = profile.get('min_compatibility', 0.6)

        for df_idx in current_toolkit:
            method_name = candidates.loc[df_idx, 'name'] if df_idx in candidates.index else None

            if not method_name:
                method_name = self.methods.loc[df_idx, 'name']

            # Get compatibility scores for all candidates with this toolkit member
            compat_mask = pd.Series(True, index=candidates.index)

            for cand_idx in candidates.index:
                cand_name = candidates.loc[cand_idx, 'name']

                # Get compatibility score
                compat_score = self.compatibility.get_compatibility(method_name, cand_name)

                if compat_score < min_compatibility:
                    compat_mask[cand_idx] = False

            # Filter candidates
            candidates = candidates[compat_mask]

        # Apply category constraints
        current_categories = self.methods.loc[current_toolkit, 'category'].value_counts()
        max_per_category = profile.get('max_per_category', 3)

        for category, count in current_categories.items():
            if count >= max_per_category:
                candidates = candidates[candidates['category'] != category]

        # Filter out avoided categories
        avoid_categories = profile.get('avoid_categories', [])
        if avoid_categories:
            candidates = candidates[~candidates['category'].isin(avoid_categories)]

        return candidates

    def _calculate_synergy_bonus(
        self,
        candidate_category: str,
        current_categories: pd.Series,
        profile: Dict
    ) -> float:
        """
        Calculate synergy bonus for completing synergy groups (legacy category-based).

        Args:
            candidate_category: Category of candidate method
            current_categories: Categories already in toolkit
            profile: Context profile

        Returns:
            Synergy bonus score (0-1)
        """
        bonus = 0.0

        for synergy_name, synergy_def in CATEGORY_SYNERGIES.items():
            required_cats = set(synergy_def['categories'])

            # Check if candidate would help complete this synergy group
            if candidate_category in required_cats:
                present = set(current_categories.unique()) & required_cats
                would_have = present | {candidate_category}

                # Bonus increases with coverage of synergy group
                coverage = len(would_have) / len(required_cats)

                # Apply strength multiplier
                strength_bonus = synergy_def.get('bonus', 1.0)

                # Partial credit for increasing coverage
                if len(would_have) > len(present):
                    bonus += coverage * (strength_bonus - 1.0)

        return min(bonus, 1.0)  # Cap at 1.0

    def _calculate_pairwise_synergy(
        self,
        candidate_name: str,
        current_toolkit: List[int]
    ) -> float:
        """
        Calculate actual pairwise synergy using overlap_analysis data.

        Uses refined rules to detect:
        - Synergistic pairs (score >= 0.95, not conflicting)
        - Complementary methods (same problem, different approaches)
        - Problematic overlap (should penalize)

        Args:
            candidate_name: Name of candidate method
            current_toolkit: List of df indices in current toolkit

        Returns:
            Synergy score (0-1, can go negative for conflicts)
        """
        synergy_score = 0.0
        synergy_count = 0
        conflict_count = 0

        for tk_idx in current_toolkit:
            tk_name = self.methods.loc[tk_idx, 'name']

            # Get overlap analysis
            overlap = self.compatibility.get_overlap_analysis(candidate_name, tk_name)
            if not overlap:
                continue

            # Check if synergistic (refined rule: score >= 0.95 and not conflicting)
            if self.compatibility.is_synergistic(candidate_name, tk_name):
                synergy_score += 0.3  # Strong bonus for actual synergy
                synergy_count += 1

            # Check for complementary (same problem, but no problematic overlap)
            elif (overlap.get('same_problem', False) and
                  not overlap.get('has_problematic_overlap', False) and
                  overlap.get('overlap_type') == 'none'):
                synergy_score += 0.15  # Bonus for complementary approaches
                synergy_count += 1

            # Penalize problematic overlap
            if overlap.get('has_problematic_overlap', False):
                synergy_score -= 0.2  # Penalty for conflicts
                conflict_count += 1

        # Normalize by number of comparisons (more synergies = better)
        if current_toolkit:
            normalized_score = synergy_score / len(current_toolkit)
            return max(-0.5, min(1.0, normalized_score))  # Clamp to [-0.5, 1.0]

        return 0.0

    def _score_candidates(
        self,
        candidates: pd.DataFrame,
        current_toolkit: List[int],
        profile: Dict,
        fitness_scores: pd.Series
    ) -> pd.Series:
        """
        Score candidates based on multiple factors:
        - Fitness for context
        - Compatibility with toolkit (avg + min)
        - Category diversity
        - Synergy bonus

        Args:
            candidates: DataFrame of candidate methods
            current_toolkit: List of df indices in toolkit
            profile: Context profile
            fitness_scores: Pre-computed fitness scores

        Returns:
            Series with composite scores
        """
        scores = pd.Series(index=candidates.index, dtype=float)

        current_categories = self.methods.loc[current_toolkit, 'category']

        for idx in candidates.index:
            candidate = candidates.loc[idx]

            # 1. Base fitness score (0-100)
            fitness = fitness_scores.loc[idx] if idx in fitness_scores.index else 50.0

            # 2. Compatibility scores
            compatibilities = []
            for tk_idx in current_toolkit:
                tk_name = self.methods.loc[tk_idx, 'name']
                cand_name = candidate['name']
                compat = self.compatibility.get_compatibility(tk_name, cand_name)
                compatibilities.append(compat)

            avg_compat = np.mean(compatibilities) if compatibilities else 1.0
            min_compat = np.min(compatibilities) if compatibilities else 1.0

            # 3. Category diversity bonus
            category = candidate['category']
            diversity_bonus = 0.1 if category not in current_categories.values else 0.0

            # 4. Synergy bonus - using ACTUAL pairwise overlap_analysis
            synergy_bonus = self._calculate_pairwise_synergy(
                candidate['name'],
                current_toolkit
            )

            # Composite score (weighted combination)
            scores[idx] = (
                fitness * 0.4 +                      # Fitness: 40%
                avg_compat * 100 * 0.25 +            # Avg compatibility: 25%
                min_compat * 100 * 0.20 +            # Min compatibility (no weak links): 20%
                diversity_bonus * 100 +              # Category diversity: up to 10%
                synergy_bonus * 100 * 0.15           # Pairwise synergy from overlap_analysis: up to 15%
            )

        return scores

    def build_compatible_toolkit(
        self,
        context_key: str,
        use_seeds: Optional[List[str]] = None
    ) -> Dict:
        """
        Build toolkit with full compatibility checking.

        Args:
            context_key: Which context profile to use
            use_seeds: Optional list of method names that must be included

        Returns:
            Dict with toolkit and metadata
        """
        if context_key not in self.context_profiles:
            raise ValueError(f"Unknown context: {context_key}")

        profile = self.context_profiles[context_key]

        logger.info(f"\n{'='*80}")
        logger.info(f"Building toolkit for: {profile['name']}")
        logger.info(f"  Description: {profile['description']}")
        logger.info(f"  Target size: {profile['toolkit_size']}")
        logger.info(f"  Min compatibility: {profile['min_compatibility']}")
        logger.info(f"{'='*80}\n")

        # Calculate context fitness scores
        fitness_scores = self._calculate_context_fitness(profile)

        # Initialize toolkit with seeds or best method from preferred categories
        toolkit = []

        if use_seeds:
            # Convert seed names to df indices
            for seed_name in use_seeds:
                if seed_name in self.method_name_to_df_idx:
                    df_idx = self.method_name_to_df_idx[seed_name]
                    toolkit.append(df_idx)
                    logger.info(f"  Seed: {seed_name}")
                else:
                    logger.warning(f"  Seed method not found: {seed_name}")
        else:
            # Start with highest fitness method from preferred category
            preferred_categories = profile.get('preferred_categories', [])

            if preferred_categories:
                preferred_methods = self.methods[
                    self.methods['category'].isin(preferred_categories)
                ].copy()

                if not preferred_methods.empty:
                    # Add fitness scores to dataframe
                    preferred_methods['fitness'] = fitness_scores

                    # Get best method
                    best_idx = preferred_methods['fitness'].idxmax()
                    toolkit.append(best_idx)

                    logger.info(f"  Starting with: {self.methods.loc[best_idx, 'name']} "
                               f"(fitness: {fitness_scores.loc[best_idx]:.1f}, "
                               f"category: {self.methods.loc[best_idx, 'category']})")

            # Fallback: use highest fitness method overall
            if not toolkit:
                best_idx = fitness_scores.idxmax()
                toolkit.append(best_idx)
                logger.info(f"  Starting with: {self.methods.loc[best_idx, 'name']} "
                           f"(fitness: {fitness_scores.loc[best_idx]:.1f})")

        # Greedy compatible expansion
        target_size = profile['toolkit_size']
        iteration = 1

        while len(toolkit) < target_size:
            logger.info(f"\n  Iteration {iteration}: Finding compatible candidates...")

            # Get compatible candidates
            candidates = self._get_compatible_candidates(
                toolkit,
                profile,
                fitness_scores
            )

            if candidates.empty:
                logger.warning(f"    No more compatible candidates found! Stopping at {len(toolkit)} methods.")
                break

            logger.info(f"    Compatible candidates: {len(candidates)}")

            # Score candidates
            candidate_scores = self._score_candidates(
                candidates,
                toolkit,
                profile,
                fitness_scores
            )

            # Add best candidate
            best_candidate_idx = candidate_scores.idxmax()
            best_score = candidate_scores.loc[best_candidate_idx]
            best_method = self.methods.loc[best_candidate_idx]

            toolkit.append(best_candidate_idx)

            logger.info(f"    Selected: {best_method['name']}")
            logger.info(f"      Score: {best_score:.1f}")
            logger.info(f"      Category: {best_method['category']}")
            logger.info(f"      Fitness: {fitness_scores.loc[best_candidate_idx]:.1f}")

            iteration += 1

        # Phase 2: Ensure all required categories are covered
        if self.required_categories:
            logger.info(f"\n  Phase 2: Ensuring category coverage...")
            covered_categories = set(self.methods.loc[toolkit, 'category'].unique())
            missing_categories = set(self.required_categories) - covered_categories

            if missing_categories:
                logger.info(f"    Missing categories: {list(missing_categories)}")

                for missing_cat in missing_categories:
                    logger.info(f"\n    Adding method from category: {missing_cat}")

                    # Find compatible methods from this category
                    cat_methods = self.methods[self.methods['category'] == missing_cat].copy()

                    if cat_methods.empty:
                        logger.warning(f"      No methods available in category: {missing_cat}")
                        continue

                    # Filter for compatibility with existing toolkit
                    compatible_from_cat = []
                    for idx in cat_methods.index:
                        if idx in toolkit:
                            continue

                        # Check compatibility with all toolkit members
                        is_compatible = True
                        for tk_idx in toolkit:
                            method_name = self.methods.loc[idx, 'name']
                            tk_method_name = self.methods.loc[tk_idx, 'name']
                            compat = self.compatibility.get_compatibility(method_name, tk_method_name)

                            if compat < profile['min_compatibility']:
                                is_compatible = False
                                break

                        if is_compatible:
                            compatible_from_cat.append(idx)

                    if not compatible_from_cat:
                        logger.warning(f"      No compatible methods found for category: {missing_cat}")
                        continue

                    # Score candidates from this category
                    cat_candidates = self.methods.loc[compatible_from_cat]
                    candidate_scores = self._score_candidates(
                        cat_candidates,
                        toolkit,
                        profile,
                        fitness_scores
                    )

                    # Add best from this category
                    best_idx = candidate_scores.idxmax()
                    best_method = self.methods.loc[best_idx]
                    toolkit.append(best_idx)

                    logger.info(f"      Added: {best_method['name']}")
                    logger.info(f"      Fitness: {fitness_scores.loc[best_idx]:.1f}")
            else:
                logger.info(f"    ✓ All {len(self.required_categories)} categories already covered!")

        # Create result
        result = self._create_toolkit_result(toolkit, context_key, fitness_scores)

        logger.info(f"\n{'='*80}")
        logger.info(f"Toolkit complete: {len(toolkit)} methods")
        logger.info(f"  Categories covered: {len(set(self.methods.loc[toolkit, 'category'].unique()))}/{len(self.required_categories)}")
        logger.info(f"{'='*80}\n")

        return result

    def _create_toolkit_result(
        self,
        toolkit: List[int],
        context_key: str,
        fitness_scores: pd.Series
    ) -> Dict:
        """
        Create formatted toolkit result with metadata.

        Args:
            toolkit: List of df indices (in selection order)
            context_key: Context profile key
            fitness_scores: Fitness scores

        Returns:
            Dict with complete toolkit information
        """
        profile = self.context_profiles[context_key]

        methods = []
        categories = []
        avg_compatibility_scores = []

        # Track categories added so far (for diversity calculation)
        categories_so_far = set()

        for position, df_idx in enumerate(toolkit):
            method = self.methods.loc[df_idx]

            # Calculate average compatibility with other toolkit members
            compatibilities = []
            for other_idx in toolkit:
                if other_idx != df_idx:
                    compat = self.compatibility.get_compatibility(
                        method['name'],
                        self.methods.loc[other_idx, 'name']
                    )
                    compatibilities.append(compat)

            avg_compat = np.mean(compatibilities) if compatibilities else 1.0
            min_compat = np.min(compatibilities) if compatibilities else 1.0

            # Calculate diversity score AT TIME OF SELECTION
            # (was this category already present when method was selected?)
            category = method['category']
            was_new_category = category not in categories_so_far
            diversity_score = 10.0 if was_new_category else 0.0

            # Calculate synergy score AT TIME OF SELECTION
            # (what was the synergy with methods already in toolkit?)
            methods_before = toolkit[:position]  # Methods added before this one
            if methods_before:
                synergy_score = self._calculate_pairwise_synergy(
                    method['name'],
                    methods_before
                ) * 100  # Scale to 0-100
            else:
                synergy_score = 0.0  # First method has no synergy yet

            # Update categories tracker
            categories_so_far.add(category)

            method_data = {
                'name': method['name'],
                'category': method['category'],
                'fitness_score': float(fitness_scores.loc[df_idx]),
                'avg_compatibility': float(avg_compat),
                'min_compatibility': float(min_compat),
                'diversity_score': float(diversity_score),  # NEW: actual diversity
                'synergy_score': float(synergy_score),      # NEW: actual synergy
                'impact_potential': float(method.get('impact_potential', 0)),
                'implementation_difficulty': float(method.get('implementation_difficulty', 0)),
                'time_to_value': float(method.get('time_to_value', 0)),
                'ease_adoption': float(method.get('ease_adoption', 0))
            }

            methods.append(method_data)
            categories.append(method['category'])
            avg_compatibility_scores.append(avg_compat)

        # Calculate statistics
        category_counts = pd.Series(categories).value_counts().to_dict()

        result = {
            'context': context_key,
            'context_name': profile['name'],
            'context_description': profile['description'],
            'methods': methods,
            'method_names': [m['name'] for m in methods],
            'method_indices': [int(idx) for idx in toolkit],  # Convert to native int
            'size': len(toolkit),
            'statistics': {
                'avg_fitness': float(np.mean([m['fitness_score'] for m in methods])),
                'avg_compatibility': float(np.mean(avg_compatibility_scores)),
                'min_compatibility': float(np.min(avg_compatibility_scores)),
                'avg_impact': float(np.mean([m['impact_potential'] for m in methods])),
                'avg_difficulty': float(np.mean([m['implementation_difficulty'] for m in methods])),
                'avg_time_to_value': float(np.mean([m['time_to_value'] for m in methods])),
                'categories_covered': len(category_counts),
                'category_distribution': {k: int(v) for k, v in category_counts.items()}  # Convert values to int
            }
        }

        return result

    def compare_contexts(self, contexts_to_compare: Optional[List[str]] = None) -> Dict:
        """
        Build and compare toolkits across multiple contexts.

        Returns comparative analysis with:
        - Toolkits for each context
        - Overlap analysis
        - Transition paths
        - Dimension coverage comparison

        Args:
            contexts_to_compare: List of context keys, or None for all

        Returns:
            Dict with comparison results
        """
        if contexts_to_compare is None:
            contexts_to_compare = list(self.context_profiles.keys())

        logger.info(f"\n{'='*80}")
        logger.info(f"MULTI-CONTEXT COMPARISON")
        logger.info(f"  Comparing {len(contexts_to_compare)} contexts")
        logger.info(f"{'='*80}\n")

        results = {}

        # Build optimal toolkit for each context
        for context in contexts_to_compare:
            toolkit = self.build_compatible_toolkit(context)
            results[context] = toolkit

        # Analyze overlaps
        overlap_matrix = self._calculate_overlap_matrix(results)

        # Find unique methods per context
        unique_methods = self._identify_unique_methods(results)

        # Analyze transitions
        transition_analysis = self._analyze_transitions(results)

        # Compare dimension coverage
        dimension_coverage = self._compare_dimension_coverage(results)

        comparison = {
            'toolkits': results,
            'overlap_matrix': overlap_matrix,
            'unique_methods': unique_methods,
            'transition_analysis': transition_analysis,
            'dimension_coverage': dimension_coverage,
            'summary': {
                'contexts_compared': len(contexts_to_compare),
                'total_unique_methods': len(set(
                    method for tk in results.values()
                    for method in tk['method_names']
                ))
            }
        }

        return comparison

    def _calculate_overlap_matrix(self, toolkits: Dict) -> Dict:
        """Calculate pairwise overlap between toolkits."""
        contexts = list(toolkits.keys())
        overlap = {}

        for i, ctx_a in enumerate(contexts):
            overlap[ctx_a] = {}
            methods_a = set(toolkits[ctx_a]['method_names'])

            for ctx_b in contexts:
                methods_b = set(toolkits[ctx_b]['method_names'])

                shared = methods_a & methods_b
                total = methods_a | methods_b

                overlap[ctx_a][ctx_b] = {
                    'shared_count': len(shared),
                    'shared_methods': list(shared),
                    'jaccard_similarity': len(shared) / len(total) if total else 0
                }

        return overlap

    def _identify_unique_methods(self, toolkits: Dict) -> Dict:
        """Identify methods unique to each context."""
        unique = {}

        for context, toolkit in toolkits.items():
            methods_this = set(toolkit['method_names'])

            # Find methods in other toolkits
            methods_others = set()
            for other_context, other_toolkit in toolkits.items():
                if other_context != context:
                    methods_others.update(other_toolkit['method_names'])

            # Unique to this context
            unique_to_context = methods_this - methods_others

            unique[context] = {
                'unique_methods': list(unique_to_context),
                'count': len(unique_to_context),
                'percentage': len(unique_to_context) / len(methods_this) * 100 if methods_this else 0
            }

        return unique

    def _analyze_transitions(self, toolkits: Dict) -> Dict:
        """Analyze how to transition between toolkits as organization evolves."""

        transitions = {}

        # Define natural progression paths
        progression_paths = [
            ['startup_mvp', 'startup_scaling', 'enterprise_transformation'],
            ['startup_mvp', 'hardware_product'],
            ['startup_scaling', 'regulated_industry']
        ]

        for path in progression_paths:
            path_key = ' → '.join(path)
            transitions[path_key] = {'stages': []}

            for i in range(len(path) - 1):
                from_context = path[i]
                to_context = path[i + 1]

                if from_context in toolkits and to_context in toolkits:
                    from_methods = set(toolkits[from_context]['method_names'])
                    to_methods = set(toolkits[to_context]['method_names'])

                    keep = from_methods & to_methods
                    add = to_methods - from_methods
                    remove = from_methods - to_methods

                    stage = {
                        'from': from_context,
                        'from_name': toolkits[from_context]['context_name'],
                        'to': to_context,
                        'to_name': toolkits[to_context]['context_name'],
                        'keep': list(keep),
                        'keep_count': len(keep),
                        'add': list(add),
                        'add_count': len(add),
                        'remove': list(remove),
                        'remove_count': len(remove),
                        'change_magnitude': len(add) + len(remove),
                        'continuity_percentage': len(keep) / len(from_methods) * 100 if from_methods else 0
                    }

                    transitions[path_key]['stages'].append(stage)

        return transitions

    def _compare_dimension_coverage(self, toolkits: Dict) -> Dict:
        """Compare how different toolkits cover the 12D space."""

        dimension_columns = [
            'scope', 'temporality', 'impact_potential', 'implementation_difficulty',
            'change_management_difficulty', 'resources_required', 'time_to_value',
            'ease_adoption', 'applicability', 'people_focus', 'process_focus',
            'purpose_orientation'
        ]

        coverage = {}

        for context, toolkit in toolkits.items():
            method_names = toolkit['method_names']

            # Get subset of methods dataframe
            toolkit_methods = self.methods[self.methods['name'].isin(method_names)]

            # Calculate average values for each dimension
            dim_averages = {}
            for dim in dimension_columns:
                if dim in toolkit_methods.columns:
                    dim_averages[dim] = float(toolkit_methods[dim].mean())

            coverage[context] = {
                'context_name': toolkit['context_name'],
                'dimension_averages': dim_averages,
                'balance_score': self._calculate_balance_score(dim_averages)
            }

        return coverage

    def _calculate_balance_score(self, dimension_averages: Dict) -> float:
        """Calculate how balanced a toolkit is across dimensions."""
        values = list(dimension_averages.values())
        if not values:
            return 0.0

        # Lower variance = more balanced
        variance = np.var(values)
        # Normalize to 0-100 (lower variance = higher score)
        balance_score = max(0, 100 - variance / 10)

        return float(balance_score)

    def generate_visualization_data(self, toolkit_result: Dict) -> Dict:
        """
        Generate data for multiple visualization types.

        Args:
            toolkit_result: Result from build_compatible_toolkit()

        Returns:
            Dict with visualization data for different chart types
        """
        viz_data = {
            'network_graph': self._generate_network_data(toolkit_result),
            'spider_chart': self._generate_spider_data(toolkit_result),
            'category_distribution': self._generate_category_chart_data(toolkit_result),
            'implementation_timeline': self._generate_timeline_data(toolkit_result)
        }

        return viz_data

    def _generate_network_data(self, toolkit_result: Dict) -> Dict:
        """Generate force-directed graph data showing method relationships."""

        methods = toolkit_result['methods']
        nodes = []
        edges = []

        # Create nodes
        for method in methods:
            nodes.append({
                'id': method['name'],
                'label': method['name'],
                'category': method['category'],
                'fitness': method['fitness_score'],
                'impact': method['impact_potential'],
                'difficulty': method['implementation_difficulty'],
                'size': method['impact_potential']  # Node size based on impact
            })

        # Create edges based on compatibility
        for i, method_a in enumerate(methods):
            for j in range(i + 1, len(methods)):
                method_b = methods[j]

                compatibility = self.compatibility.get_compatibility(
                    method_a['name'],
                    method_b['name']
                )

                # Only show meaningful connections
                if compatibility > 0.5:
                    edges.append({
                        'source': method_a['name'],
                        'target': method_b['name'],
                        'weight': compatibility,
                        'strength': 'strong' if compatibility > 0.8 else 'medium'
                    })

        return {'nodes': nodes, 'edges': edges}

    def _generate_spider_data(self, toolkit_result: Dict) -> Dict:
        """Generate radar/spider chart data for 12D dimensions."""

        methods = toolkit_result['methods']
        method_names = [m['name'] for m in methods]

        # Get methods from dataframe
        toolkit_methods = self.methods[self.methods['name'].isin(method_names)]

        # Calculate averages for each dimension
        dimensions = [
            'scope', 'temporality', 'impact_potential', 'implementation_difficulty',
            'change_management_difficulty', 'resources_required', 'time_to_value',
            'ease_adoption', 'applicability', 'people_focus', 'process_focus',
            'purpose_orientation'
        ]

        averages = {}
        for dim in dimensions:
            if dim in toolkit_methods.columns:
                averages[dim] = float(toolkit_methods[dim].mean())

        return {
            'dimensions': dimensions,
            'values': averages,
            'context': toolkit_result['context_name']
        }

    def _generate_category_chart_data(self, toolkit_result: Dict) -> Dict:
        """Generate data for category distribution charts."""

        category_dist = toolkit_result['statistics']['category_distribution']

        return {
            'categories': list(category_dist.keys()),
            'counts': list(category_dist.values()),
            'total': toolkit_result['size']
        }

    def _generate_timeline_data(self, toolkit_result: Dict) -> Dict:
        """Generate implementation timeline/sequencing data."""

        methods = toolkit_result['methods']

        # Sort by implementation difficulty (easier first) and time to value (faster first)
        sorted_methods = sorted(
            methods,
            key=lambda m: (m['implementation_difficulty'], -m['time_to_value'])
        )

        # Create waves based on difficulty
        waves = {
            'quick_wins': [],
            'core_implementations': [],
            'advanced_practices': []
        }

        for method in sorted_methods:
            if method['implementation_difficulty'] < 40 and method['time_to_value'] > 60:
                waves['quick_wins'].append(method['name'])
            elif method['implementation_difficulty'] < 65:
                waves['core_implementations'].append(method['name'])
            else:
                waves['advanced_practices'].append(method['name'])

        return {
            'waves': waves,
            'sequenced_methods': [m['name'] for m in sorted_methods],
            'timeline_metadata': {
                'quick_wins_period': '0-3 months',
                'core_period': '3-9 months',
                'advanced_period': '9+ months'
            }
        }


def main():
    """Main orchestration for toolkit building."""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description='Enhanced Situational Toolkit Generator - Build context-optimized method toolkits'
    )
    parser.add_argument(
        '--scores',
        default='results/method_scores_12d_deduplicated.json',
        help='12D scores JSON file'
    )
    parser.add_argument(
        '--compatibility',
        default='results/compatibility_checkpoint.pkl',
        help='Compatibility matrix pickle file'
    )
    parser.add_argument(
        '--context',
        choices=['startup_mvp', 'startup_scaling', 'enterprise_transformation',
                 'regulated_industry', 'hardware_product'],
        help='Single context to build toolkit for'
    )
    parser.add_argument(
        '--compare-all',
        action='store_true',
        help='Compare toolkits across all contexts'
    )
    parser.add_argument(
        '--seeds',
        nargs='+',
        help='Method names to use as seeds (must be included in toolkit)'
    )
    parser.add_argument(
        '--output-dir',
        default='results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--output-prefix',
        default='toolkit',
        help='Prefix for output files'
    )

    args = parser.parse_args()

    # Load data
    logger.info(f"Loading 12D scores from {args.scores}...")
    with open(args.scores, 'r') as f:
        scores_data = json.load(f)

    methods_df = pd.DataFrame(scores_data['methods'])
    logger.info(f"  Loaded {len(methods_df)} methods")

    # Load semantic cluster categories from combined_clusters.json
    clusters_file = Path('results_semantic_clustering_combined/combined_clusters.json')
    category_definitions = None

    if clusters_file.exists():
        logger.info("Loading semantic cluster categories from combined_clusters.json...")
        with open(clusters_file, 'r') as f:
            clusters_data = json.load(f)

        # Build category definitions from cluster data
        category_definitions = {
            'categories': [
                {'id': cluster_id, 'name': cluster_data.get('name', cluster_id)}
                for cluster_id, cluster_data in clusters_data['clusters'].items()
                if cluster_id != 'U'  # Exclude unclustered
            ]
        }
        logger.info(f"  Loaded {len(category_definitions['categories'])} cluster categories")

        # Build name -> cluster mapping (using name because indices differ between files)
        method_name_to_category = {}
        for cluster_id, cluster_data in clusters_data['clusters'].items():
            for method_info in cluster_data['methods']:
                method_name = method_info['Method'].strip().lower()
                method_name_to_category[method_name] = cluster_id

        # Apply to DataFrame (using 'name' column from methods)
        if 'name' in methods_df.columns:
            methods_df['category'] = methods_df['name'].apply(
                lambda name: method_name_to_category.get(name.strip().lower() if isinstance(name, str) else '', 'U')
            )
            logger.info(f"  Applied semantic cluster categories to {len(methods_df)} methods")

            # Log category distribution
            cat_dist = methods_df['category'].value_counts()
            logger.info(f"  Category distribution:")
            for cat, count in cat_dist.head(10).items():
                logger.info(f"    {cat}: {count}")
            if len(cat_dist) > 10:
                logger.info(f"    ... and {len(cat_dist) - 10} more categories")
        else:
            logger.warning("  'name' column not found in methods_df, cannot apply categories")
    else:
        logger.warning(f"Semantic clusters file not found: {clusters_file}")
        logger.warning("Methods will be auto-categorized based on P×P×P dimensions")

    # Load compatibility matrix
    compat_matrix = load_compatibility_matrix(args.compatibility)

    # Initialize builder with category definitions
    builder = SituationalToolkitBuilder(methods_df, compat_matrix, category_definitions)

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # ========================================================================
    # Build Toolkits
    # ========================================================================

    if args.compare_all:
        # Multi-context comparison
        logger.info("\n" + "="*80)
        logger.info("RUNNING MULTI-CONTEXT COMPARISON")
        logger.info("="*80 + "\n")

        comparison = builder.compare_contexts()
        results['comparison'] = comparison

        # Save comparison results
        output_file = output_dir / f'{args.output_prefix}_comparison.json'
        with open(output_file, 'w') as f:
            json.dump(comparison, f, indent=2)

        logger.info(f"\n✓ Comparison saved to: {output_file}")

        # Print summary
        print_comparison_summary(comparison)

    elif args.context:
        # Single context toolkit
        logger.info("\n" + "="*80)
        logger.info(f"BUILDING TOOLKIT FOR: {args.context}")
        logger.info("="*80 + "\n")

        toolkit = builder.build_compatible_toolkit(args.context, use_seeds=args.seeds)
        results['toolkit'] = toolkit

        # Generate visualization data
        viz_data = builder.generate_visualization_data(toolkit)
        results['visualization'] = viz_data

        # Save toolkit
        output_file = output_dir / f'{args.output_prefix}_{args.context}.json'
        with open(output_file, 'w') as f:
            json.dump({'toolkit': toolkit, 'visualization': viz_data}, f, indent=2)

        logger.info(f"\n✓ Toolkit saved to: {output_file}")

        # Print summary
        print_toolkit_summary(toolkit)

    else:
        # Default: Build toolkits for all contexts
        logger.info("\n" + "="*80)
        logger.info("BUILDING TOOLKITS FOR ALL CONTEXTS")
        logger.info("="*80 + "\n")

        all_toolkits = {}
        for context_key in builder.context_profiles.keys():
            toolkit = builder.build_compatible_toolkit(context_key)
            all_toolkits[context_key] = toolkit

        results['all_toolkits'] = all_toolkits

        # Save all toolkits
        output_file = output_dir / f'{args.output_prefix}_all_contexts.json'
        with open(output_file, 'w') as f:
            json.dump(all_toolkits, f, indent=2)

        logger.info(f"\n✓ All toolkits saved to: {output_file}")

        # Print summary for each
        for context_key, toolkit in all_toolkits.items():
            print("\n" + "="*80)
            print_toolkit_summary(toolkit)


def print_toolkit_summary(toolkit: Dict):
    """Print executive summary of a toolkit."""
    print(f"\n{'='*80}")
    print(f"TOOLKIT: {toolkit['context_name']}")
    print(f"{'='*80}")
    print(f"\n{toolkit['context_description']}")

    stats = toolkit['statistics']

    print(f"\n📊 STATISTICS")
    print(f"  Methods: {toolkit['size']}")
    print(f"  Categories: {stats['categories_covered']}")
    print(f"  Avg Fitness: {stats['avg_fitness']:.1f}")
    print(f"  Avg Compatibility: {stats['avg_compatibility']:.3f}")
    print(f"  Min Compatibility: {stats['min_compatibility']:.3f}")
    print(f"  Avg Impact: {stats['avg_impact']:.1f}")
    print(f"  Avg Difficulty: {stats['avg_difficulty']:.1f}")
    print(f"  Avg Time-to-Value: {stats['avg_time_to_value']:.1f}")

    print(f"\n📦 METHODS ({toolkit['size']})")
    for i, method in enumerate(toolkit['methods'], 1):
        print(f"  {i:2d}. {method['name']:50s} [{method['category']}]")
        print(f"      Fitness: {method['fitness_score']:5.1f} | "
              f"Compat: {method['avg_compatibility']:.3f} | "
              f"Impact: {method['impact_potential']:5.1f}")

    print(f"\n🏷️  CATEGORY DISTRIBUTION")
    for category, count in sorted(stats['category_distribution'].items(),
                                  key=lambda x: x[1], reverse=True):
        print(f"  {category:30s}: {count}")


def print_comparison_summary(comparison: Dict):
    """Print summary of multi-context comparison."""
    print(f"\n{'='*80}")
    print(f"MULTI-CONTEXT COMPARISON SUMMARY")
    print(f"{'='*80}")

    summary = comparison['summary']
    print(f"\nContexts compared: {summary['contexts_compared']}")
    print(f"Total unique methods: {summary['total_unique_methods']}")

    print(f"\n📊 TOOLKIT SIZES")
    for context, toolkit in comparison['toolkits'].items():
        print(f"  {toolkit['context_name']:45s}: {toolkit['size']} methods")

    print(f"\n🔄 TRANSITION PATHS")
    for path_name, path_data in comparison['transition_analysis'].items():
        print(f"\n  {path_name}")
        for stage in path_data['stages']:
            print(f"    {stage['from_name']} → {stage['to_name']}")
            print(f"      Keep: {stage['keep_count']} | Add: {stage['add_count']} | Remove: {stage['remove_count']}")
            print(f"      Continuity: {stage['continuity_percentage']:.1f}%")

    print(f"\n🎯 UNIQUE METHODS PER CONTEXT")
    for context, unique_data in comparison['unique_methods'].items():
        if unique_data['count'] > 0:
            toolkit_name = comparison['toolkits'][context]['context_name']
            print(f"  {toolkit_name}: {unique_data['count']} unique ({unique_data['percentage']:.1f}%)")
            for method in unique_data['unique_methods'][:3]:  # Show first 3
                print(f"    - {method}")


if __name__ == "__main__":
    main()
