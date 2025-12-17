"""
Smart sampling strategy to reduce N² comparison problem.
Instead of 319,600 comparisons for 800 methods, we sample ~3-5% strategically.
"""
import logging
import random
from typing import List, Tuple, Set
import numpy as np

from src.data import Method

logger = logging.getLogger(__name__)


class SmartSampler:
    """
    Implements smart sampling to reduce pairwise comparisons.

    Strategy:
    1. High-similarity pairs (cosine > threshold) - likely duplicates
    2. Same-source sampling - methods from same source often related
    3. Cross-source sampling - important for cross-domain patterns
    4. Random baseline - ensures coverage of edge cases
    """

    def __init__(self, config: dict):
        self.config = config['analysis']['sampling']
        self.high_sim_threshold = self.config['high_similarity_threshold']
        self.same_source_rate = self.config['same_source_sample_rate']
        self.cross_source_rate = self.config['cross_source_sample_rate']
        self.random_baseline = self.config['random_baseline_count']
        self.medium_sim_max = self.config.get('medium_similarity_max_pairs', 3000)

    def sample_for_duplicate_detection(
        self,
        methods: List[Method],
        similarity_matrix: np.ndarray
    ) -> List[Tuple[Method, Method, float]]:
        """
        Sample method pairs for duplicate detection.
        Returns: List of (method1, method2, similarity_score)
        """
        logger.info("Sampling pairs for duplicate detection...")

        sampled_pairs = set()  # Use set to avoid duplicates
        n = len(methods)
        total_possible = n * (n - 1) // 2

        # 1. High-similarity pairs (most likely duplicates)
        high_sim_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                similarity = similarity_matrix[i, j]
                if similarity >= self.high_sim_threshold:
                    pair = self._make_pair_key(methods[i], methods[j])
                    if pair not in sampled_pairs:
                        sampled_pairs.add(pair)
                        high_sim_pairs.append((methods[i], methods[j], float(similarity)))

        logger.info(f"High-similarity pairs (>={self.high_sim_threshold}): {len(high_sim_pairs)}")

        # 2. Same-source sampling
        same_source_pairs = self._sample_same_source(methods, similarity_matrix, sampled_pairs)
        logger.info(f"Same-source sampled pairs: {len(same_source_pairs)}")

        # 3. Random baseline
        random_pairs = self._sample_random(methods, similarity_matrix, sampled_pairs)
        logger.info(f"Random baseline pairs: {len(random_pairs)}")

        # Combine all samples
        all_pairs = high_sim_pairs + same_source_pairs + random_pairs

        logger.info(f"Total sampled pairs: {len(all_pairs)} ({100 * len(all_pairs) / total_possible:.2f}% of {total_possible})")

        return all_pairs

    def sample_for_compatibility_analysis(
        self,
        methods: List[Method],
        similarity_matrix: np.ndarray,
        duplicate_indices: Set[Tuple[int, int]]
    ) -> List[Tuple[Method, Method]]:
        """
        Sample method pairs for compatibility analysis.
        Excludes duplicates and focuses on potentially complementary methods.
        """
        logger.info("Sampling pairs for compatibility analysis...")

        sampled_pairs = set()
        n = len(methods)

        # 1. Medium similarity pairs (potentially complementary, not duplicates)
        # Look for pairs with similarity between 0.3 and 0.7
        medium_sim_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                # Skip if identified as duplicate
                if (methods[i].index, methods[j].index) in duplicate_indices:
                    continue
                if (methods[j].index, methods[i].index) in duplicate_indices:
                    continue

                similarity = similarity_matrix[i, j]
                if 0.3 <= similarity < 0.7:
                    pair = self._make_pair_key(methods[i], methods[j])
                    if pair not in sampled_pairs:
                        sampled_pairs.add(pair)
                        medium_sim_pairs.append((methods[i], methods[j]))

        # Sample from medium similarity if too many
        if len(medium_sim_pairs) > self.medium_sim_max:
            medium_sim_pairs = random.sample(medium_sim_pairs, self.medium_sim_max)

        logger.info(f"Medium-similarity pairs (0.3-0.7): {len(medium_sim_pairs)}")

        # 2. Cross-source sampling (often complementary)
        cross_source_pairs = self._sample_cross_source(methods, similarity_matrix, sampled_pairs, duplicate_indices)
        logger.info(f"Cross-source sampled pairs: {len(cross_source_pairs)}")

        # 3. Random diverse pairs
        diverse_pairs = self._sample_diverse(methods, similarity_matrix, sampled_pairs, duplicate_indices, count=500)
        logger.info(f"Diverse random pairs: {len(diverse_pairs)}")

        all_pairs = medium_sim_pairs + cross_source_pairs + diverse_pairs

        logger.info(f"Total compatibility pairs: {len(all_pairs)}")
        return all_pairs

    def _sample_same_source(
        self,
        methods: List[Method],
        similarity_matrix: np.ndarray,
        existing_pairs: Set
    ) -> List[Tuple[Method, Method, float]]:
        """Sample pairs from the same source."""
        # Build index mapping for O(1) lookups
        method_idx_map = {m: i for i, m in enumerate(methods)}

        # Group by source
        by_source = {}
        for method in methods:
            if method.source not in by_source:
                by_source[method.source] = []
            by_source[method.source].append(method)

        pairs = []
        for source, source_methods in by_source.items():
            n = len(source_methods)
            total_pairs = n * (n - 1) // 2
            sample_count = int(total_pairs * self.same_source_rate)

            if sample_count == 0 and total_pairs > 0:
                sample_count = min(5, total_pairs)  # At least a few

            # Sample pairs
            for _ in range(sample_count):
                i, j = random.sample(range(n), 2)
                method_i = source_methods[i]
                method_j = source_methods[j]

                pair_key = self._make_pair_key(method_i, method_j)
                if pair_key not in existing_pairs:
                    existing_pairs.add(pair_key)

                    # Get similarity using index map
                    idx_i = method_idx_map[method_i]
                    idx_j = method_idx_map[method_j]
                    similarity = float(similarity_matrix[idx_i, idx_j])

                    pairs.append((method_i, method_j, similarity))

        return pairs

    def _sample_cross_source(
        self,
        methods: List[Method],
        similarity_matrix: np.ndarray,
        existing_pairs: Set,
        duplicate_indices: Set[Tuple[int, int]]
    ) -> List[Tuple[Method, Method]]:
        """Sample pairs from different sources."""
        # Group by source
        by_source = {}
        for method in methods:
            if method.source not in by_source:
                by_source[method.source] = []
            by_source[method.source].append(method)

        sources = list(by_source.keys())
        pairs = []

        # Calculate target sample count
        total_cross_pairs = sum(
            len(by_source[s1]) * len(by_source[s2])
            for i, s1 in enumerate(sources)
            for s2 in sources[i + 1:]
        )
        sample_count = int(total_cross_pairs * self.cross_source_rate)

        attempts = 0
        max_attempts = sample_count * 10  # Prevent infinite loop

        while len(pairs) < sample_count and attempts < max_attempts:
            attempts += 1

            # Pick two different sources
            if len(sources) < 2:
                break
            s1, s2 = random.sample(sources, 2)

            # Pick random method from each
            method1 = random.choice(by_source[s1])
            method2 = random.choice(by_source[s2])

            # Skip if duplicate
            if (method1.index, method2.index) in duplicate_indices:
                continue
            if (method2.index, method1.index) in duplicate_indices:
                continue

            pair_key = self._make_pair_key(method1, method2)
            if pair_key not in existing_pairs:
                existing_pairs.add(pair_key)
                pairs.append((method1, method2))

        return pairs

    def _sample_random(
        self,
        methods: List[Method],
        similarity_matrix: np.ndarray,
        existing_pairs: Set
    ) -> List[Tuple[Method, Method, float]]:
        """Sample random pairs for baseline coverage."""
        pairs = []
        n = len(methods)
        attempts = 0
        max_attempts = self.random_baseline * 10

        while len(pairs) < self.random_baseline and attempts < max_attempts:
            attempts += 1

            i, j = random.sample(range(n), 2)
            if i > j:
                i, j = j, i

            method_i = methods[i]
            method_j = methods[j]

            pair_key = self._make_pair_key(method_i, method_j)
            if pair_key not in existing_pairs:
                existing_pairs.add(pair_key)
                similarity = float(similarity_matrix[i, j])
                pairs.append((method_i, method_j, similarity))

        return pairs

    def _sample_diverse(
        self,
        methods: List[Method],
        similarity_matrix: np.ndarray,
        existing_pairs: Set,
        duplicate_indices: Set[Tuple[int, int]],
        count: int
    ) -> List[Tuple[Method, Method]]:
        """Sample diverse random pairs (low similarity)."""
        pairs = []
        n = len(methods)
        attempts = 0
        max_attempts = count * 10

        while len(pairs) < count and attempts < max_attempts:
            attempts += 1

            i, j = random.sample(range(n), 2)
            method_i = methods[i]
            method_j = methods[j]

            # Skip duplicates
            if (method_i.index, method_j.index) in duplicate_indices:
                continue
            if (method_j.index, method_i.index) in duplicate_indices:
                continue

            # Prefer low similarity (diverse)
            similarity = similarity_matrix[i, j]
            if similarity > 0.6:  # Skip high similarity
                continue

            pair_key = self._make_pair_key(method_i, method_j)
            if pair_key not in existing_pairs:
                existing_pairs.add(pair_key)
                pairs.append((method_i, method_j))

        return pairs

    def _make_pair_key(self, method1: Method, method2: Method) -> Tuple[int, int]:
        """Create a consistent key for a pair of methods."""
        return tuple(sorted([method1.index, method2.index]))
