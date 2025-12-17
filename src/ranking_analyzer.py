"""
Ranking-based LLM analysis for Scope × Temporality.
Solves clustering problem by forcing relative ranking instead of absolute scoring.
"""
import asyncio
import logging
import json
import re
import time
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential
from dataclasses import dataclass
from scipy import stats

from src.data import Method

logger = logging.getLogger(__name__)


@dataclass
class RankingConfig:
    """Configuration for ranking-based analysis"""
    chunk_size: int = 18  # Items per LLM call (optimal for context)
    overlap_size: int = 4  # Overlap for calibration between chunks
    parallel_chunks: int = 4  # Parallel processing
    force_distribution: bool = True  # Force distribution (uniform or gaussian)
    distribution_type: str = 'gaussian'  # 'uniform' or 'gaussian'
    gaussian_std: float = 17.0  # Standard deviation for gaussian (17 ≈ 68% within ±17 of mean)
    add_jitter: bool = True  # Add small random variation
    jitter_amount: float = 0.5  # Max jitter in score points

    # Multi-pass validation for quality
    ranking_rounds: int = 1  # Multiple passes per dimension
    cross_validation: bool = False  # Validate consistency across chunks
    consistency_threshold: float = 0.8  # Minimum correlation between passes
    fail_on_low_consistency: bool = False  # Raise error if consistency below threshold

    # Calibrated scoring (realistic ranges, no forced 0s/100s)
    use_calibrated_scoring: bool = True  # Use realistic score ranges per dimension

    @staticmethod
    def get_dimension_ranges():
        """Define realistic score ranges for each dimension"""
        return {
            'scope': {
                'min_realistic': 5,   # Even tactical methods have some scope
                'max_realistic': 95,  # Even strategic methods aren't everywhere
                'typical_range': (20, 80)
            },
            'temporality': {
                'min_realistic': 5,   # Even immediate methods take some time
                'max_realistic': 95,  # Even long-term isn't infinite
                'typical_range': (20, 80)
            },
            'ease_adoption': {
                'min_realistic': 10,  # Even hardest aren't impossible
                'max_realistic': 90,  # Even easiest need some effort
                'typical_range': (25, 75)
            },
            'resources': {
                'min_realistic': 5,   # Some truly need minimal resources
                'max_realistic': 95,  # Some are extremely resource-intensive
                'typical_range': (20, 80)
            },
            'complexity': {
                'min_realistic': 5,   # Some methods are truly simple
                'max_realistic': 95,  # Some are extremely complex
                'typical_range': (20, 80)
            },
            'change_mgmt': {
                'min_realistic': 10,  # Some changes are quite easy
                'max_realistic': 90,  # Some require major cultural shift
                'typical_range': (25, 75)
            },
            'impact': {
                'min_realistic': 10,  # All methods have some impact
                'max_realistic': 95,  # Rarely perfect impact
                'typical_range': (25, 80)
            },
            'time_to_value': {
                'min_realistic': 10,  # Even slow methods eventually deliver
                'max_realistic': 90,  # Even fast methods need some time
                'typical_range': (25, 75)
            },
            'applicability': {
                'min_realistic': 10,  # Even niche has some applicability
                'max_realistic': 95,  # Few are truly universal
                'typical_range': (30, 85)
            },
            'people_focus': {
                'min_realistic': 5,   # Some are pure technical
                'max_realistic': 95,  # Some are pure human
                'typical_range': (20, 80)
            },
            'process_focus': {
                'min_realistic': 5,   # Some are truly ad-hoc
                'max_realistic': 95,  # Some are extremely rigid
                'typical_range': (20, 80)
            },
            'purpose_orientation': {
                'min_realistic': 10,  # Some are very internal
                'max_realistic': 90,  # Some are very external
                'typical_range': (25, 75)
            }
        }


class RankingLLMAnalyzer:
    """Ranking-based analyzer using existing LLM interface"""

    def __init__(self, config: dict, prompts_dir: str = "./prompts"):
        self.config = config['llm']
        self.base_url = self.config['base_url']
        self.api_key = self.config['api_key']
        self.model = self.config['model']
        self.temperature = self.config['temperature']
        self.timeout = self.config['timeout']
        self.max_concurrent = self.config['max_concurrent']

        self.prompts_dir = Path(prompts_dir)

        # Load ranking configuration from config file if available
        if 'ranking' in config:
            ranking_cfg = config['ranking']
            self.ranking_config = RankingConfig(
                chunk_size=ranking_cfg.get('chunk_size', 18),
                overlap_size=ranking_cfg.get('overlap_size', 4),
                parallel_chunks=ranking_cfg.get('parallel_chunks', 4),
                force_distribution=ranking_cfg.get('force_distribution', True),
                distribution_type=ranking_cfg.get('distribution_type', 'gaussian'),
                gaussian_std=ranking_cfg.get('gaussian_std', 17.0),
                add_jitter=ranking_cfg.get('add_jitter', True),
                jitter_amount=ranking_cfg.get('jitter_amount', 0.5),
                ranking_rounds=ranking_cfg.get('ranking_rounds', 1),
                cross_validation=ranking_cfg.get('cross_validation', False),
                consistency_threshold=ranking_cfg.get('consistency_threshold', 0.8),
                fail_on_low_consistency=ranking_cfg.get('fail_on_low_consistency', False),
                use_calibrated_scoring=ranking_cfg.get('use_calibrated_scoring', True)
            )
        else:
            self.ranking_config = RankingConfig()

        # Validate configuration for conflicting settings
        # FIX Issue #5: Auto-disable conflicting options instead of just warning
        if self.ranking_config.use_calibrated_scoring and self.ranking_config.force_distribution:
            logger.warning(
                "Both calibrated_scoring and force_distribution are enabled. "
                "Auto-disabling force_distribution since calibrated scoring takes precedence. "
                "To use Gaussian/uniform distributions, set use_calibrated_scoring=False."
            )
            # Auto-disable the conflicting option
            self.ranking_config.force_distribution = False

        # Load ranking prompts
        self.prompts = self._load_ranking_prompts()

    def _load_ranking_prompts(self) -> Dict[str, str]:
        """Load ranking prompts from external files in prompts directory"""
        prompts = {}

        # List of all ranking dimensions
        dimensions = [
            'scope',
            'temporality',
            'ease_adoption',
            'resources',
            'complexity',
            'change_mgmt',
            'impact',
            'time_to_value',
            'applicability',
            'people_focus',
            'process_focus',
            'purpose_orientation'
        ]

        # Load each prompt from file
        for dimension in dimensions:
            prompt_key = f'rank_{dimension}'
            prompt_file = self.prompts_dir / f'{prompt_key}.txt'

            if not prompt_file.exists():
                logger.error(f"Prompt file not found: {prompt_file}")
                raise FileNotFoundError(f"Missing ranking prompt: {prompt_file}")

            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    prompts[prompt_key] = f.read().strip()
                logger.debug(f"Loaded prompt for dimension: {dimension}")
            except Exception as e:
                logger.error(f"Failed to load prompt from {prompt_file}: {e}")
                raise

        logger.info(f"Loaded {len(prompts)} ranking prompts from {self.prompts_dir}")
        return prompts

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60))
    async def _call_llm(self, prompt: str, max_tokens: int = 800) -> str:
        """Call LLM API with retry logic"""
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

                    # Debug: Log response structure for diagnostics
                    if 'choices' not in data or len(data['choices']) == 0:
                        logger.error(f"Invalid response structure: {data}")
                        raise ValueError(f"No choices in response: {data}")

                    choice = data['choices'][0]
                    content = choice['message']['content']

                    # Check if response was truncated (hit token limit)
                    finish_reason = choice.get('finish_reason', 'unknown')
                    if finish_reason == 'length':
                        logger.warning(f"⚠️  Response truncated (hit max_tokens={max_tokens} limit)")
                        logger.warning(f"   Partial content: {content[:200] if content else 'None'}...")
                        raise ValueError(f"Response truncated at token limit - need higher max_tokens")

                    # Handle None/empty content from LLM
                    if content is None:
                        logger.error(f"LLM returned None content. finish_reason={finish_reason}")
                        logger.error(f"Full response: {data}")
                        raise ValueError(f"LLM returned None content (finish_reason={finish_reason})")

                    if not content.strip():
                        logger.error(f"LLM returned empty content. finish_reason={finish_reason}")
                        raise ValueError(f"LLM returned empty content (finish_reason={finish_reason})")

                    return content.strip()

        except asyncio.TimeoutError:
            logger.error(f"LLM call timeout after {self.timeout}s")
            raise
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    def _create_chunks(self, methods: List[Method], shuffle_seed: Optional[int] = None) -> List[List[Method]]:
        """Create overlapping chunks for ranking - ALL chunks exactly chunk_size

        Args:
            methods: List of methods to chunk
            shuffle_seed: If provided, shuffle methods using this seed before chunking
                         (enables different comparison groups across passes)

        Returns:
            List of chunks where EVERY chunk has exactly chunk_size items.
            Last chunk may have larger overlap to ensure exact size.
        """
        chunks = []
        chunk_size = self.ranking_config.chunk_size
        overlap = self.ranking_config.overlap_size
        step = chunk_size - overlap

        # Shuffle if seed provided (for cross-validation across passes)
        if shuffle_seed is not None:
            rng = np.random.RandomState(shuffle_seed)
            methods = list(methods)  # Create copy
            rng.shuffle(methods)
            logger.debug(f"Shuffled methods with seed {shuffle_seed} for diverse comparison groups")

        n = len(methods)

        if n <= chunk_size:
            # All methods fit in one chunk - pad if needed or return as-is
            return [methods]

        # Create chunks with fixed step, ensuring ALL chunks are exactly chunk_size
        i = 0
        while i < n:
            # If we have enough items left, take chunk_size items
            if i + chunk_size <= n:
                chunk = methods[i:i + chunk_size]
                chunks.append(chunk)
                i += step
            else:
                # Last chunk: take the final chunk_size items
                # This ensures exactly chunk_size items (may have larger overlap)
                last_chunk = methods[n - chunk_size:n]

                # Only add if it's different from the previous chunk
                if not chunks or last_chunk != chunks[-1]:
                    chunks.append(last_chunk)
                break

        # Log chunk size distribution - should all be chunk_size now
        chunk_sizes = [len(c) for c in chunks]
        logger.debug(f"Created {len(chunks)} uniform chunks: all size={chunk_size}")

        # Verify all chunks are exactly chunk_size
        if any(size != chunk_size for size in chunk_sizes):
            logger.warning(f"⚠️  Non-uniform chunks detected: {chunk_sizes}")

        return chunks

    def _format_methods_for_ranking(self, methods: List[Method]) -> str:
        """Format methods list for ranking prompt with full descriptions"""
        formatted = []
        for i, method in enumerate(methods, 1):
            # Use FULL description (no truncation) for better LLM understanding
            desc = method.description
            formatted.append(f"{i}. {method.name}")
            if desc:
                formatted.append(f"   {desc}")

        return "\n".join(formatted)

    def _parse_ranking_response(self, response: str, methods: List[Method]) -> Dict[str, int]:
        """Parse LLM ranking response - handles both method numbers and names"""
        try:
            # Debug: Log response length and preview
            response_preview = response[:300] + "..." if len(response) > 300 else response
            logger.debug(f"Parsing response ({len(response)} chars): {response_preview}")

            # Try to extract JSON array
            json_match = re.search(r'\[\s*\[.*?\]\s*\]', response, re.DOTALL)
            if json_match:
                rankings_data = json.loads(json_match.group())
                logger.debug(f"Extracted JSON array with {len(rankings_data)} items")
            else:
                # Try parsing entire response as JSON
                logger.debug("No JSON array pattern found, trying to parse entire response")
                rankings_data = json.loads(response)

            # Convert to dict: method_name -> rank
            rankings = {}
            parse_errors = 0
            invalid_ranks = []

            for item in rankings_data:
                method_identifier, rank = item

                # CRITICAL: Validate rank value first - reject bad values outright
                if not isinstance(rank, (int, float)):
                    parse_errors += 1
                    invalid_ranks.append(f"{method_identifier}: type={type(rank)}")
                    logger.warning(f"Invalid rank type: {type(rank)} for {method_identifier}")
                    continue

                # REJECT out-of-range ranks (trigger retry instead of clipping)
                if rank < 1 or rank > len(methods):
                    parse_errors += 1
                    invalid_ranks.append(f"{method_identifier}: rank={rank} (valid: 1-{len(methods)})")
                    logger.warning(f"Rank {rank} out of range [1, {len(methods)}] for {method_identifier} - REJECTING")
                    continue

                # Handle both integer (method number) and string (method name)
                if isinstance(method_identifier, int):
                    # Method number (1-indexed)
                    if 1 <= method_identifier <= len(methods):
                        method = methods[method_identifier - 1]
                        rankings[method.name] = int(rank)  # Ensure integer
                    else:
                        parse_errors += 1
                        logger.warning(f"Method number {method_identifier} out of range (1-{len(methods)})")

                elif isinstance(method_identifier, str):
                    # Method name - find matching method
                    method_name = method_identifier.strip()
                    matched = False

                    # Try exact match first
                    for method in methods:
                        if method.name == method_name:
                            rankings[method.name] = int(rank)  # Ensure integer
                            matched = True
                            break

                    # Try partial match if exact fails
                    if not matched:
                        for method in methods:
                            if method_name in method.name or method.name in method_name:
                                rankings[method.name] = int(rank)  # Ensure integer
                                matched = True
                                logger.debug(f"Partial match: '{method_name}' -> '{method.name}'")
                                break

                    if not matched:
                        parse_errors += 1
                        logger.warning(f"Could not match method name: '{method_name}'")

            if parse_errors > 0:
                logger.warning(f"Parse errors: {parse_errors}/{len(rankings_data)} items failed")
                if invalid_ranks:
                    logger.warning(f"Invalid ranks detected: {', '.join(invalid_ranks[:5])}")  # Show first 5

            # STRICT: Reject response if ANY invalid ranks detected
            if invalid_ranks:
                logger.error(f"Response contains {len(invalid_ranks)} invalid ranks - REJECTING entire response")
                logger.error(f"Invalid ranks: {', '.join(invalid_ranks)}")
                raise ValueError(f"Invalid ranks detected: {len(invalid_ranks)} out-of-range or invalid values")

            # CRITICAL: Check for duplicate ranks (violates "no ties" requirement)
            rank_values = list(rankings.values())
            unique_ranks = set(rank_values)

            if len(rank_values) != len(unique_ranks):
                # Find which ranks are duplicated
                rank_counts = {}
                for rank in rank_values:
                    rank_counts[rank] = rank_counts.get(rank, 0) + 1

                duplicates = {rank: count for rank, count in rank_counts.items() if count > 1}
                logger.error(f"Duplicate ranks detected (violates 'no ties' rule): {duplicates}")
                logger.error(f"Example: {len(duplicates)} ranks appear multiple times")
                raise ValueError(f"Duplicate ranks detected: {len(duplicates)} ranks used multiple times (expected unique ranks 1-{len(methods)})")

            # Validate that we have a proper ranking (all ranks 1-N used exactly once)
            # Only check this if we have complete rankings
            if len(rankings) == len(methods):
                expected_ranks = set(range(1, len(methods) + 1))
                actual_ranks = set(int(r) for r in rank_values)

                if actual_ranks != expected_ranks:
                    missing_ranks = expected_ranks - actual_ranks
                    extra_ranks = actual_ranks - expected_ranks
                    logger.warning(f"Ranking not contiguous: missing {missing_ranks}, extra {extra_ranks}")
                    # Don't reject, but log warning - LLM may have skipped/misused ranks

            # STRICT: Require ALL methods to be ranked (no partial rankings allowed)
            if len(rankings) < len(methods):
                missing_count = len(methods) - len(rankings)
                missing_methods = [m.name for m in methods if m.name not in rankings]
                logger.error(f"LLM failed to rank {missing_count}/{len(methods)} methods")
                logger.error(f"Missing methods: {', '.join(missing_methods[:5])}{'...' if len(missing_methods) > 5 else ''}")
                raise ValueError(f"Incomplete rankings: only {len(rankings)}/{len(methods)} methods ranked - REJECTING")

            return rankings

        except Exception as e:
            logger.error(f"Failed to parse ranking response: {e}")
            logger.error(f"Response was: {response[:500]}")

            # Fallback: return None to trigger retry
            # Don't assign random sequential ranks!
            return None

    async def _rank_chunk(self,
                         chunk: List[Method],
                         dimension: str,
                         chunk_num: int,
                         max_retries: int = 50) -> Dict[str, int]:
        """Rank a single chunk on one dimension with retry until success

        Retries with exponential backoff if:
        - LLM returns invalid ranks (out of range)
        - LLM returns None/empty content
        - Parse errors occur
        - Network/API errors

        Args:
            max_retries: Maximum number of retry attempts (default: 50)

        Raises:
            RuntimeError: If max retries exceeded without success
        """
        # Verbose chunk logging is now shown at batch level

        # Prepare prompt
        methods_list = self._format_methods_for_ranking(chunk)
        prompt_key = f"rank_{dimension}"

        if prompt_key not in self.prompts:
            raise ValueError(f"No prompt for dimension: {dimension}")

        prompt = self.prompts[prompt_key].format(
            count=len(chunk),
            methods_list=methods_list
        )

        # Retry with limit
        attempt = 0
        max_backoff = 30  # Max 30 seconds between retries

        while attempt < max_retries:
            attempt += 1
            try:
                # Call LLM (very high token limit for thinking models that add reasoning/comments)
                # Thinking models can generate 2-3k tokens with explanations for 10 items
                response = await self._call_llm(prompt, max_tokens=12000)

                # Parse rankings
                rankings = self._parse_ranking_response(response, chunk)

                if rankings is not None:
                    if attempt > 1:
                        logger.info(f"  ✓ Chunk {chunk_num} succeeded on attempt {attempt}")
                    return rankings
                else:
                    logger.warning(f"Attempt {attempt}: Parse failed for chunk {chunk_num}, retrying...")
                    backoff = min(2 ** min(attempt - 1, 5), max_backoff)  # Exponential backoff
                    await asyncio.sleep(backoff)

            except ValueError as e:
                # Invalid ranks detected - retry with backoff
                if "Invalid ranks detected" in str(e):
                    logger.warning(f"Attempt {attempt}: Invalid ranks in chunk {chunk_num}, retrying...")
                    backoff = min(2 ** min(attempt - 1, 5), max_backoff)
                    await asyncio.sleep(backoff)
                else:
                    # Other ValueError - log and retry
                    logger.error(f"Attempt {attempt} failed for chunk {chunk_num}: {e}, retrying...")
                    backoff = min(2 ** min(attempt - 1, 5), max_backoff)
                    await asyncio.sleep(backoff)

            except Exception as e:
                # Network/API errors - retry with backoff
                logger.error(f"Attempt {attempt} failed for chunk {chunk_num}: {e}, retrying...")
                backoff = min(2 ** min(attempt - 1, 5), max_backoff)
                await asyncio.sleep(backoff)

            # Progress: Warn if many retries
            if attempt % 10 == 0 and attempt < max_retries:
                logger.warning(f"⚠️  Chunk {chunk_num} still failing after {attempt} attempts - {max_retries - attempt} attempts remaining...")

        # If we get here, max retries exceeded
        error_msg = (
            f"❌ Chunk {chunk_num} failed after {max_retries} attempts. "
            f"This chunk consistently returns invalid/None responses. "
            f"Possible issues: prompt too complex, context length exceeded, or model incompatibility."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    async def _rank_all_chunks(self,
                               chunks: List[List[Method]],
                               dimension: str) -> List[Dict[str, int]]:
        """Rank all chunks with parallel processing"""
        all_rankings = []

        # Process in batches to respect max_concurrent
        batch_size = self.ranking_config.parallel_chunks
        total_batches = (len(chunks) + batch_size - 1) // batch_size

        for batch_idx, batch_start in enumerate(range(0, len(chunks), batch_size), 1):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch = chunks[batch_start:batch_end]

            # Show batch progress
            print(f"    Batch {batch_idx}/{total_batches}: Processing chunks {batch_start+1}-{batch_end} ({len(batch)} parallel)...", end='', flush=True)
            batch_start_time = time.time()

            # Create tasks for this batch
            tasks = []
            for i, chunk in enumerate(batch):
                chunk_num = batch_start + i + 1
                task = self._rank_chunk(chunk, dimension, chunk_num)
                tasks.append(task)

            # Execute batch in parallel
            batch_results = await asyncio.gather(*tasks)
            all_rankings.extend(batch_results)

            batch_time = time.time() - batch_start_time
            print(f" ✓ ({batch_time:.1f}s)")

            # Small delay between batches
            if batch_end < len(chunks):
                await asyncio.sleep(0.5)

        return all_rankings

    def _merge_chunk_rankings(self,
                             chunk_rankings: List[Dict[str, int]],
                             chunks: List[List[Method]]) -> Dict[str, float]:
        """Merge chunk rankings into global scores using overlap calibration"""
        logger.info("  Merging chunk rankings with calibration...")

        # Collect all methods with their chunk-local normalized scores
        method_scores = {}  # method_name -> list of (chunk_idx, normalized_score)

        for chunk_idx, (rankings, chunk) in enumerate(zip(chunk_rankings, chunks)):
            chunk_size = len(chunk)

            for method_name, rank in rankings.items():
                # Defensive: Validate rank is in valid range (should never trigger - parser rejects invalid ranks)
                if rank < 1 or rank > chunk_size:
                    logger.error(f"UNEXPECTED: Invalid rank {rank} for {method_name} in chunk of {chunk_size} - this should have been caught by parser!")
                    rank = max(1, min(chunk_size, rank))

                # Normalize rank to 0-1 within chunk
                normalized = (rank - 1) / (chunk_size - 1) if chunk_size > 1 else 0.5

                # Defensive: Clip to ensure [0, 1] (should never trigger)
                if normalized < 0.0 or normalized > 1.0:
                    logger.error(f"UNEXPECTED: normalized score {normalized} out of range for {method_name}!")
                    normalized = max(0.0, min(1.0, normalized))

                if method_name not in method_scores:
                    method_scores[method_name] = []

                method_scores[method_name].append((chunk_idx, normalized))

        # For methods in multiple chunks, average their scores
        # This handles overlap naturally
        averaged_scores = {}
        for method_name, scores in method_scores.items():
            if len(scores) == 1:
                averaged_scores[method_name] = scores[0][1]
            else:
                # Average scores from different chunks
                avg = np.mean([score for _, score in scores])
                averaged_scores[method_name] = avg

            # Final clip to [0, 1] (defensive)
            averaged_scores[method_name] = max(0.0, min(1.0, averaged_scores[method_name]))

        return averaged_scores

    def _validate_overlap_consistency(self,
                                      chunk_rankings: List[Dict[str, int]],
                                      chunks: List[List[Method]]) -> Dict[str, float]:
        """Validate that methods in overlap regions maintain consistent rankings

        For methods appearing in multiple chunks, check if their relative positions
        (normalized ranks) are consistent across chunks.

        Args:
            chunk_rankings: List of ranking dicts from each chunk
            chunks: List of method chunks

        Returns:
            Dict with consistency metrics:
            - 'overlap_methods': number of methods in overlaps
            - 'avg_variance': average variance of normalized ranks for overlap methods
            - 'max_variance': maximum variance observed
            - 'consistent_count': number of methods with low variance (< 0.1)
        """
        # Collect normalized ranks for each method across chunks
        method_ranks = {}  # method_name -> list of (chunk_idx, normalized_rank)

        for chunk_idx, (rankings, chunk) in enumerate(zip(chunk_rankings, chunks)):
            chunk_size = len(chunk)

            for method_name, rank in rankings.items():
                # Normalize rank to 0-1 within chunk
                normalized = (rank - 1) / (chunk_size - 1) if chunk_size > 1 else 0.5

                if method_name not in method_ranks:
                    method_ranks[method_name] = []

                method_ranks[method_name].append((chunk_idx, normalized))

        # Analyze methods in multiple chunks (overlap regions)
        overlap_methods = {name: ranks for name, ranks in method_ranks.items() if len(ranks) > 1}

        if not overlap_methods:
            logger.info("  No overlap methods to validate (single chunk)")
            return {
                'overlap_methods': 0,
                'avg_variance': 0.0,
                'max_variance': 0.0,
                'consistent_count': 0
            }

        variances = []
        inconsistent_methods = []
        variance_threshold = 0.1  # Normalized ranks should vary by < 0.1 (10 percentile points)

        for method_name, ranks in overlap_methods.items():
            # Calculate variance of normalized ranks
            normalized_values = [norm_rank for _, norm_rank in ranks]
            variance = np.var(normalized_values)
            variances.append(variance)

            # Flag high-variance methods
            if variance > variance_threshold:
                min_rank = min(normalized_values)
                max_rank = max(normalized_values)
                rank_spread = max_rank - min_rank
                inconsistent_methods.append({
                    'name': method_name,
                    'variance': variance,
                    'spread': rank_spread,
                    'num_chunks': len(ranks),
                    'chunks': [chunk_idx for chunk_idx, _ in ranks]
                })

        # Calculate metrics
        avg_variance = float(np.mean(variances))
        max_variance = float(np.max(variances))
        consistent_count = len([v for v in variances if v <= variance_threshold])
        consistency_pct = (consistent_count / len(overlap_methods)) * 100

        # Log results
        logger.info(f"  Overlap consistency validation:")
        logger.info(f"    Methods in overlaps: {len(overlap_methods)}")
        logger.info(f"    Avg variance: {avg_variance:.4f}")
        logger.info(f"    Max variance: {max_variance:.4f}")
        logger.info(f"    Consistent methods: {consistent_count}/{len(overlap_methods)} ({consistency_pct:.1f}%)")

        # Warn about inconsistent methods
        if inconsistent_methods:
            inconsistent_methods.sort(key=lambda x: x['variance'], reverse=True)
            logger.warning(f"    ⚠️  {len(inconsistent_methods)} methods show inconsistent rankings across chunks")

            # Show worst offenders (top 5)
            for method in inconsistent_methods[:5]:
                logger.warning(f"      - {method['name']}: variance={method['variance']:.4f}, "
                             f"spread={method['spread']:.2f} across {method['num_chunks']} chunks")

            if len(inconsistent_methods) > 5:
                logger.warning(f"      ... and {len(inconsistent_methods) - 5} more")

            # If many methods are inconsistent, this suggests chunking issues
            if len(inconsistent_methods) > len(overlap_methods) * 0.3:
                logger.warning(f"    ⚠️  HIGH INCONSISTENCY: >30% of overlap methods have high variance")
                logger.warning(f"    This may indicate: chunk size too small, or dimension is ambiguous")
        else:
            logger.info(f"    ✅ All overlap methods show consistent rankings")

        return {
            'overlap_methods': len(overlap_methods),
            'avg_variance': avg_variance,
            'max_variance': max_variance,
            'consistent_count': consistent_count,
            'inconsistent_methods': inconsistent_methods
        }

    def _convert_to_final_scores(self,
                                 normalized_scores: Dict[str, float],
                                 dimension: str = None) -> Dict[str, float]:
        """Convert normalized scores (0-1) to final scores (0-100) with calibrated ranges

        Args:
            normalized_scores: Scores in 0-1 range from ranking
            dimension: Dimension name (for calibrated scoring)

        Note:
            If use_calibrated_scoring is enabled, it takes precedence over
            force_distribution/distribution_type settings.
        """

        # Use calibrated scoring if enabled and dimension provided
        # This bypasses Gaussian/uniform distribution settings
        if self.ranking_config.use_calibrated_scoring and dimension:
            return self._calibrated_score_conversion(normalized_scores, dimension)

        if not self.ranking_config.force_distribution:
            # Simple linear mapping with strict clipping
            final_scores = {}
            for name, score in normalized_scores.items():
                # Clip input to [0, 1] (defensive)
                score = max(0.0, min(1.0, score))
                # Scale to [0, 100]
                final_score = score * 100.0
                # Clip output to [0, 100] (defensive)
                final_score = max(0.0, min(100.0, final_score))
                final_scores[name] = round(final_score, 2)
            return final_scores

        # Sort items by their normalized score
        sorted_items = sorted(normalized_scores.items(), key=lambda x: x[1])
        n = len(sorted_items)

        final_scores = {}

        if self.ranking_config.distribution_type == 'gaussian':
            # Gaussian/Normal distribution (more realistic)
            # Convert rank percentile to normal distribution with mean=50, std=gaussian_std
            for i, (name, _) in enumerate(sorted_items):
                # FIX Issue #2: Calculate percentile avoiding exact 0.0 and 1.0
                # Use (i + 0.5) / n to map to (0.5/n, 1 - 0.5/n) range
                # This prevents stats.norm.ppf(0.0) = -inf and stats.norm.ppf(1.0) = +inf
                percentile = (i + 0.5) / n

                # Convert percentile to z-score using inverse normal CDF
                # percentile 0.5 → z=0 → score=50 (mean)
                # percentile ~0.16 → z≈-1 → score≈33 (one std below mean)
                # percentile ~0.84 → z≈+1 → score≈67 (one std above mean)
                z_score = stats.norm.ppf(percentile)

                # Convert z-score to final score: mean + (z * std)
                base_score = 50 + (z_score * self.ranking_config.gaussian_std)

                # Clip to valid range (0-100)
                base_score = np.clip(base_score, 0, 100)

                # FIX Issue #3: Apply order-preserving jitter
                # Jitter is only applied if it won't flip rankings
                if self.ranking_config.add_jitter:
                    # Calculate safe jitter bounds to preserve order
                    prev_score = final_scores.get(sorted_items[i-1][0], base_score - 1) if i > 0 else 0
                    next_score = 100  # Will be set by next iteration

                    # Apply jitter that preserves order
                    jitter = np.random.uniform(-self.ranking_config.jitter_amount,
                                              self.ranking_config.jitter_amount)

                    # Ensure score with jitter stays above previous and leaves room for next
                    score = base_score + jitter
                    score = max(prev_score + 0.01, min(score, 100))  # Stay above previous
                else:
                    score = base_score

                final_scores[name] = round(score, 2)

        else:  # 'uniform' distribution
            # Force even/uniform distribution
            for i, (name, _) in enumerate(sorted_items):
                # Even distribution
                base_score = (i / (n - 1)) * 100 if n > 1 else 50

                # FIX Issue #3: Apply order-preserving jitter
                if self.ranking_config.add_jitter:
                    # Calculate safe jitter bounds to preserve order
                    prev_score = final_scores.get(sorted_items[i-1][0], base_score - 1) if i > 0 else 0

                    # Apply jitter that preserves order
                    jitter = np.random.uniform(-self.ranking_config.jitter_amount,
                                              self.ranking_config.jitter_amount)

                    # Ensure score with jitter stays above previous
                    score = base_score + jitter
                    score = max(prev_score + 0.01, min(score, 100))
                else:
                    score = base_score

                final_scores[name] = round(score, 2)

        return final_scores

    def _calibrated_score_conversion(self,
                                     normalized_scores: Dict[str, float],
                                     dimension: str) -> Dict[str, float]:
        """Convert normalized (0-1) scores to realistic calibrated scores (not forced 0-100)

        Maps to dimension-specific realistic ranges to avoid artificial extremes.
        FIX Issue #3: Process in ranked order to apply order-preserving jitter.
        """
        dimension_ranges = RankingConfig.get_dimension_ranges()

        if dimension not in dimension_ranges:
            # Fallback to simple linear if dimension not configured
            logger.warning(f"No calibration range for {dimension}, using linear mapping")
            return {name: score * 100.0 for name, score in normalized_scores.items()}

        dim_config = dimension_ranges[dimension]
        min_realistic = dim_config['min_realistic']
        max_realistic = dim_config['max_realistic']
        typical_min, typical_max = dim_config['typical_range']

        # FIX: Sort items by normalized score to process in rank order
        sorted_items = sorted(normalized_scores.items(), key=lambda x: x[1])
        final_scores = {}

        for i, (name, norm_score) in enumerate(sorted_items):
            # Clip to [0, 1]
            norm_score = max(0.0, min(1.0, norm_score))

            # Map to realistic range with soft boundaries
            # Use typical range for middle 80%, compress extremes

            if norm_score < 0.1:
                # Bottom 10% compressed into min_realistic to typical_min
                compressed_pos = norm_score / 0.1  # 0-1 within bottom 10%
                base_score = min_realistic + compressed_pos * (typical_min - min_realistic)

            elif norm_score > 0.9:
                # Top 10% compressed into typical_max to max_realistic
                compressed_pos = (norm_score - 0.9) / 0.1  # 0-1 within top 10%
                base_score = typical_max + compressed_pos * (max_realistic - typical_max)

            else:
                # Middle 80% linearly mapped to typical range
                middle_pos = (norm_score - 0.1) / 0.8  # 0-1 within middle 80%
                base_score = typical_min + middle_pos * (typical_max - typical_min)

            # FIX Issue #3: Apply order-preserving jitter
            if self.ranking_config.add_jitter:
                # Get previous score to ensure we stay above it
                prev_score = final_scores.get(sorted_items[i-1][0], min_realistic - 1) if i > 0 else min_realistic - 1

                # Add jitter using normal distribution
                jitter = np.random.normal(0, self.ranking_config.jitter_amount)
                score = base_score + jitter

                # Ensure order is preserved: stay above previous, within bounds
                score = max(prev_score + 0.01, min(score, max_realistic))
            else:
                score = base_score
                # Clip to absolute bounds
                score = max(min_realistic, min(max_realistic, score))

            final_scores[name] = round(score, 2)

        logger.info(f"  Calibrated scores: {min(final_scores.values()):.1f} - {max(final_scores.values()):.1f}")

        return final_scores

    def _trim_outliers(self, scores: List[float], trim_pct_total: float = 0.2) -> List[float]:
        """Remove outlier scores using trimmed mean approach

        Args:
            scores: List of scores from multiple passes
            trim_pct_total: Total percentage to trim (split between both ends)
                          Default 0.2 means remove 10% from bottom + 10% from top = 20% total

        Returns:
            Trimmed list of scores (middle values after removing outliers)
        """
        if len(scores) <= 2:
            return scores  # Can't trim with 2 or fewer scores

        # Sort scores
        sorted_scores = sorted(scores)
        n = len(sorted_scores)

        # Calculate how many to trim from each end
        # Example: 0.2 total → 0.1 from each end
        trim_count = max(1, int(n * trim_pct_total / 2))

        # Return middle scores
        if trim_count * 2 >= n:
            # If trimming would remove everything, just remove extremes
            return sorted_scores[1:-1] if n > 2 else sorted_scores

        return sorted_scores[trim_count:-trim_count]

    def _combine_multiple_passes(self, all_passes_scores: List[Dict[str, float]]) -> Dict[str, float]:
        """Combine scores from multiple ranking passes using trimmed mean

        Uses trimmed mean instead of median to reduce outlier impact while
        preserving more information than median alone.
        """
        if len(all_passes_scores) == 1:
            # Ensure single-pass scores are also clipped
            return {name: max(0.0, min(100.0, score)) for name, score in all_passes_scores[0].items()}

        # Get all method names
        all_methods = set()
        for pass_scores in all_passes_scores:
            all_methods.update(pass_scores.keys())

        # Calculate trimmed mean score for each method across passes
        combined_scores = {}
        for method_name in all_methods:
            # Collect scores from all passes (use 50 if method missing in a pass)
            scores = [pass_scores.get(method_name, 50.0) for pass_scores in all_passes_scores]

            if len(scores) >= 5:
                # With 5+ passes, use trimmed mean (remove outliers)
                trimmed = self._trim_outliers(scores)
                final_score = float(np.mean(trimmed))
            elif len(scores) >= 3:
                # With 3-4 passes, use median (more robust)
                final_score = float(np.median(scores))
            else:
                # With 1-2 passes, just use mean
                final_score = float(np.mean(scores))

            # Clip to [0, 100] (defensive)
            combined_scores[method_name] = max(0.0, min(100.0, final_score))

        return combined_scores

    def _global_calibration(self,
                           scores: Dict[str, float],
                           dimension: str) -> Dict[str, float]:
        """Final global calibration to ensure realistic distribution across all methods

        This is a light touch-up to ensure no artificial extremes while preserving
        the ranking relationships from the multi-pass analysis.
        """
        if not self.ranking_config.use_calibrated_scoring:
            return scores  # Skip if calibrated scoring disabled

        dimension_ranges = RankingConfig.get_dimension_ranges()

        if dimension not in dimension_ranges:
            return scores

        dim_config = dimension_ranges[dimension]
        values = list(scores.values())

        if not values:
            return scores

        actual_min = min(values)
        actual_max = max(values)
        actual_mean = np.mean(values)
        actual_std = np.std(values)

        min_realistic = dim_config['min_realistic']
        max_realistic = dim_config['max_realistic']
        # typical_range not used in global calibration, only min/max bounds

        logger.info(f"  Global calibration: current range {actual_min:.1f}-{actual_max:.1f}, "
                   f"mean={actual_mean:.1f}, std={actual_std:.1f}")

        # Apply soft compression to outliers beyond realistic bounds
        # Uses asymptotic compression to avoid hard clipping while preventing extreme values
        calibrated = {}
        compression_applied = 0
        compression_buffer = 3.0  # How much space to leave for compressed outliers

        for name, score in scores.items():
            if score < min_realistic:
                # Soft compression below minimum
                # Map [actual_min, min_realistic] → [min_realistic - buffer, min_realistic]
                if actual_min < min_realistic:
                    # Linear interpolation into compressed range
                    ratio = (score - actual_min) / (min_realistic - actual_min)
                    calibrated[name] = (min_realistic - compression_buffer) + ratio * compression_buffer
                else:
                    # Edge case: all scores already above min
                    calibrated[name] = min_realistic
                compression_applied += 1

            elif score > max_realistic:
                # Soft compression above maximum
                # Map [max_realistic, actual_max] → [max_realistic, max_realistic + buffer]
                if actual_max > max_realistic:
                    # Linear interpolation into compressed range
                    ratio = (score - max_realistic) / (actual_max - max_realistic)
                    calibrated[name] = max_realistic + ratio * compression_buffer
                else:
                    # Edge case: all scores already below max
                    calibrated[name] = max_realistic
                compression_applied += 1

            else:
                # Keep as-is if within realistic bounds
                calibrated[name] = score

        if compression_applied > 0:
            logger.info(f"  Compressed {compression_applied} extreme scores to realistic bounds")

        return calibrated

    def _calculate_pass_consistency(self, all_passes_scores: List[Dict[str, float]]) -> float:
        """Calculate consistency (correlation) between ranking passes"""
        if len(all_passes_scores) < 2:
            return 1.0

        # Get common methods across all passes
        common_methods = set(all_passes_scores[0].keys())
        for pass_scores in all_passes_scores[1:]:
            common_methods &= set(pass_scores.keys())

        if len(common_methods) < 10:
            logger.warning(f"Too few common methods to calculate consistency ({len(common_methods)} < 10)")
            # Return 1.0 (assume consistent) rather than 0.0 (inconsistent)
            # to avoid false warnings when we simply can't measure
            return 1.0

        # Extract score vectors for each pass
        pass_vectors = []
        for pass_scores in all_passes_scores:
            vector = [pass_scores[method] for method in sorted(common_methods)]
            pass_vectors.append(vector)

        # Calculate average pairwise correlation
        correlations = []
        for i in range(len(pass_vectors)):
            for j in range(i + 1, len(pass_vectors)):
                corr = np.corrcoef(pass_vectors[i], pass_vectors[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

        return float(np.mean(correlations)) if correlations else 0.0

    def _validate_distribution(self, scores: Dict[str, float], dimension: str):
        """Validate score distribution"""
        values = list(scores.values())

        unique_values = len(set(values))
        mean = np.mean(values)
        std = np.std(values)

        # Check distribution across bins
        hist, _ = np.histogram(values, bins=10)
        uniformity = np.std(hist) / np.mean(hist) if np.mean(hist) > 0 else 0

        logger.info(f"\n  {dimension.upper()} Distribution:")
        logger.info(f"    Range: {min(values):.1f} - {max(values):.1f}")
        logger.info(f"    Mean: {mean:.1f}, Std: {std:.1f}")
        logger.info(f"    Unique values: {unique_values}/{len(values)}")
        logger.info(f"    Uniformity: {uniformity:.2f} (lower=better)")

        if uniformity > 1.0:
            logger.warning(f"    ⚠️  High clustering detected")
        else:
            logger.info(f"    ✅ Good distribution achieved")

    async def rank_dimension(self, methods: List[Method], dimension: str) -> Dict[int, float]:
        """
        Rank all methods on one dimension using chunked approach.

        Args:
            methods: List of Method objects
            dimension: 'scope' or 'temporality'

        Returns:
            Dictionary mapping method index to score (0-100)
        """
        logger.info(f"\nRanking {len(methods)} methods on {dimension.upper()}:")

        # Calculate expected chunk count (for info display)
        num_chunks_approx = (len(methods) + self.ranking_config.chunk_size - self.ranking_config.overlap_size - 1) // (self.ranking_config.chunk_size - self.ranking_config.overlap_size)
        num_batches = (num_chunks_approx + self.ranking_config.parallel_chunks - 1) // self.ranking_config.parallel_chunks
        num_passes = self.ranking_config.ranking_rounds

        print(f"  → ~{num_chunks_approx} chunks of ~{self.ranking_config.chunk_size} methods")
        print(f"  → ~{num_batches} batches ({self.ranking_config.parallel_chunks} chunks/batch)")
        print(f"  → {num_passes} validation passes (shuffled comparison groups)")

        logger.info(f"  Will create ~{num_chunks_approx} chunks (size ~{self.ranking_config.chunk_size}, overlap={self.ranking_config.overlap_size})")

        # Step 2: Multi-pass ranking with shuffled comparison groups
        all_passes_scores = []

        for pass_num in range(1, num_passes + 1):
            # CRITICAL: Shuffle methods differently for each pass
            # This ensures each method gets ranked against diverse comparison groups
            shuffle_seed = 42 + pass_num - 1  # All passes shuffled with different seeds

            print(f"  → Pass {pass_num}/{num_passes}: Creating chunks (shuffled)...")

            # Create chunks for this pass (potentially shuffled)
            chunks = self._create_chunks(methods, shuffle_seed=shuffle_seed)

            print(f"     Ranking {len(chunks)} chunks...")

            # Rank each chunk in this pass
            chunk_rankings = await self._rank_all_chunks(chunks, dimension)

            # Validate overlap consistency before merging
            if len(chunks) > 1:
                overlap_metrics = self._validate_overlap_consistency(chunk_rankings, chunks)
                # Store metrics for analysis (could be returned/logged later if needed)

            # Merge this pass's rankings
            normalized_scores = self._merge_chunk_rankings(chunk_rankings, chunks)

            # Convert to 0-100 scale with calibrated ranges
            pass_scores = self._convert_to_final_scores(normalized_scores, dimension=dimension)

            all_passes_scores.append(pass_scores)

            # Show pass completion
            if pass_num < num_passes:
                print(f"     Pass {pass_num} complete. Starting pass {pass_num + 1}...")

        # Step 3: Combine multiple passes using trimmed mean
        combine_method = "trimmed mean" if num_passes >= 5 else ("median" if num_passes >= 3 else "mean")
        print(f"  → Combining {num_passes} passes using {combine_method}...")
        combined_scores = self._combine_multiple_passes(all_passes_scores)

        # Validate consistency across passes
        # FIX Issue #4: Add option to fail on low consistency
        if num_passes > 1:
            consistency = self._calculate_pass_consistency(all_passes_scores)
            print(f"  → Pass consistency: {consistency:.2f} (correlation between passes)")
            if consistency < self.ranking_config.consistency_threshold:
                warning_msg = f"Low consistency ({consistency:.2f} < {self.ranking_config.consistency_threshold}) - results may be unstable"
                print(f"     ⚠️  {warning_msg}")

                if self.ranking_config.fail_on_low_consistency:
                    error_msg = (
                        f"Consistency check failed for dimension '{dimension}': "
                        f"correlation {consistency:.2f} is below threshold {self.ranking_config.consistency_threshold}. "
                        f"This indicates unstable rankings across passes. "
                        f"Consider increasing ranking_rounds or reviewing the dimension definition."
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)

        # Step 4: Global calibration to ensure realistic distribution
        print(f"  → Applying global calibration...")
        final_scores = self._global_calibration(combined_scores, dimension)

        # Step 5: Validate final distribution
        self._validate_distribution(final_scores, dimension)

        # Convert to index-based dict
        index_scores = {}
        for method in methods:
            if method.name in final_scores:
                index_scores[method.index] = final_scores[method.name]

        return index_scores

    async def batch_analyze_scope_temporality_ranked(self,
                                                    methods: List[Method]) -> Dict[int, Dict]:
        """
        Analyze all methods using ranking approach.

        Returns:
            Dictionary mapping method index to:
            {
                'scope_score': float (0-100),
                'temporality_score': float (0-100),
                'reasoning': str
            }
        """
        logger.info("\n" + "="*70)
        logger.info("RANKING-BASED SCOPE × TEMPORALITY ANALYSIS")
        logger.info("="*70)

        # Rank on both dimensions
        scope_scores = await self.rank_dimension(methods, 'scope')
        temporality_scores = await self.rank_dimension(methods, 'temporality')

        # Combine results
        results = {}
        for method in methods:
            idx = method.index

            # Convert to 0-1 scale for consistency with existing code
            scope = scope_scores.get(idx, 50) / 100.0
            temporality = temporality_scores.get(idx, 50) / 100.0

            results[idx] = {
                'scope_score': scope,
                'temporality_score': temporality,
                'reasoning': f"Ranked by relative position: scope={scope:.2f}, temporality={temporality:.2f}"
            }

        logger.info(f"\n✅ Ranking analysis complete for {len(results)} methods")

        return results

    async def batch_analyze_9d_ranked(self,
                                     methods: List[Method]) -> Dict[int, Dict]:
        """
        Comprehensive 9-dimensional analysis using ranking approach.

        Ranks all methods on 9 dimensions:
        1. Scope (Tactical → Strategic)
        2. Temporality (Immediate → Evolutionary)
        3. Ease of Adoption
        4. Resources Required (inverted to "low resources needed")
        5. Technical Complexity (inverted to "low complexity")
        6. Change Management (inverted to "low change resistance")
        7. Impact Potential
        8. Time to Value (inverted to "fast delivery")
        9. Applicability (Narrow → Broad)

        Also calculates:
        - Implementation Difficulty = average of (100 - ease_adoption, 100 - resources, 100 - complexity, 100 - change_mgmt)

        Returns:
            Dictionary mapping method index to all dimension scores (0-100)
        """
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE 9-DIMENSIONAL RANKING ANALYSIS")
        logger.info("="*80)
        logger.info("\nThis will rank methods on 9 separate dimensions:")
        logger.info("  1. Scope (Tactical → Strategic)")
        logger.info("  2. Temporality (Immediate → Evolutionary)")
        logger.info("  3. Ease of Adoption")
        logger.info("  4. Resources Required")
        logger.info("  5. Technical Complexity")
        logger.info("  6. Change Management Difficulty")
        logger.info("  7. Impact Potential")
        logger.info("  8. Time to Value")
        logger.info("  9. Applicability (Narrow → Broad)")
        logger.info("\nEstimated time: ~45-60 minutes for 681 methods")
        logger.info("="*80)

        # Define all dimensions to rank
        dimensions = [
            'scope',
            'temporality',
            'ease_adoption',
            'resources',
            'complexity',
            'change_mgmt',
            'impact',
            'time_to_value',
            'applicability'
        ]

        # Rank each dimension
        all_scores = {}
        for i, dimension in enumerate(dimensions, 1):
            logger.info(f"\n[{i}/9] Ranking {dimension.upper()}...")
            dim_scores = await self.rank_dimension(methods, dimension)
            all_scores[dimension] = dim_scores

        # Combine results
        logger.info("\n" + "="*80)
        logger.info("CALCULATING DERIVED METRICS")
        logger.info("="*80)

        results = {}
        for method in methods:
            idx = method.index

            # Get scores for all dimensions
            scope = all_scores['scope'].get(idx, 50)
            temporality = all_scores['temporality'].get(idx, 50)
            ease_adoption = all_scores['ease_adoption'].get(idx, 50)
            resources = all_scores['resources'].get(idx, 50)
            complexity = all_scores['complexity'].get(idx, 50)
            change_mgmt = all_scores['change_mgmt'].get(idx, 50)
            impact = all_scores['impact'].get(idx, 50)
            time_to_value = all_scores['time_to_value'].get(idx, 50)
            applicability = all_scores['applicability'].get(idx, 50)

            # Calculate Implementation Difficulty as average of difficulty metrics
            # Higher score = more difficult
            # Note: ease_adoption is inverted (high = easy), others measure difficulty directly
            implementation_difficulty = np.mean([
                100 - ease_adoption,    # Invert: low ease → high difficulty
                resources,              # High resources → high difficulty
                complexity,             # High complexity → high difficulty
                change_mgmt             # High change difficulty → high difficulty
            ])

            results[idx] = {
                # Core dimensions
                'scope': scope,
                'temporality': temporality,
                'ease_adoption': ease_adoption,
                'resources_required': resources,  # Now directly from ranking (low→high resources)
                'technical_complexity': complexity,  # Now directly from ranking (low→high complexity)
                'change_management_difficulty': change_mgmt,  # Now directly from ranking (low→high difficulty)
                'impact_potential': impact,
                'time_to_value': time_to_value,
                'applicability': applicability,

                # Derived metric
                'implementation_difficulty': implementation_difficulty,

                # For backward compatibility with visualization (0-1 scale)
                'scope_score': scope / 100.0,
                'temporality_score': temporality / 100.0,

                # Metadata
                'reasoning': f"Comprehensive 9D ranking: S={scope:.0f} T={temporality:.0f} I={implementation_difficulty:.0f} P={impact:.0f}"
            }

        logger.info(f"\n✅ Comprehensive 9D ranking complete for {len(results)} methods")
        logger.info("\nDimension ranges:")
        logger.info("  All scores: 0-100 scale")
        logger.info("  Implementation Difficulty: Average of 4 sub-dimensions")

        return results

    async def batch_analyze_12d_ranked(self,
                                     methods: List[Method]) -> Dict[int, Dict]:
        """
        Comprehensive 12-dimensional analysis using ranking approach.
        Extends 9D with People × Process × Purpose framework.

        Ranks all methods on 12 dimensions:
        1. Scope (Tactical → Strategic)
        2. Temporality (Immediate → Evolutionary)
        3. Ease of Adoption
        4. Resources Required (inverted to "low resources needed")
        5. Technical Complexity (inverted to "low complexity")
        6. Change Management (inverted to "low change resistance")
        7. Impact Potential
        8. Time to Value (inverted to "fast delivery")
        9. Applicability (Narrow → Broad)
        10. People Focus (Technical → Human)
        11. Process Focus (Ad-hoc → Systematic)
        12. Purpose Orientation (Internal → External)

        Also calculates:
        - Implementation Difficulty = average of (100 - ease_adoption, 100 - resources, 100 - complexity, 100 - change_mgmt)

        Returns:
            Dictionary mapping method index to all dimension scores (0-100)
        """
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE 12-DIMENSIONAL RANKING ANALYSIS")
        logger.info("="*80)
        logger.info("\nThis will rank methods on 12 separate dimensions:")
        logger.info("  1. Scope (Tactical → Strategic)")
        logger.info("  2. Temporality (Immediate → Evolutionary)")
        logger.info("  3. Ease of Adoption")
        logger.info("  4. Resources Required")
        logger.info("  5. Technical Complexity")
        logger.info("  6. Change Management Difficulty")
        logger.info("  7. Impact Potential")
        logger.info("  8. Time to Value")
        logger.info("  9. Applicability (Narrow → Broad)")
        logger.info("  10. People Focus (Technical → Human)")
        logger.info("  11. Process Focus (Ad-hoc → Systematic)")
        logger.info("  12. Purpose Orientation (Internal → External)")
        logger.info("\nEstimated time: ~75-100 minutes for 595 methods (with 5-pass validation, 20 concurrent)")
        logger.info("="*80)

        # Define all dimensions to rank
        dimensions = [
            'scope',
            'temporality',
            'ease_adoption',
            'resources',
            'complexity',
            'change_mgmt',
            'impact',
            'time_to_value',
            'applicability',
            'people_focus',
            'process_focus',
            'purpose_orientation'
        ]

        # Rank each dimension
        all_scores = {}
        total_dimensions = len(dimensions)
        start_time = time.time()

        for i, dimension in enumerate(dimensions, 1):
            dim_start = time.time()

            # Progress header
            print(f"\n{'='*80}")
            print(f"DIMENSION {i}/{total_dimensions}: {dimension.upper().replace('_', ' ')}")
            print(f"{'='*80}")

            logger.info(f"\n[{i}/{total_dimensions}] Ranking {dimension.upper()}...")
            dim_scores = await self.rank_dimension(methods, dimension)
            all_scores[dimension] = dim_scores

            # Show completion and timing
            dim_elapsed = time.time() - dim_start
            total_elapsed = time.time() - start_time
            avg_time_per_dim = total_elapsed / i
            remaining_dims = total_dimensions - i
            estimated_remaining = avg_time_per_dim * remaining_dims

            print(f"\n✓ Completed {dimension.upper()} in {dim_elapsed:.1f}s")
            print(f"  Progress: {i}/{total_dimensions} dimensions ({i/total_dimensions*100:.1f}%)")
            print(f"  Elapsed: {total_elapsed/60:.1f} min | Estimated remaining: {estimated_remaining/60:.1f} min")
            print(f"{'='*80}\n")

        # Combine results
        logger.info("\n" + "="*80)
        logger.info("CALCULATING DERIVED METRICS")
        logger.info("="*80)

        results = {}
        for method in methods:
            idx = method.index

            # Get scores for all dimensions
            scope = all_scores['scope'].get(idx, 50)
            temporality = all_scores['temporality'].get(idx, 50)
            ease_adoption = all_scores['ease_adoption'].get(idx, 50)
            resources = all_scores['resources'].get(idx, 50)
            complexity = all_scores['complexity'].get(idx, 50)
            change_mgmt = all_scores['change_mgmt'].get(idx, 50)
            impact = all_scores['impact'].get(idx, 50)
            time_to_value = all_scores['time_to_value'].get(idx, 50)
            applicability = all_scores['applicability'].get(idx, 50)
            people_focus = all_scores['people_focus'].get(idx, 50)
            process_focus = all_scores['process_focus'].get(idx, 50)
            purpose_orientation = all_scores['purpose_orientation'].get(idx, 50)

            # Calculate Implementation Difficulty as average of difficulty metrics
            # Higher score = more difficult
            # Note: ease_adoption is inverted (high = easy), others measure difficulty directly
            implementation_difficulty = np.mean([
                100 - ease_adoption,    # Invert: low ease → high difficulty
                resources,              # High resources → high difficulty
                complexity,             # High complexity → high difficulty
                change_mgmt             # High change difficulty → high difficulty
            ])

            results[idx] = {
                # Core 9 dimensions
                'scope': scope,
                'temporality': temporality,
                'ease_adoption': ease_adoption,
                'resources_required': resources,  # Direct from ranking (low→high resources)
                'technical_complexity': complexity,  # Direct from ranking (low→high complexity)
                'change_management_difficulty': change_mgmt,  # Direct from ranking (low→high difficulty)
                'impact_potential': impact,
                'time_to_value': time_to_value,
                'applicability': applicability,

                # People × Process × Purpose dimensions
                'people_focus': people_focus,
                'process_focus': process_focus,
                'purpose_orientation': purpose_orientation,

                # Derived metric (can be reconstructed from stored dimensions)
                'implementation_difficulty': implementation_difficulty,

                # For backward compatibility with visualization (0-1 scale)
                'scope_score': scope / 100.0,
                'temporality_score': temporality / 100.0,

                # Metadata
                'reasoning': f"Comprehensive 12D ranking: S={scope:.0f} T={temporality:.0f} I={implementation_difficulty:.0f} P={impact:.0f} PPP=({people_focus:.0f}/{process_focus:.0f}/{purpose_orientation:.0f})"
            }

        logger.info(f"\n✅ Comprehensive 12D ranking complete for {len(results)} methods")
        logger.info("\nDimension ranges:")
        logger.info("  All scores: 0-100 scale")
        logger.info("  Implementation Difficulty: Average of 4 sub-dimensions")
        logger.info("  People × Process × Purpose: 3 new dimensions added")

        return results
