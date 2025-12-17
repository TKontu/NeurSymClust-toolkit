"""
LLM-based analysis of method relationships and properties.
"""
import asyncio
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from src.data import Method

logger = logging.getLogger(__name__)


class LLMAnalyzer:
    """Handles LLM-based analysis with async batch processing."""

    def __init__(self, config: dict, prompts_dir: str = "./prompts"):
        self.config = config['llm']
        self.base_url = self.config['base_url']
        self.api_key = self.config['api_key']
        self.model = self.config['model']
        self.temperature = self.config['temperature']
        self.timeout = self.config['timeout']
        self.max_concurrent = self.config['max_concurrent']

        self.prompts_dir = Path(prompts_dir)
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Dict[str, str]:
        """Load analysis prompts from files."""
        prompts = {}
        prompt_files = {
            'duplicate': 'duplicate.txt',
            'compatibility': 'compatibility.txt',
            'abstraction': 'abstraction.txt',
            'effectiveness': 'effectiveness.txt',
            'overlap': 'overlap.txt',
            'categorization': 'categorization.txt',
            'category_fit': 'category_fit.txt',
            'scope_temporality': 'scope_temporality.txt'
        }

        missing_prompts = []
        for key, filename in prompt_files.items():
            filepath = self.prompts_dir / filename
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    prompts[key] = f.read().strip()
            else:
                missing_prompts.append(str(filepath))

        # Fail fast if critical prompts are missing
        if missing_prompts:
            raise FileNotFoundError(
                f"Required prompt files not found:\n" +
                "\n".join(f"  - {p}" for p in missing_prompts)
            )

        return prompts

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60))
    async def _call_llm(self, prompt: str, max_tokens: int = 500) -> str:
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
                    return data['choices'][0]['message']['content'].strip()

        except asyncio.TimeoutError:
            logger.error(f"LLM call timeout after {self.timeout}s")
            raise
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    async def analyze_duplicate(self, method1: Method, method2: Method) -> Dict:
        """
        Analyze if two methods are duplicates (same technique, different names).
        Returns: {"is_duplicate": bool, "confidence": float, "reasoning": str}
        """
        prompt = self.prompts['duplicate'].format(
            method1_name=method1.name,
            method1_desc=method1.description,
            method2_name=method2.name,
            method2_desc=method2.description
        )

        try:
            response = await self._call_llm(prompt, max_tokens=300)
            return self._parse_duplicate_response(response)
        except Exception as e:
            logger.error(f"Failed to analyze duplicate for {method1.name} vs {method2.name}: {e}")
            return {"is_duplicate": False, "confidence": 0.0, "reasoning": f"Error: {e}"}

    async def analyze_compatibility(self, method1: Method, method2: Method) -> Dict:
        """
        Analyze if two methods are compatible (complementary, address different aspects).
        Returns: {"compatibility_score": float (0-1), "reasoning": str}
        """
        prompt = self.prompts['compatibility'].format(
            method1_name=method1.name,
            method1_desc=method1.description,
            method2_name=method2.name,
            method2_desc=method2.description
        )

        try:
            response = await self._call_llm(prompt, max_tokens=400)
            return self._parse_compatibility_response(response)
        except Exception as e:
            logger.error(f"Failed to analyze compatibility for {method1.name} vs {method2.name}: {e}")
            return {"compatibility_score": 0.5, "reasoning": f"Error: {e}"}

    async def analyze_abstraction(self, method: Method) -> Dict:
        """
        Analyze abstraction level (general principle vs specific technique).
        Returns: {"level": str ("high"|"medium"|"low"), "reasoning": str}
        """
        prompt = self.prompts['abstraction'].format(
            method_name=method.name,
            method_desc=method.description
        )

        try:
            response = await self._call_llm(prompt, max_tokens=300)
            return self._parse_abstraction_response(response)
        except Exception as e:
            logger.error(f"Failed to analyze abstraction for {method.name}: {e}")
            return {"level": "medium", "reasoning": f"Error: {e}"}

    async def analyze_category(self, method: Method, categories: List[Dict]) -> Dict:
        """
        Categorize a method into one of the predefined categories.
        Returns: {"category_id": str, "category_name": str, "reasoning": str}
        """
        # Format categories list
        categories_list = "\n".join([
            f"- {cat['id']}: {cat['name']} - {cat['description']}"
            for cat in categories
        ])

        prompt = self.prompts['categorization'].format(
            method_name=method.name,
            method_desc=method.description,
            categories_list=categories_list
        )

        try:
            response = await self._call_llm(prompt, max_tokens=300)
            return self._parse_category_response(response, categories)
        except Exception as e:
            logger.error(f"Failed to categorize {method.name}: {e}")
            return {
                "category_id": "planning_adaptation",
                "category_name": "Planning and Adaptation",
                "reasoning": f"Error: {e}"
            }

    async def analyze_category_fit(self, method: Method, assigned_category: Dict,
                                   all_categories: List[Dict]) -> Dict:
        """
        Analyze how well a method fits its assigned category.
        Returns: {
            "fit_score": int (1-10),
            "fit_level": str,
            "alternative_category": str or None,
            "missing_dimension": bool,
            "dimension_description": str,
            "category_suggestion": str,
            "reasoning": str
        }
        """
        # Format all categories list
        categories_list = "\n".join([
            f"- {cat['id']}: {cat['name']} - {cat['description']}"
            for cat in all_categories
        ])

        prompt = self.prompts['category_fit'].format(
            method_name=method.name,
            method_description=method.description,
            assigned_category_name=assigned_category['name'],
            assigned_category_description=assigned_category['description'],
            all_categories=categories_list
        )

        try:
            response = await self._call_llm(prompt, max_tokens=500)
            return self._parse_category_fit_response(response)
        except Exception as e:
            logger.error(f"Failed to analyze category fit for {method.name}: {e}")
            return {
                "fit_score": 7,
                "fit_level": "good",
                "alternative_category": None,
                "missing_dimension": False,
                "dimension_description": "none",
                "category_suggestion": "none",
                "reasoning": f"Error: {e}"
            }

    async def batch_analyze_duplicates(self, pairs: List[Tuple[Method, Method, float]]) -> List[Dict]:
        """
        Analyze multiple method pairs for duplicates with concurrency control.
        pairs: List of (method1, method2, similarity_score)
        """
        logger.info(f"Analyzing {len(pairs)} pairs for duplicates (concurrent={self.max_concurrent})")

        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def analyze_with_sem(pair):
            async with semaphore:
                method1, method2, sim_score = pair
                result = await self.analyze_duplicate(method1, method2)
                result['method1_index'] = method1.index
                result['method2_index'] = method2.index
                result['method1_name'] = method1.name
                result['method2_name'] = method2.name
                result['embedding_similarity'] = sim_score
                return result

        tasks = [analyze_with_sem(pair) for pair in pairs]
        results = []

        # Use tqdm for progress tracking
        with tqdm(total=len(tasks), desc="Analyzing duplicates") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                pbar.update(1)

        return results

    async def batch_analyze_compatibility(self, pairs: List[Tuple[Method, Method]]) -> List[Dict]:
        """
        Analyze multiple method pairs for compatibility with concurrency control.
        """
        logger.info(f"Analyzing {len(pairs)} pairs for compatibility (concurrent={self.max_concurrent})")

        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def analyze_with_sem(pair):
            async with semaphore:
                method1, method2 = pair
                result = await self.analyze_compatibility(method1, method2)
                result['method1_index'] = method1.index
                result['method2_index'] = method2.index
                result['method1_name'] = method1.name
                result['method2_name'] = method2.name
                return result

        tasks = [analyze_with_sem(pair) for pair in pairs]
        results = []

        with tqdm(total=len(tasks), desc="Analyzing compatibility") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                pbar.update(1)

        return results

    async def batch_analyze_abstraction(self, methods: List[Method]) -> Dict[int, Dict]:
        """
        Analyze abstraction levels for multiple methods with concurrency control.
        Returns dict mapping method index to analysis result.
        """
        logger.info(f"Analyzing abstraction levels for {len(methods)} methods (concurrent={self.max_concurrent})")

        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def analyze_with_sem(method):
            async with semaphore:
                result = await self.analyze_abstraction(method)
                return method.index, result

        tasks = [analyze_with_sem(method) for method in methods]
        results = {}

        with tqdm(total=len(tasks), desc="Analyzing abstraction") as pbar:
            for coro in asyncio.as_completed(tasks):
                method_idx, result = await coro
                results[method_idx] = result
                pbar.update(1)

        return results

    async def batch_analyze_categories(self, methods: List[Method], categories: List[Dict]) -> Dict[int, Dict]:
        """
        Categorize multiple methods with concurrency control.
        Returns dict mapping method index to category result.
        """
        logger.info(f"Categorizing {len(methods)} methods (concurrent={self.max_concurrent})")

        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def analyze_with_sem(method):
            async with semaphore:
                result = await self.analyze_category(method, categories)
                return method.index, result

        tasks = [analyze_with_sem(method) for method in methods]
        results = {}

        with tqdm(total=len(tasks), desc="Categorizing methods") as pbar:
            for coro in asyncio.as_completed(tasks):
                method_idx, result = await coro
                results[method_idx] = result
                pbar.update(1)

        return results

    async def batch_analyze_category_fit(self, methods: List[Method],
                                         category_assignments: Dict[int, Dict],
                                         categories: List[Dict]) -> Dict[int, Dict]:
        """
        Analyze category fit for multiple methods with concurrency control.

        Args:
            methods: List of methods to analyze
            category_assignments: Dict mapping method index to assigned category result
            categories: List of all category definitions

        Returns:
            Dict mapping method index to category fit result
        """
        logger.info(f"Analyzing category fit for {len(methods)} methods (concurrent={self.max_concurrent})")

        # Build category lookup
        category_lookup = {cat['id']: cat for cat in categories}

        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def analyze_with_sem(method):
            async with semaphore:
                # Get assigned category for this method
                assignment = category_assignments.get(method.index)
                if not assignment:
                    logger.warning(f"No category assignment found for method {method.index}")
                    return method.index, {
                        "fit_score": 5,
                        "fit_level": "moderate",
                        "alternative_category": None,
                        "missing_dimension": False,
                        "dimension_description": "No assignment found",
                        "category_suggestion": "none",
                        "reasoning": "Method was not categorized"
                    }

                category_id = assignment.get('category_id')
                assigned_category = category_lookup.get(category_id)

                if not assigned_category:
                    logger.warning(f"Unknown category {category_id} for method {method.index}")
                    return method.index, {
                        "fit_score": 5,
                        "fit_level": "moderate",
                        "alternative_category": None,
                        "missing_dimension": False,
                        "dimension_description": "Unknown category",
                        "category_suggestion": "none",
                        "reasoning": "Category not found in category definitions"
                    }

                result = await self.analyze_category_fit(method, assigned_category, categories)
                return method.index, result

        tasks = [analyze_with_sem(method) for method in methods]
        results = {}

        with tqdm(total=len(tasks), desc="Analyzing category fit") as pbar:
            for coro in asyncio.as_completed(tasks):
                method_idx, result = await coro
                results[method_idx] = result
                pbar.update(1)

        return results

    def _parse_duplicate_response(self, response: str) -> Dict:
        """Parse LLM response for duplicate analysis."""
        response_lower = response.lower()

        # Look for yes/no indicators
        is_duplicate = False
        if "yes" in response_lower[:50] or "duplicate" in response_lower[:100]:
            is_duplicate = True
        elif "no" in response_lower[:50] or "not duplicate" in response_lower[:100]:
            is_duplicate = False

        # Try to extract confidence
        confidence = 0.8 if is_duplicate else 0.2  # Default confidence
        if "high confidence" in response_lower or "definitely" in response_lower:
            confidence = 0.95
        elif "medium confidence" in response_lower or "likely" in response_lower:
            confidence = 0.75
        elif "low confidence" in response_lower or "possibly" in response_lower:
            confidence = 0.5

        return {
            "is_duplicate": is_duplicate,
            "confidence": confidence,
            "reasoning": response
        }

    def _parse_compatibility_response(self, response: str) -> Dict:
        """Parse LLM response for compatibility analysis."""
        response_lower = response.lower()

        # Try to extract score
        score = 0.5  # Default neutral
        if "highly compatible" in response_lower or "very compatible" in response_lower:
            score = 0.9
        elif "compatible" in response_lower and "not" not in response_lower[:100]:
            score = 0.7
        elif "somewhat compatible" in response_lower or "moderately" in response_lower:
            score = 0.6
        elif "incompatible" in response_lower or "conflict" in response_lower:
            score = 0.2

        return {
            "compatibility_score": score,
            "reasoning": response
        }

    def _parse_abstraction_response(self, response: str) -> Dict:
        """Parse LLM response for abstraction analysis."""
        response_lower = response.lower()

        # Determine level
        level = "medium"  # Default
        if any(word in response_lower[:200] for word in ["high", "general", "principle", "philosophy", "mindset"]):
            level = "high"
        elif any(word in response_lower[:200] for word in ["low", "specific", "concrete", "technique", "practice", "tool"]):
            level = "low"

        return {
            "level": level,
            "reasoning": response
        }

    def _parse_category_response(self, response: str, categories: List[Dict]) -> Dict:
        """Parse LLM response for categorization."""
        response_lower = response.lower()

        # Try to extract category ID
        category_id = None
        for cat in categories:
            if cat['id'] in response_lower[:100]:
                category_id = cat['id']
                break

        # Fallback: look for category name
        if not category_id:
            for cat in categories:
                if cat['name'].lower() in response_lower[:200]:
                    category_id = cat['id']
                    break

        # Default fallback
        if not category_id:
            category_id = "planning_adaptation"

        # Find category name
        category_name = next(
            (cat['name'] for cat in categories if cat['id'] == category_id),
            "Planning and Adaptation"
        )

        return {
            "category_id": category_id,
            "category_name": category_name,
            "reasoning": response
        }

    def _parse_category_fit_response(self, response: str) -> Dict:
        """Parse LLM response for category fit analysis."""
        lines = response.strip().split('\n')
        result = {
            "fit_score": 7,
            "fit_level": "good",
            "alternative_category": None,
            "missing_dimension": False,
            "dimension_description": "none",
            "category_suggestion": "none",
            "reasoning": ""
        }

        for line in lines:
            line = line.strip()
            if line.startswith('FIT_SCORE:'):
                try:
                    result['fit_score'] = int(line.split(':', 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith('FIT_LEVEL:'):
                result['fit_level'] = line.split(':', 1)[1].strip()
            elif line.startswith('ALTERNATIVE_CATEGORY:'):
                alt = line.split(':', 1)[1].strip()
                result['alternative_category'] = None if alt.lower() == 'none' else alt
            elif line.startswith('MISSING_DIMENSION:'):
                result['missing_dimension'] = 'yes' in line.lower()
            elif line.startswith('DIMENSION_DESCRIPTION:'):
                desc = line.split(':', 1)[1].strip()
                result['dimension_description'] = desc
            elif line.startswith('CATEGORY_SUGGESTION:'):
                sugg = line.split(':', 1)[1].strip()
                result['category_suggestion'] = sugg
            elif line.startswith('REASONING:'):
                result['reasoning'] = line.split(':', 1)[1].strip()

        return result

    async def analyze_scope_temporality(self, method: Method) -> Dict:
        """
        Analyze method's position on Scope × Temporality dimensions.
        Returns: {
            "scope_score": float (0-1),
            "temporality_score": float (0-1),
            "reasoning": str
        }
        """
        prompt = self.prompts['scope_temporality'].format(
            method_name=method.name,
            method_description=method.description
        )

        try:
            response = await self._call_llm(prompt, max_tokens=300)
            return self._parse_scope_temporality_response(response)
        except Exception as e:
            logger.error(f"Failed to analyze scope/temporality for {method.name}: {e}")
            return {
                "scope_score": 0.5,
                "temporality_score": 0.5,
                "reasoning": f"Error: {e}"
            }

    async def batch_analyze_scope_temporality(self, methods: List[Method]) -> Dict[int, Dict]:
        """
        Analyze scope/temporality for multiple methods with concurrency control.
        Returns dict mapping method index to analysis result.
        """
        logger.info(f"Analyzing scope/temporality for {len(methods)} methods (concurrent={self.max_concurrent})")

        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def analyze_with_sem(method):
            async with semaphore:
                result = await self.analyze_scope_temporality(method)
                return method.index, result

        tasks = [analyze_with_sem(method) for method in methods]
        results = {}

        with tqdm(total=len(tasks), desc="Analyzing scope/temporality") as pbar:
            for coro in asyncio.as_completed(tasks):
                method_idx, result = await coro
                results[method_idx] = result
                pbar.update(1)

        return results

    def _parse_scope_temporality_response(self, response: str) -> Dict:
        """Parse LLM response for scope/temporality scores (0-100 scale)."""
        lines = response.strip().split('\n')
        result = {
            "scope_score": 0.5,
            "temporality_score": 0.5,
            "reasoning": ""
        }

        for line in lines:
            line = line.strip()
            if line.startswith('SCOPE_SCORE:'):
                try:
                    score_str = line.split(':', 1)[1].strip()
                    # Parse as integer (0-100) and convert to 0.0-1.0
                    score_int = int(float(score_str))  # Handle "35.0" or "35"
                    result['scope_score'] = score_int / 100.0
                except (ValueError, IndexError):
                    pass
            elif line.startswith('TEMPORALITY_SCORE:'):
                try:
                    score_str = line.split(':', 1)[1].strip()
                    # Parse as integer (0-100) and convert to 0.0-1.0
                    score_int = int(float(score_str))  # Handle "42.0" or "42"
                    result['temporality_score'] = score_int / 100.0
                except (ValueError, IndexError):
                    pass
            elif line.startswith('REASONING:'):
                result['reasoning'] = line.split(':', 1)[1].strip()

        # Clamp scores to valid range [0.0, 1.0]
        result['scope_score'] = max(0.0, min(1.0, result['scope_score']))
        result['temporality_score'] = max(0.0, min(1.0, result['temporality_score']))

        return result
