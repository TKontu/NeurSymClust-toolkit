"""
LLM-based duplicate synthesis: select best name and create unified description.
"""
import asyncio
import json
import logging
from typing import List, Dict, Tuple
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class DuplicateSynthesizer:
    """Uses LLM to select best name and synthesize descriptions for duplicate groups"""

    def __init__(self, config: dict):
        self.config = config['llm']
        self.base_url = self.config['base_url']
        self.api_key = self.config['api_key']
        self.model = self.config['model']
        self.temperature = 0.1  # Low temperature for consistent choices
        self.timeout = self.config['timeout']
        self.max_concurrent = self.config['max_concurrent']

        self.prompt_template = self._create_prompt_template()

    def _create_prompt_template(self) -> str:
        """Create prompt for name selection and description synthesis"""
        return """You are synthesizing duplicate product development methods into one canonical representation.

TASK:
1. Select the BEST method name (most commonly known, clear, standard terminology)
2. Create a UNIFIED description that captures the essence of all variants

DUPLICATE GROUP:
{methods_info}

SELECTION CRITERIA:
- Choose the most widely recognized name in industry
- Prefer standard terminology over proprietary names
- Prefer simpler, clearer names over complex ones
- The unified description should capture all key aspects from variants
- Keep description concise (2-3 sentences max)

OUTPUT FORMAT (JSON):
{{
  "selected_name": "The best method name",
  "unified_description": "A synthesized description capturing all variants",
  "reasoning": "Brief explanation of why this name was chosen"
}}

JSON Response:"""

    def _create_batch_prompt_template(self) -> str:
        """Create prompt for batched synthesis of multiple groups"""
        return """You are synthesizing multiple duplicate product development method groups into canonical representations.

TASK: For each group below, select the BEST method name and create a UNIFIED description.

{batch_groups_info}

SELECTION CRITERIA:
- Choose the most widely recognized name in industry
- Prefer standard terminology over proprietary names
- Prefer simpler, clearer names over complex ones
- The unified description should capture all key aspects from variants
- Keep description concise (2-3 sentences max)

OUTPUT FORMAT (JSON array):
[
  {{
    "group_id": 0,
    "selected_name": "The best method name",
    "unified_description": "A synthesized description capturing all variants",
    "reasoning": "Brief explanation of why this name was chosen"
  }},
  {{
    "group_id": 1,
    ...
  }}
]

Return ONLY the JSON array, no additional text.

JSON Response:"""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    async def _call_llm(self, prompt: str, max_tokens: int = 400) -> str:
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
                    return data['choices'][0]['message']['content'].strip()

        except asyncio.TimeoutError:
            logger.error(f"LLM call timeout after {self.timeout}s")
            raise
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    def _parse_response(self, response: str, methods: List[Dict]) -> Dict:
        """Parse LLM response to extract name and description"""
        import json
        import re

        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    'name': result.get('selected_name', methods[0]['name']),
                    'description': result.get('unified_description', methods[0]['description']),
                    'reasoning': result.get('reasoning', 'LLM selection')
                }
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")

        # Fallback: use first method
        return {
            'name': methods[0]['name'],
            'description': methods[0]['description'],
            'reasoning': 'Fallback to first method'
        }

    async def synthesize_duplicate_group(self,
                                        methods: List[Dict],
                                        group_id: int) -> Dict:
        """
        Synthesize a group of duplicate methods into one canonical version.

        Args:
            methods: List of duplicate method dicts with 'name', 'description', 'source'
            group_id: Group identifier

        Returns:
            Dict with synthesized 'name', 'description', 'reasoning'
        """
        if len(methods) == 1:
            # No duplicates, return as-is
            return {
                'name': methods[0]['name'],
                'description': methods[0]['description'],
                'reasoning': 'Single method, no synthesis needed',
                'source': methods[0]['source']
            }

        # Format methods for prompt
        methods_info = ""
        for i, method in enumerate(methods, 1):
            methods_info += f"\n{i}. NAME: {method['name']}\n"
            methods_info += f"   SOURCE: {method['source']}\n"
            methods_info += f"   DESCRIPTION: {method['description']}\n"

        # Create prompt
        prompt = self.prompt_template.format(methods_info=methods_info)

        logger.info(f"  Synthesizing group {group_id}: {len(methods)} methods")

        # Call LLM
        response = await self._call_llm(prompt)

        # Parse response
        result = self._parse_response(response, methods)

        # Add metadata
        result['source'] = self._select_best_source(methods)
        result['original_count'] = len(methods)
        result['original_names'] = [m['name'] for m in methods]

        return result

    def _select_best_source(self, methods: List[Dict]) -> str:
        """Select the best source (prefer shorter, more authoritative)"""
        sources = [m['source'] for m in methods]
        # Prefer shorter sources (often more authoritative)
        return min(sources, key=len)

    async def synthesize_batch(self,
                              batch_groups: Dict[int, List[Dict]]) -> Dict[int, Dict]:
        """
        Synthesize multiple groups in a single LLM call for better GPU utilization.

        Args:
            batch_groups: Dict mapping group_id to list of duplicate methods

        Returns:
            Dict mapping group_id to synthesized method
        """
        if not batch_groups:
            return {}

        # Format all groups for batch prompt
        batch_groups_info = ""
        for group_id, methods in batch_groups.items():
            batch_groups_info += f"\n=== GROUP {group_id} ===\n"
            for i, method in enumerate(methods, 1):
                batch_groups_info += f"{i}. NAME: {method['name']}\n"
                batch_groups_info += f"   SOURCE: {method['source']}\n"
                batch_groups_info += f"   DESCRIPTION: {method['description']}\n"

        # Create batch prompt
        batch_template = self._create_batch_prompt_template()
        prompt = batch_template.format(batch_groups_info=batch_groups_info)

        logger.info(f"  Synthesizing batch of {len(batch_groups)} groups in single LLM call...")

        # Call LLM with larger max_tokens for batch response
        # Estimate ~200 tokens per group (name + description + reasoning)
        max_tokens = min(4000, len(batch_groups) * 200 + 200)

        try:
            response = await self._call_llm(prompt, max_tokens=max_tokens)

            # Parse JSON array response
            response_clean = response.strip()
            if response_clean.startswith('```'):
                # Remove markdown code blocks
                lines = response_clean.split('\n')
                response_clean = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_clean

            results_array = json.loads(response_clean)

            # Convert array to dict keyed by group_id
            results = {}
            for result in results_array:
                group_id = result['group_id']
                methods = batch_groups[group_id]

                results[group_id] = {
                    'name': result['selected_name'],
                    'description': result['unified_description'],
                    'reasoning': result.get('reasoning', 'Batch synthesis'),
                    'source': self._select_best_source(methods),
                    'original_count': len(methods),
                    'original_names': [m['name'] for m in methods]
                }

            return results

        except Exception as e:
            logger.warning(f"Batch synthesis failed: {e}. Falling back to individual synthesis.")
            # Fallback: synthesize individually
            results = {}
            for group_id, methods in batch_groups.items():
                results[group_id] = await self.synthesize_duplicate_group(methods, group_id)
            return results

    async def batch_synthesize_groups(self,
                                     groups: Dict[int, List[Dict]],
                                     batch_size: int = 10,
                                     max_concurrent: int = 4) -> Dict[int, Dict]:
        """
        Synthesize multiple duplicate groups using aggressive batching.

        Instead of making 60 individual LLM calls, this batches multiple groups
        into single prompts (e.g., 10 groups per call = only 6 API calls total).
        This dramatically improves GPU utilization on the vLLM server.

        Args:
            groups: Dict mapping group_id to list of duplicate methods
            batch_size: Number of groups to synthesize per LLM call (default 10)
            max_concurrent: Max parallel batch calls (default 4)

        Returns:
            Dict mapping group_id to synthesized method
        """
        logger.info(f"\nSynthesizing {len(groups)} duplicate groups with aggressive batching...")
        logger.info(f"  Batch size: {batch_size} groups per LLM call")
        logger.info(f"  Max concurrent: {max_concurrent} parallel batch calls")

        # Split groups into chunks for batching
        group_items = list(groups.items())
        batches = []
        for i in range(0, len(group_items), batch_size):
            batch = dict(group_items[i:i + batch_size])
            batches.append(batch)

        num_batches = len(batches)
        logger.info(f"  Total LLM calls: {num_batches} (down from {len(groups)})")

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        async def synthesize_batch_with_semaphore(batch_idx: int, batch_groups: Dict):
            async with semaphore:
                logger.info(f"  Processing batch {batch_idx + 1}/{num_batches}...")
                return await self.synthesize_batch(batch_groups)

        # Create tasks for all batches
        tasks = []
        for batch_idx, batch_groups in enumerate(batches):
            task = synthesize_batch_with_semaphore(batch_idx, batch_groups)
            tasks.append((batch_idx, task))

        # Execute all batch tasks
        all_results = {}
        completed_batches = 0

        for batch_idx, task in tasks:
            try:
                batch_results = await task
                all_results.update(batch_results)
                completed_batches += 1
                logger.info(f"  ✓ Batch {completed_batches}/{num_batches} complete ({len(batch_results)} groups)")

            except Exception as e:
                logger.error(f"  Failed to synthesize batch {batch_idx}: {e}")
                # Fallback: use first method from each group in batch
                batch_groups = batches[batch_idx]
                for group_id, methods in batch_groups.items():
                    all_results[group_id] = {
                        'name': methods[0]['name'],
                        'description': methods[0]['description'],
                        'source': methods[0]['source'],
                        'reasoning': f'Batch synthesis failed: {str(e)}',
                        'original_count': len(methods),
                        'original_names': [m['name'] for m in methods]
                    }

        logger.info(f"✓ Completed synthesis of {len(all_results)} groups using {num_batches} LLM calls")
        return all_results
