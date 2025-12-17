"""
Embedding service with batch processing and similarity search using FAISS.
"""
import asyncio
import logging
from typing import List, Tuple, Dict
import numpy as np
import aiohttp
from tqdm import tqdm

try:
    import faiss
except ImportError:
    faiss = None
    logging.warning("FAISS not installed. Run: pip install faiss-cpu")

from src.data import Method

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Handles embedding generation with batch processing."""

    def __init__(self, config: dict):
        self.config = config['embedding']
        self.base_url = self.config['base_url']
        self.api_key = self.config['api_key']
        self.model = self.config['model']
        self.batch_size = self.config['batch_size']
        self.max_concurrent = self.config['max_concurrent']

        self.embeddings_cache: Dict[int, np.ndarray] = {}
        self.dimension = None  # Will be set after first embedding

    async def generate_embeddings(self, methods: List[Method]) -> Dict[int, np.ndarray]:
        """
        Generate embeddings for all methods with batch processing.
        Returns dict mapping method index to embedding vector.
        """
        logger.info(f"Generating embeddings for {len(methods)} methods (batch_size={self.batch_size})")

        # Create batches
        batches = [methods[i:i + self.batch_size]
                  for i in range(0, len(methods), self.batch_size)]

        # Process batches with concurrency limit
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_batch_with_sem(batch):
            async with semaphore:
                return await self._generate_batch(batch)

        # Process all batches
        tasks = [process_batch_with_sem(batch) for batch in batches]
        results = []

        # Use tqdm for progress tracking
        with tqdm(total=len(batches), desc="Embedding batches") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.extend(result)
                pbar.update(1)

        # Build cache dictionary
        for method, embedding in results:
            self.embeddings_cache[method.index] = embedding

        if self.embeddings_cache and self.dimension is None:
            self.dimension = len(next(iter(self.embeddings_cache.values())))
            logger.info(f"Embedding dimension: {self.dimension}")

        logger.info(f"Generated {len(self.embeddings_cache)} embeddings")
        return self.embeddings_cache

    async def _generate_batch(self, methods: List[Method]) -> List[Tuple[Method, np.ndarray]]:
        """Generate embeddings for a batch of methods."""
        texts = [m.get_text_for_embedding() for m in methods]

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/embeddings",
                    json={
                        "model": self.model,
                        "input": texts
                    },
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Embedding API error: {response.status} - {error_text}")

                    data = await response.json()
                    embeddings = [np.array(item['embedding'], dtype=np.float32)
                                 for item in data['data']]

                    return list(zip(methods, embeddings))

        except Exception as e:
            logger.error(f"Failed to generate embeddings for batch of {len(methods)}: {e}")
            # Fallback: try one by one
            if len(methods) > 1:
                logger.info("Retrying batch items individually...")
                results = []
                failed_methods = []
                for method in methods:
                    try:
                        single_result = await self._generate_batch([method])
                        results.extend(single_result)
                    except Exception as single_e:
                        logger.error(f"Failed to embed method {method.index} ({method.name}): {single_e}")
                        failed_methods.append(method.index)

                if failed_methods:
                    logger.warning(f"Failed to generate embeddings for {len(failed_methods)} methods: {failed_methods}")
                    # Only raise if more than 10% failed
                    if len(failed_methods) / len(methods) > 0.1:
                        raise Exception(f"Too many embedding failures: {len(failed_methods)}/{len(methods)}")

                return results
            raise

    def compute_similarity_matrix(self, methods: List[Method]) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix using embeddings.
        Returns N x N matrix where entry (i,j) is similarity between methods[i] and methods[j].
        """
        if not self.embeddings_cache:
            raise ValueError("No embeddings available. Call generate_embeddings() first.")

        # Validate all methods have embeddings
        missing = [m.index for m in methods if m.index not in self.embeddings_cache]
        if missing:
            raise ValueError(f"Missing embeddings for {len(missing)} methods (indices: {missing[:10]}...)")

        logger.info("Computing similarity matrix...")

        # Build embedding matrix
        embeddings = np.array([self.embeddings_cache[m.index] for m in methods])

        # Normalize embeddings for cosine similarity (add epsilon to prevent division by zero)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-8)  # Add epsilon for numerical stability

        # Compute cosine similarity: dot product of normalized vectors
        similarity_matrix = normalized @ normalized.T

        logger.info(f"Computed {similarity_matrix.shape[0]}x{similarity_matrix.shape[1]} similarity matrix")
        return similarity_matrix

    def find_similar_pairs(self, methods: List[Method], threshold: float = 0.85) -> List[Tuple[Method, Method, float]]:
        """
        Find all pairs of methods with similarity above threshold.
        Returns list of (method1, method2, similarity_score).
        """
        similarity_matrix = self.compute_similarity_matrix(methods)

        similar_pairs = []
        n = len(methods)

        for i in range(n):
            for j in range(i + 1, n):  # Only upper triangle (avoid duplicates)
                similarity = similarity_matrix[i, j]
                if similarity >= threshold:
                    similar_pairs.append((methods[i], methods[j], float(similarity)))

        similar_pairs.sort(key=lambda x: x[2], reverse=True)  # Sort by similarity
        logger.info(f"Found {len(similar_pairs)} pairs with similarity >= {threshold}")

        return similar_pairs

    def build_faiss_index(self, methods: List[Method]) -> 'faiss.IndexFlatIP':
        """
        Build FAISS index for efficient similarity search.
        Returns FAISS index object.
        """
        if faiss is None:
            raise ImportError("FAISS not available. Install with: pip install faiss-cpu")

        if not self.embeddings_cache:
            raise ValueError("No embeddings available. Call generate_embeddings() first.")

        logger.info("Building FAISS index...")

        # Build embedding matrix
        embeddings = np.array([self.embeddings_cache[m.index] for m in methods])

        # Normalize for cosine similarity (Inner Product on normalized vectors = cosine similarity)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-8)  # Add epsilon for numerical stability

        # Create FAISS index (Inner Product for cosine similarity)
        index = faiss.IndexFlatIP(self.dimension)
        index.add(normalized.astype(np.float32))

        logger.info(f"Built FAISS index with {index.ntotal} vectors")
        return index

    def search_similar(self, method: Method, methods: List[Method],
                      faiss_index, k: int = 10) -> List[Tuple[Method, float]]:
        """
        Find k most similar methods to the given method using FAISS.
        Returns list of (method, similarity_score) tuples.
        """
        if method.index not in self.embeddings_cache:
            raise ValueError(f"Method {method.index} not in embeddings cache")

        # Get and normalize query embedding
        query_embedding = self.embeddings_cache[method.index]
        norm = np.linalg.norm(query_embedding)
        normalized_query = (query_embedding / (norm + 1e-8)).reshape(1, -1).astype(np.float32)

        # Search
        similarities, indices = faiss_index.search(normalized_query, k + 1)  # +1 to exclude self

        # Build results (exclude the query method itself)
        results = []
        for idx, sim in zip(indices[0], similarities[0]):
            if methods[idx].index != method.index:  # Skip self
                results.append((methods[idx], float(sim)))

        return results[:k]
