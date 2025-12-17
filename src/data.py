"""
Data loading and validation for product development methods.
"""
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Method:
    """Represents a product development method."""
    index: int
    name: str
    description: str
    source: str

    def __hash__(self):
        return hash(self.index)

    def __eq__(self, other):
        if not isinstance(other, Method):
            return False
        return self.index == other.index

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "index": self.index,
            "name": self.name,
            "description": self.description,
            "source": self.source
        }

    def get_text_for_embedding(self) -> str:
        """Get combined text for embedding generation."""
        return f"{self.name}. {self.description}"


class MethodLoader:
    """Loads and validates product development methods from CSV."""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

    def load(self) -> List[Method]:
        """
        Load methods from CSV file.
        Expected format: Index|Method|Description|Source
        """
        methods = []

        with open(self.file_path, 'r', encoding='utf-8') as f:
            # Use pipe delimiter
            reader = csv.DictReader(f, delimiter='|')

            for row_num, row in enumerate(reader, start=2):  # start=2 because header is line 1
                try:
                    method = Method(
                        index=int(row['Index']),
                        name=row['Method'].strip(),
                        description=row['Description'].strip(),
                        source=row['Source'].strip()
                    )

                    # Basic validation
                    if not method.name:
                        logger.warning(f"Row {row_num}: Empty method name, skipping")
                        continue
                    if not method.description:
                        logger.warning(f"Row {row_num}: Empty description for {method.name}, skipping")
                        continue

                    methods.append(method)

                except (KeyError, ValueError) as e:
                    logger.error(f"Row {row_num}: Failed to parse - {e}")
                    continue

        logger.info(f"Loaded {len(methods)} methods from {self.file_path}")

        # Log source distribution
        sources = {}
        for method in methods:
            sources[method.source] = sources.get(method.source, 0) + 1

        logger.info(f"Methods by source: {sources}")

        return methods

    def validate(self, methods: List[Method]) -> bool:
        """Validate loaded methods for basic consistency."""
        if not methods:
            logger.error("No methods loaded")
            return False

        # Check for duplicate indices
        indices = [m.index for m in methods]
        if len(indices) != len(set(indices)):
            logger.warning("Duplicate indices found in data")

        # Check for very short descriptions (likely data quality issues)
        short_descriptions = [m for m in methods if len(m.description) < 50]
        if short_descriptions:
            logger.warning(f"{len(short_descriptions)} methods have very short descriptions (< 50 chars)")

        return True


def load_methods(file_path: str) -> List[Method]:
    """Convenience function to load and validate methods."""
    loader = MethodLoader(file_path)
    methods = loader.load()
    loader.validate(methods)
    return methods
