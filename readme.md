# Product Development Methods Analysis Tool

A practical tool for analyzing 800+ product development methods to identify duplicates, assess compatibility, and build effective method toolkits.

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Configure servers (copy your existing config)
cp config.example.yaml config.yaml

# Run analysis
python analyze_methods.py --input methods.csv --output results/

# Generate report
python generate_report.py results/
```

## What It Does

1. **Duplicate Detection**: Finds methods that are essentially the same
2. **Compatibility Analysis**: Identifies which methods work well together
3. **Abstraction Check**: Ensures methods are at comparable levels
4. **Toolkit Generation**: Suggests optimal method combinations
5. **Overlap Assessment**: Maps problem space coverage

## Configuration

Uses same config structure as your existing project:
- LLM: Gemma3-12b-awq at `http://192.168.0.247:9003/v1`
- Embeddings: BGE-large-en at `http://192.168.0.136:9003/v1`
- Reranking: BGE-reranker-v2-m3 (same server)

## Input Format

CSV with columns: Index, Method, Description, Source

## Output

- `duplicates.json`: Grouped duplicate methods
- `compatibility_matrix.pkl`: Pairwise compatibility scores
- `toolkits.json`: Recommended method combinations
- `report.html`: Visual summary with charts

## Performance

- ~2-4 hours for 800 methods (with caching)
- Parallel processing: 25 embeddings/batch, 4-16 LLM calls concurrent
- Smart sampling reduces comparisons by ~90%