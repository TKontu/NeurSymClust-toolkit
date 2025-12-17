# Deduplication Architecture

## Overview

The deduplication pipeline transforms the raw methods dataset (`methods.csv`, 681 methods) into a clean, deduplicated dataset (`methods_deduplicated.csv`, 595 methods), removing 86 duplicate entries (12.6% reduction). This process combines semantic embedding similarity with LLM-based description synthesis.

## Data Flow

```
┌──────────────────────────────────────┐
│         input/methods.csv            │
│         681 methods                  │
│         Format: Index|Method|        │
│                 Description|Source   │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 1: EMBEDDING GENERATION & DUPLICATE DETECTION     │
│  (Prerequisite: Full analysis pipeline)                  │
│                                                          │
│  Uses: src/embeddings.py, src/analyzer.py               │
│  Model: bge-large-en (embedding)                        │
│  Server: http://192.168.0.136:9003/v1                   │
│                                                          │
│  Output: results/duplicates.json                        │
│          - 2,596 pairs analyzed                          │
│          - 158 duplicates found                          │
│          - 73 duplicate groups                           │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 2: TRANSITIVE CLOSURE & GROUP FORMATION           │
│  filter_methods_for_analysis.py                          │
│                                                          │
│  Method:                                                 │
│  • Build graph from duplicate pairs                      │
│  • Edges: embedding similarity > 0.8                     │
│  • Find connected components (transitive closure)        │
│                                                          │
│  Result: 60 duplicate groups identified                  │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 3: LLM-BASED SYNTHESIS                            │
│  src/duplicate_synthesizer.py                            │
│                                                          │
│  Model: gpt-oss-20b (local VLLM server)                 │
│  Server: http://192.168.0.247:9003/v1                   │
│  Temperature: 0.1 (low for consistency)                  │
│                                                          │
│  For each duplicate group:                               │
│  • Select best method name                               │
│  • Synthesize unified description                        │
│  • Batch processing: 10 groups per LLM call              │
│                                                          │
│  Result: 60 groups synthesized                           │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│    input/methods_deduplicated.csv    │
│    595 unique methods                │
│    (Original indices preserved)      │
└──────────────────────────────────────┘
```

## Components

### 1. Embedding Generation (`src/embeddings.py`)

**Purpose**: Generate semantic embeddings for all methods to enable similarity-based duplicate detection.

**Technology Stack**:
- Embedding model: `bge-large-en`
- API: OpenAI-compatible (local server at `http://192.168.0.136:9003/v1`)
- Batch size: 25 methods per request
- Max concurrent requests: 50

**Algorithm**:
```python
# Generate embeddings for all methods
embeddings = await embedding_service.generate_embeddings(methods)

# Compute pairwise similarity matrix
similarity_matrix = embedding_service.compute_similarity_matrix(methods)
```

**Text Representation**:
Each method is embedded using concatenated name and description:
```python
def get_text_for_embedding(self) -> str:
    return f"{self.name}. {self.description}"
```

### 2. Duplicate Pair Detection (`src/analyzer.py`)

**Purpose**: Use LLM to verify high-similarity pairs as actual duplicates.

**Prompt Template** (`prompts/duplicate.txt`):
```
You are analyzing product development methods to identify duplicates.

Two methods are considered DUPLICATES if they are essentially the same
technique or practice but with different names. This means:
- They describe the same core process or approach
- They have the same purpose and outcomes
- The main difference is terminology or naming convention
- Example: "Daily Standup" vs "Daily Scrum Meeting"

Two methods are NOT duplicates if:
- They are similar but have different specific implementations
- They address different aspects of a problem
- One is more general/specific than the other

**Method 1:**
Name: {method1_name}
Description: {method1_desc}

**Method 2:**
Name: {method2_name}
Description: {method2_desc}

Answer with:
1. YES or NO (first line)
2. Confidence level: HIGH, MEDIUM, or LOW
3. Brief reasoning (2-3 sentences)
```

**Output**: `results/duplicates.json` containing:
- 2,596 pairs analyzed
- 158 confirmed duplicates
- 73 duplicate groups via transitive closure

### 3. Transitive Closure (`filter_methods_for_analysis.py`)

**Purpose**: Group all semantically similar methods using graph connectivity.

**Algorithm**:
```python
# Build graph where edges exist if embedding similarity > 0.8
G = nx.Graph()
for pair in duplicate_pairs:
    if pair['embedding_similarity'] > 0.8:
        G.add_edge(pair['method1_index'], pair['method2_index'])

# Find connected components (transitive closure)
groups = list(nx.connected_components(G))
```

**Key Characteristics**:
- Similarity threshold: 0.8 (cosine similarity)
- Uses NetworkX for graph operations
- Transitive: if A~B and B~C, then {A, B, C} form one group

**Result**: 60 duplicate groups formed from the original 73 groups

### 4. Description Synthesis (`src/duplicate_synthesizer.py`)

**Purpose**: For each duplicate group, select the best canonical name and create a unified description.

**LLM Configuration**:
- Model: `gpt-oss-20b` (from `config.yaml`)
- Server: `http://192.168.0.247:9003/v1` (local VLLM)
- Temperature: 0.1 (low for consistency)
- Max tokens: 400 per response

**Prompt Template**:
```
You are synthesizing duplicate product development methods into one canonical representation.

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
{
  "selected_name": "The best method name",
  "unified_description": "A synthesized description capturing all variants",
  "reasoning": "Brief explanation of why this name was chosen"
}
```

**Batch Processing**:
- Groups per LLM call: 10
- Max concurrent batch calls: 4
- Total API calls: ~6 for 60 groups (vs 60 individual calls)

**Merge Strategy**:
- Keep the first occurrence's index as representative
- Replace name and description with synthesized version
- Preserve source from the best matching variant
- Track original count for metadata

## Configuration

### config.yaml Settings

```yaml
# LLM Settings (for synthesis)
llm:
  model: gpt-oss-20b
  base_url: http://192.168.0.247:9003/v1
  api_key: ollama
  temperature: 0.1
  timeout: 900
  max_concurrent: 100

# Embedding Settings
embedding:
  model: bge-large-en
  base_url: http://192.168.0.136:9003/v1
  batch_size: 25
  max_concurrent: 50

# Analysis Settings
analysis:
  duplicate_similarity_threshold: 0.9  # For initial detection
  sampling:
    high_similarity_threshold: 0.85
```

### Thresholds

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Embedding similarity (detection) | 0.85-0.9 | Candidate pair selection |
| Graph edge threshold | 0.8 | Transitive closure formation |
| LLM confidence | 0.7+ | Duplicate confirmation |

## Output Files

### Primary Output
- **`input/methods_deduplicated.csv`**: 595 deduplicated methods
  - Format: `Index|Method|Description|Source`
  - Original indices preserved (not re-indexed)

### Metadata
- **`results/filter_metadata.json`**: Filtering statistics
  ```json
  {
    "original_count": 681,
    "filtered_count": 595,
    "removed_count": 86,
    "duplicate_groups": 60,
    "similarity_threshold": 0.8,
    "selection_strategy": "llm_synthesis",
    "synthesis": {
      "groups_synthesized": 60,
      "methods_updated": 60
    },
    "removed_methods": [...]
  }
  ```

- **`results/duplicates.json`**: Raw duplicate detection results
  - All analyzed pairs with similarity scores
  - LLM-confirmed duplicate pairs
  - Initial group assignments

## Duplicate Categories Found

### Large Duplicate Groups (Example)

**Group 0** (43 methods): Continuous Improvement / Kata practices
- Feedback-Driven Improvements
- PDCA (Plan-Do-Check-Act)
- Improvement Kata
- Coaching Kata
- Continuous Learning
- Iterative Refinement
- ... and 37 more variants

This group consolidates various terminology for iterative improvement practices across Lean, Agile, and Toyota Kata sources.

### Common Duplicate Patterns

1. **Terminology Variations**
   - "Daily Standup" / "Daily Scrum" / "Daily Scrum (Stand-up Meetings)"
   - "Sprint Retrospective" / "Retrospectives" / "Retrospective Reviews"

2. **Framework-Specific Naming**
   - "Burn Down Chart Usage" / "Burndown Charts"
   - "Task Board" / "Task Boards for Workflow Management" / "Kanban Board"

3. **Abstraction Level Overlap**
   - "Continuous Integration" (general) / "Shortened Feedback Loops" (specific aspect)
   - "Stream-Aligned Teams" / "Stream-Aligned Workflows" / "Systemic Flow Alignment"

## Performance

| Stage | Time | Bottleneck |
|-------|------|------------|
| Embedding generation | ~5-10 min | API batch calls |
| Duplicate detection | ~30-60 min | LLM verification calls |
| Transitive closure | <1 second | Graph computation |
| Description synthesis | ~2-5 min | Batched LLM calls |
| **Total** | ~40-75 min | LLM API latency |

## Execution

```bash
# Prerequisites: Run full analysis pipeline first to generate duplicates.json
# This creates embeddings and detects duplicate pairs

# Then run deduplication
python filter_methods_for_analysis.py

# Output:
# - input/methods_deduplicated.csv (595 methods)
# - results/filter_metadata.json (statistics)
```

## Alternative Script (Not Used)

The codebase also contains `find_and_merge_duplicates.py` which uses:
- Exact name matching (case-sensitive)
- Anthropic Claude API for description synthesis

This script was **not used** for the final `methods_deduplicated.csv`. The semantic approach via `filter_methods_for_analysis.py` was preferred because it:
1. Catches semantically similar methods with different names
2. Uses the local VLLM server (no external API costs)
3. Integrates with the existing embedding infrastructure

## File Inventory

### Core Scripts
| File | Purpose |
|------|---------|
| `filter_methods_for_analysis.py` | Main deduplication orchestrator |
| `find_exact_duplicates.py` | Name-based duplicate detection (exploratory) |
| `find_and_merge_duplicates.py` | Alternative approach using Claude (not used) |

### Supporting Modules
| File | Purpose |
|------|---------|
| `src/data.py` | CSV loading, Method dataclass |
| `src/embeddings.py` | BGE embedding generation, similarity matrix |
| `src/sampler.py` | Smart sampling for LLM calls |
| `src/analyzer.py` | LLM-based duplicate verification |
| `src/duplicate_synthesizer.py` | Batch description synthesis |

### Configuration
| File | Purpose |
|------|---------|
| `config.yaml` | LLM/embedding server endpoints, thresholds |
| `prompts/duplicate.txt` | LLM prompt for duplicate detection |

### Input/Output
| File | Records | Purpose |
|------|---------|---------|
| `input/methods.csv` | 681 | Raw input |
| `input/methods_deduplicated.csv` | 595 | Deduplicated output |
| `results/duplicates.json` | 2,596 pairs | Duplicate detection results |
| `results/filter_metadata.json` | - | Deduplication statistics |

## Design Decisions

### Why Semantic Similarity Over Exact Matching?

1. **Coverage**: Catches methods with different names but same meaning
2. **Cross-source**: Identifies duplicates across different source documents
3. **Robustness**: Handles terminology variations and abbreviations

### Why Transitive Closure?

1. **Completeness**: If A~B and B~C, all three should be in one group
2. **Consistency**: Avoids partial merges where related methods are split
3. **Simplicity**: Single threshold decision, no complex clustering

### Why Local LLM (gpt-oss-20b)?

1. **Cost**: No API charges for synthesis calls
2. **Speed**: Local VLLM server with high concurrency
3. **Privacy**: Data doesn't leave local infrastructure
4. **Consistency**: Same model used throughout the pipeline

### Why Batch Synthesis?

1. **Efficiency**: 6 API calls vs 60 individual calls
2. **GPU Utilization**: Better batching on VLLM server
3. **Fallback**: Automatic retry with individual calls on failure

## Validation

### Completeness Check
- Input: 681 methods
- Output: 595 methods
- Removed: 86 duplicates
- Groups: 60 synthesized

### Quality Indicators
1. Similarity threshold (0.8) balances precision vs recall
2. LLM synthesis produces coherent 2-3 sentence descriptions
3. Representative selection preserves original indices
4. Metadata tracks all removed methods for audit

### Manual Verification Points
1. Review `results/filter_metadata.json` for removed methods list
2. Sample check large groups (e.g., Group 0 with 43 methods)
3. Verify synthesized descriptions capture key aspects
4. Check no false positives (incorrectly merged distinct methods)
