# Method Analysis Architecture: Process Flow Documentation

## Overview

This document describes the detailed process flow for analyzing product development methods through two key stages: 12-Dimensional Analysis and Compatibility Analysis. These stages form the analytical core of the computational methods tool, transforming raw method descriptions into structured, quantified data suitable for toolkit generation and visualization.

---

## Stage 1: 12-Dimensional Analysis

### Purpose
The 12-dimensional analysis scores each method across 12 independent dimensions plus 1 derived composite metric, enabling multi-faceted comparison and toolkit optimization.

### Scripts and Components

| Component | Role |
|-----------|------|
| `analyze_12d.py` | Main orchestrator script |
| `src/ranking_analyzer.py` | Core ranking engine with LLM integration |
| `src/data.py` | Method data loading and validation |
| `prompts/rank_*.txt` | 12 external prompt templates for each dimension |
| `config.yaml` | LLM and ranking configuration parameters |

### Input/Output

**Input:**
- `input/methods_deduplicated.csv` - Pipe-delimited CSV with columns: Index|Method|Description|Source

**Output:**
- `results/method_scores_12d_deduplicated.json` - Complete scores with metadata

### The 12 Dimensions

| # | Dimension | Scale Description | Low End | High End |
|---|-----------|-------------------|---------|----------|
| 1 | Scope | Tactical ↔ Strategic | Individual/team, specific tasks | Enterprise-wide, systemic change |
| 2 | Temporality | Immediate ↔ Evolutionary | Hours/days impact | Years impact, cultural change |
| 3 | Ease of Adoption | Hard ↔ Easy | Significant learning curve | Minimal training needed |
| 4 | Resources Required | Low ↔ High | Minimal investment | Significant people/tools/infrastructure |
| 5 | Technical Complexity | Simple ↔ Complex | Low technical barriers | Specialized knowledge required |
| 6 | Change Management Difficulty | Easy ↔ Hard | Natural fit | Major cultural transformation |
| 7 | Impact Potential | Low ↔ High | Incremental improvements | Transformative results |
| 8 | Time to Value | Slow ↔ Fast | Months/years to benefits | Days/weeks to value |
| 9 | Applicability | Niche ↔ Universal | Specific contexts only | Broadly applicable |
| 10 | People Focus | Technical/System ↔ Human | Automation-focused | Relationships, behavior, culture |
| 11 | Process Focus | Ad-hoc ↔ Systematic | Flexible, situational | Fully procedural, standardized |
| 12 | Purpose Orientation | Internal ↔ External | Cost cutting, productivity | Customer value, innovation |

**Derived Metric:**
- **Implementation Difficulty** (13th dimension) = Average of:
  - (100 - Ease of Adoption)
  - Resources Required
  - Technical Complexity
  - Change Management Difficulty

### Ranking Process Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: CHUNKING                                                       │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  595 methods → Create overlapping chunks                                 │
│                                                                          │
│  Configuration:                                                          │
│    • Chunk size: 18 methods per LLM call                                │
│    • Overlap size: 4 methods between consecutive chunks                 │
│    • Result: ~60-70 chunks with 4-method overlap for calibration        │
│                                                                          │
│  Purpose of Overlap:                                                     │
│    • Methods appearing in multiple chunks provide calibration points    │
│    • Enables merging chunk-local rankings into global ranking           │
│    • Validates consistency across chunk boundaries                      │
└────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: LLM RANKING (Per Dimension)                                   │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  For each of 12 dimensions:                                              │
│                                                                          │
│  1. Load dimension-specific prompt template                              │
│     • Provides scale definition (low → high)                            │
│     • Includes comparison anchors (known methods as reference)          │
│     • Specifies output format (JSON array)                              │
│                                                                          │
│  2. Process chunks in parallel batches                                   │
│     • 4 chunks processed simultaneously (configurable)                   │
│     • Each chunk: LLM ranks 18 methods from 1-18                        │
│     • Strict validation: no ties, all ranks used exactly once           │
│                                                                          │
│  3. Retry logic with exponential backoff                                 │
│     • Up to 50 retries per chunk                                        │
│     • Rejects: out-of-range ranks, duplicates, incomplete rankings     │
│     • Max backoff: 30 seconds between retries                           │
│                                                                          │
│  LLM Prompt Structure:                                                   │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  RANKING TASK: Order these {count} methods by {DIMENSION}        │   │
│  │                                                                   │   │
│  │  SCALE:                                                           │   │
│  │  Rank 1 = [lowest end description]                               │   │
│  │  Rank {count} = [highest end description]                        │   │
│  │                                                                   │   │
│  │  COMPARISON ANCHORS:                                              │   │
│  │  - "Known method A" → near 1 (low)                               │   │
│  │  - "Known method B" → near {count} (high)                        │   │
│  │  - "Known method C" → middle                                     │   │
│  │                                                                   │   │
│  │  METHODS:                                                         │   │
│  │  1. Method Name                                                   │   │
│  │     Full description text...                                      │   │
│  │  2. Method Name                                                   │   │
│  │     Full description text...                                      │   │
│  │  ...                                                              │   │
│  │                                                                   │   │
│  │  RULES:                                                           │   │
│  │  1. Use ALL ranks 1-{count} exactly once (no ties)               │   │
│  │  2. Spread across full range - don't cluster                     │   │
│  │                                                                   │   │
│  │  OUTPUT: JSON array only [[method_number, rank], ...]            │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  Expected LLM Response:                                                  │
│  [[1, 5], [2, 12], [3, 1], [4, 18], ...]                               │
│  (method_number, assigned_rank)                                          │
└────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 3: MULTI-PASS VALIDATION                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  Configuration: 5 ranking passes per dimension                           │
│                                                                          │
│  1. Different Shuffle Seed Per Pass                                      │
│     • Pass 1: seed=42, Pass 2: seed=43, etc.                            │
│     • Methods shuffled before chunking each pass                        │
│     • Each method compared against different peer groups                 │
│                                                                          │
│  2. Why Multiple Passes Matter                                           │
│     • Single comparison group can bias rankings                         │
│     • Shuffling exposes methods to diverse comparison contexts          │
│     • Reduces impact of LLM inconsistency on final scores               │
│                                                                          │
│  3. Cross-Validation Metrics                                             │
│     • Calculate pairwise correlation between passes                      │
│     • Consistency threshold: 0.80 minimum correlation                   │
│     • Warn if consistency drops below threshold                         │
└────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 4: SCORE MERGING AND CALIBRATION                                 │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  Step 4a: Chunk-Local Normalization                                      │
│  ─────────────────────────────────                                       │
│  • Each chunk's ranks (1-18) normalized to 0-1 scale                    │
│  • Formula: normalized = (rank - 1) / (chunk_size - 1)                  │
│  • Rank 1 → 0.0, Rank 18 → 1.0                                          │
│                                                                          │
│  Step 4b: Overlap-Based Calibration                                      │
│  ────────────────────────────────                                        │
│  • Methods in multiple chunks have multiple normalized scores           │
│  • Average their scores to merge across chunk boundaries                 │
│  • Overlap methods act as calibration anchors                           │
│                                                                          │
│  Step 4c: Pass Combination (Trimmed Mean)                                │
│  ─────────────────────────────────────────                               │
│  • 5+ passes: Use trimmed mean (remove 20% outliers)                    │
│  • 3-4 passes: Use median                                               │
│  • 1-2 passes: Use mean                                                 │
│  • Result: Single 0-1 score per method per dimension                    │
│                                                                          │
│  Step 4d: Calibrated Score Conversion                                    │
│  ────────────────────────────────────                                    │
│  • Map 0-1 normalized scores to realistic 0-100 ranges                  │
│  • Avoid artificial extremes (0 or 100)                                 │
│  • Dimension-specific ranges:                                           │
│    - Scope: 5-95 realistic, 20-80 typical                               │
│    - Impact: 10-95 realistic, 25-80 typical                             │
│  • Compression at extremes:                                             │
│    - Bottom 10% compressed to below typical_min                         │
│    - Top 10% compressed to above typical_max                            │
│    - Middle 80% linearly mapped to typical range                        │
│                                                                          │
│  Step 4e: Global Calibration                                             │
│  ───────────────────────────                                             │
│  • Final soft compression for outliers beyond realistic bounds          │
│  • Asymptotic compression preserves ranking order                       │
│  • Prevents hard clipping while avoiding extreme values                 │
└────────────────────────────────────────────────────────────────────────┘
```

### Detailed Rank-to-Score Conversion

This section provides the mathematical details of how LLM rankings are converted to final scores.

#### Key Insight: No Global Re-Ranking

**Important:** The system does NOT produce a global rank (1 to 595). Instead, it converts chunk-local ranks directly to normalized scores (0-1), then uses the overlapping methods to calibrate and merge these scores across chunks. The final output is a continuous score (0-100), not a discrete rank.

**Why this approach?**
- Asking an LLM to rank 595 methods at once exceeds context limits
- Chunked ranking with overlap provides natural calibration points
- Normalized scores allow smooth merging without forcing artificial global rankings
- The overlap methods act as "bridge" calibrators between chunks

#### Step 1: Chunk-Local Normalization

Each chunk produces ranks 1 to N (where N = chunk_size, typically 18). These are normalized to a 0-1 scale:

```
normalized_score = (rank - 1) / (chunk_size - 1)
```

**Example (chunk of 18 methods):**

| Rank | Normalized Score |
|------|------------------|
| 1 | (1-1)/(18-1) = 0.000 |
| 5 | (5-1)/(18-1) = 0.235 |
| 9 | (9-1)/(18-1) = 0.471 |
| 14 | (14-1)/(18-1) = 0.765 |
| 18 | (18-1)/(18-1) = 1.000 |

#### Step 2: Overlap Averaging (The Calibration Mechanism)

Methods appearing in multiple chunks (due to 4-method overlap) have multiple normalized scores. These overlapping methods serve as calibration bridges between chunks.

**How chunks overlap:**

```
Chunk 1: [M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12, M13, M14, M15, M16, M17, M18]
                                                            ↓ overlap ↓
Chunk 2:                                     [M15, M16, M17, M18, M19, M20, M21, M22, M23, ...]
                                                            ↓ overlap ↓
Chunk 3:                                                         [M29, M30, M31, M32, M33, ...]
```

Methods M15-M18 appear in both Chunk 1 and Chunk 2. Their normalized scores from each chunk are averaged:

**Example:**

| Method | Chunk 1 Rank | Chunk 1 Norm | Chunk 2 Rank | Chunk 2 Norm | Merged Score |
|--------|--------------|--------------|--------------|--------------|--------------|
| M15 | 15/18 | 0.824 | 3/18 | 0.118 | 0.471 |
| M16 | 16/18 | 0.882 | 4/18 | 0.176 | 0.529 |
| M17 | 17/18 | 0.941 | 2/18 | 0.059 | 0.500 |
| M18 | 18/18 | 1.000 | 1/18 | 0.000 | 0.500 |

**Why this works:** Methods ranked high in Chunk 1 (positions 15-18) were the "most strategic" in that group. In Chunk 2, these same methods are now compared against even more strategic methods, so they rank lower. Averaging smooths this transition and places them appropriately on the global scale.

**Single-chunk methods:** Methods appearing in only one chunk simply use their normalized score directly from that chunk.

#### Step 3: Multi-Pass Combination

With 5 ranking passes (each with different shuffling), each method gets 5 merged scores. These are combined using trimmed mean:

1. Sort the 5 scores
2. Remove top and bottom 20% (1 value from each end)
3. Average the remaining 3 values

**Example:**
- Pass scores: [0.31, 0.28, 0.42, 0.29, 0.33]
- Sorted: [0.28, 0.29, 0.31, 0.33, 0.42]
- After trimming: [0.29, 0.31, 0.33]
- Trimmed mean: 0.31

#### Step 4: Calibrated Score Conversion

The 0-1 normalized score is mapped to a realistic 0-100 range using dimension-specific calibration. Each dimension has defined ranges:

```python
dimension_ranges = {
    'scope': {'min_realistic': 5, 'max_realistic': 95, 'typical_range': (20, 80)},
    'impact': {'min_realistic': 10, 'max_realistic': 95, 'typical_range': (25, 80)},
    # ... etc for each dimension
}
```

**Conversion logic:**

```
If normalized < 0.1 (bottom 10%):
    # Compress into [min_realistic, typical_min]
    position = normalized / 0.1
    score = min_realistic + position × (typical_min - min_realistic)

Else if normalized > 0.9 (top 10%):
    # Compress into [typical_max, max_realistic]
    position = (normalized - 0.9) / 0.1
    score = typical_max + position × (max_realistic - typical_max)

Else (middle 80%):
    # Linear mapping to [typical_min, typical_max]
    position = (normalized - 0.1) / 0.8
    score = typical_min + position × (typical_max - typical_min)
```

**Example for Scope dimension (typical_range: 20-80, realistic: 5-95):**

| Normalized | Region | Calculation | Final Score |
|------------|--------|-------------|-------------|
| 0.05 | Bottom 10% | 5 + (0.05/0.1) × (20-5) = 5 + 7.5 | 12.5 |
| 0.31 | Middle 80% | 20 + ((0.31-0.1)/0.8) × (80-20) = 20 + 15.75 | 35.8 |
| 0.50 | Middle 80% | 20 + ((0.50-0.1)/0.8) × (80-20) = 20 + 30 | 50.0 |
| 0.75 | Middle 80% | 20 + ((0.75-0.1)/0.8) × (80-20) = 20 + 48.75 | 68.8 |
| 0.95 | Top 10% | 80 + ((0.95-0.9)/0.1) × (95-80) = 80 + 7.5 | 87.5 |

#### Step 5: Jitter Addition

Small random variation is added to prevent identical scores:

```
jitter = random.normal(0, 0.5)  # Mean 0, std 0.5
final_score = base_score + jitter
```

Jitter is applied in rank order to preserve relative ordering.

#### Step 6: Global Calibration

Final pass to compress any outliers beyond realistic bounds using soft compression:

```
If score < min_realistic:
    # Compress into [min_realistic - 3, min_realistic]
    ratio = (score - actual_min) / (min_realistic - actual_min)
    score = (min_realistic - 3) + ratio × 3

If score > max_realistic:
    # Compress into [max_realistic, max_realistic + 3]
    ratio = (score - max_realistic) / (actual_max - max_realistic)
    score = max_realistic + ratio × 3
```

This preserves ranking order while avoiding extreme values.

#### Complete Worked Example

**Method: "Sprint Planning" on Scope dimension**

1. **Chunk Rankings (5 passes):**
   - Pass 1: Chunk 3 rank 7/18, Chunk 4 rank 5/18
   - Pass 2: Chunk 2 rank 8/18, Chunk 3 rank 6/18
   - Pass 3: Chunk 5 rank 9/18
   - Pass 4: Chunk 1 rank 7/18, Chunk 2 rank 8/18
   - Pass 5: Chunk 4 rank 6/18

2. **Normalize and merge per pass:**
   - Pass 1: avg(0.353, 0.235) = 0.294
   - Pass 2: avg(0.412, 0.294) = 0.353
   - Pass 3: 0.471
   - Pass 4: avg(0.353, 0.412) = 0.382
   - Pass 5: 0.294

3. **Trimmed mean:** Sort [0.294, 0.294, 0.353, 0.382, 0.471], trim ends → mean([0.294, 0.353, 0.382]) = **0.343**

4. **Calibrated conversion (Scope: typical 20-80):**
   - 0.343 is in middle 80% (0.1 to 0.9)
   - position = (0.343 - 0.1) / 0.8 = 0.304
   - score = 20 + 0.304 × 60 = **38.2**

5. **Add jitter:** 38.2 + 0.3 = **38.5**

6. **Global calibration:** 38.5 is within realistic bounds (5-95) → **38.5** (unchanged)

**Final Score: 38.5** (indicating Sprint Planning is moderately tactical on the Scope dimension)

```
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  OUTPUT: method_scores_12d_deduplicated.json                            │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  {                                                                       │
│    "metadata": {                                                         │
│      "total_methods": 595,                                               │
│      "dimensions": ["scope", "temporality", ...],                       │
│      "dimension_descriptions": {...}                                    │
│    },                                                                    │
│    "summary": {                                                          │
│      "dimensions": {                                                     │
│        "scope": {"mean": 50.2, "std": 18.3, "min": 8.5, "max": 92.1}   │
│      }                                                                   │
│    },                                                                    │
│    "methods": [                                                          │
│      {                                                                   │
│        "index": 1,                                                       │
│        "name": "Method Name",                                           │
│        "scope": 72.5,                                                    │
│        "temporality": 45.3,                                              │
│        ...all 13 dimensions...                                          │
│      }                                                                   │
│    ]                                                                     │
│  }                                                                       │
└────────────────────────────────────────────────────────────────────────┘
```

### Performance Characteristics

| Metric | Value |
|--------|-------|
| Total methods | 595 |
| Chunks per dimension | ~60-70 |
| Parallel chunks | 4 (configurable up to 20) |
| Validation passes | 5 |
| Dimensions | 12 |
| Estimated time | 75-100 minutes |
| LLM calls | ~4,200 (70 chunks × 12 dims × 5 passes) |

### Quality Assurance Mechanisms

1. **Strict Response Validation**
   - Rejects out-of-range ranks (not in 1-N)
   - Rejects duplicate ranks (violates "no ties" rule)
   - Rejects incomplete rankings (missing methods)
   - Up to 50 retries per chunk with exponential backoff

2. **Overlap Consistency Checking**
   - Calculates variance of normalized ranks for overlap methods
   - Flags methods with high variance across chunks
   - Warns if >30% of overlap methods show inconsistency

3. **Cross-Pass Correlation**
   - Computes pairwise correlation between ranking passes
   - Threshold: 0.80 minimum average correlation
   - Option to fail on low consistency (configurable)

4. **Distribution Validation**
   - Reports score range, mean, standard deviation
   - Counts unique values (should approach method count)
   - Calculates uniformity metric (lower = better)
   - Warns on high clustering

---

## Stage 2: Compatibility Analysis

### Purpose
The compatibility analysis evaluates pairwise relationships between methods to identify:
- **Incompatible pairs**: Methods that conflict or should not be used together
- **Synergistic pairs**: Methods that work better when combined
- **Overlap relationships**: Methods that address similar problems or contexts

### Scripts and Components

| Component | Role |
|-----------|------|
| `analyze_compatibility.py` | Main analyzer with LLM integration |
| `src/embeddings.py` | Embedding generation for similarity filtering |
| `config.yaml` | LLM configuration and parameters |

### Input/Output

**Inputs:**
- `input/methods_deduplicated.csv` - Method definitions
- `results/method_scores_12d_deduplicated.json` - 12D scores (for strategic sampling)

**Outputs:**
- `results/compatibility_checkpoint.pkl` - All pairwise analysis results (~176,715 pairs)
- `results/compatibility_analysis.json` - Summary report with top incompatibilities and synergies
- `results/compatibility_graph_sparse.json` - Sparse graph for visualization

### Compatibility Dimensions

The LLM evaluates each pair on 5 conceptual dimensions:

| Dimension | Scale | Low End | High End |
|-----------|-------|---------|----------|
| Resource Conflict | -1 to +1 | Compete for same resources | Complementary resource use |
| Conceptual Overlap | -1 to +1 | Redundant (solve same problem) | Distinct purposes |
| Philosophical Alignment | -1 to +1 | Contradictory principles | Reinforcing principles |
| Implementation Sequence | -1 to +1 | Prerequisite dependency | Synergistic combination |
| Cognitive Load | -1 to +1 | Overwhelming together | Manageable together |

### Overlap Analysis Fields

Each pair receives a structured overlap analysis:

| Field | Type | Description |
|-------|------|-------------|
| same_problem | boolean | Do both address the same core problem? |
| same_role | boolean | Does the same role execute both? |
| same_timing | boolean | Are they used at the same time/ceremony? |
| same_output | boolean | Do they produce the same artifact/output? |
| causes_confusion | boolean | Would using both cause confusion? |
| overlap_type | enum | none / partial / conflicting / redundant |
| has_problematic_overlap | boolean | Derived: true if conflicting or redundant |

### Classification Rules

Based on compatibility score and overlap analysis:

| Classification | Rule | Stored in Graph |
|---------------|------|-----------------|
| **Incompatible** | score < 0.7 | Yes (avoid) |
| **Synergistic** | score >= 0.95 AND overlap_type != conflicting | Yes (combine) |
| **Nonrelated** | same_problem=false AND same_output=false AND overlap_type=none | No (implicit) |
| **Compatible** | Everything else (0.7 <= score < 0.95) | No (implicit) |

### Analysis Process Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: PAIR GENERATION                                                │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  Total Possible Pairs: 595 × 594 / 2 = 176,715                          │
│                                                                          │
│  Strategic Sampling (reduces to ~10%):                                   │
│                                                                          │
│  Strategy 1: High-Impact Method Pairs                                    │
│  ─────────────────────────────────────                                   │
│  • Identify top N methods by impact_potential (from 12D scores)         │
│  • Generate all pairs within this top group                             │
│  • N scales with max_pairs budget                                       │
│                                                                          │
│  Strategy 2: Same-Source Pairs                                           │
│  ────────────────────────────────                                        │
│  • Group methods by source/origin                                       │
│  • Sample pairs within each source group                                │
│  • Higher likelihood of finding overlaps                                │
│  • Cap at 50% of budget                                                 │
│                                                                          │
│  Strategy 3: Cross-Source Pairs                                          │
│  ──────────────────────────────                                          │
│  • Sample pairs between different source groups                         │
│  • Discovers complementary approaches                                   │
│  • Cap at 80% of budget                                                 │
│                                                                          │
│  Strategy 4: Random Baseline                                             │
│  ─────────────────────────────                                           │
│  • Fill remaining quota with random pairs                               │
│  • At least 10% of budget for unbiased sampling                         │
│  • Prevents strategic sampling bias                                     │
│                                                                          │
│  Alternative: Exhaustive Mode                                            │
│  ─────────────────────────────                                           │
│  • Generates all remaining pairs systematically                         │
│  • Used after strategic sampling has covered high-value pairs           │
│  • Enables complete coverage over multiple runs                         │
└────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: EMBEDDING-BASED PRE-FILTERING                                 │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  Purpose: Auto-classify obvious redundancies without LLM calls          │
│                                                                          │
│  Process:                                                                │
│  1. Generate embeddings for all methods (via embedding API)             │
│  2. Compute cosine similarity for each candidate pair                   │
│  3. Filter by similarity threshold:                                     │
│                                                                          │
│     Similarity > 0.95 (Very High):                                      │
│     ┌────────────────────────────────────────────────────────────┐      │
│     │  AUTO-CLASSIFY as redundant without LLM                    │      │
│     │  • compatibility_score: 0.25                               │      │
│     │  • relationship_type: "redundant"                          │      │
│     │  • recommendation: "choose_one"                            │      │
│     │  • overlap_type: "redundant"                               │      │
│     │  • auto_classified: true                                   │      │
│     └────────────────────────────────────────────────────────────┘      │
│                                                                          │
│     Similarity <= 0.95:                                                  │
│     ┌────────────────────────────────────────────────────────────┐      │
│     │  SEND TO LLM for detailed analysis                         │      │
│     │  • May be compatible, synergistic, or incompatible        │      │
│     │  • Low similarity ≠ incompatible (could be complementary) │      │
│     └────────────────────────────────────────────────────────────┘      │
│                                                                          │
│  Result: Typically reduces LLM calls by 5-15%                           │
│                                                                          │
│  Note: For complete coverage analysis (all 176,715 pairs), this phase  │
│  was applied to identify obvious redundancies across the full dataset.  │
└────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 3: LLM COMPATIBILITY ANALYSIS                                    │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  For each pair, TWO LLM calls:                                          │
│                                                                          │
│  Call 1: Overlap Detection                                               │
│  ─────────────────────────────                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Analyze overlap between these methods:                          │   │
│  │                                                                   │   │
│  │  Method A: {name}                                                 │   │
│  │  {description}                                                    │   │
│  │                                                                   │   │
│  │  Method B: {name}                                                 │   │
│  │  {description}                                                    │   │
│  │                                                                   │   │
│  │  YES/NO for each:                                                 │   │
│  │  1. Same core problem?                                           │   │
│  │  2. Same role executes both?                                     │   │
│  │  3. Same timing/ceremony?                                        │   │
│  │  4. Same output/artifact?                                        │   │
│  │  5. Confusing to use both?                                       │   │
│  │                                                                   │   │
│  │  Classification:                                                  │   │
│  │  - 4-5 YES = redundant                                           │   │
│  │  - 2-3 YES = conflicting                                         │   │
│  │  - 1 YES = partial                                               │   │
│  │  - 0 YES = none                                                  │   │
│  │                                                                   │   │
│  │  JSON only: {same_problem, same_role, ..., overlap_type}        │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  Call 2: Compatibility Scoring                                           │
│  ─────────────────────────────                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Rate compatibility (0.0-1.0):                                   │   │
│  │                                                                   │   │
│  │  Method A: {name}                                                 │   │
│  │  {description}                                                    │   │
│  │                                                                   │   │
│  │  Method B: {name}                                                 │   │
│  │  {description}                                                    │   │
│  │                                                                   │   │
│  │  Calibration scale:                                               │   │
│  │  0.95 = Daily Standup + Sprint Planning                          │   │
│  │  0.85 = TDD + CI                                                 │   │
│  │  0.60 = Scrum + Kanban                                           │   │
│  │  0.35 = User Stories + Use Cases                                 │   │
│  │  0.20 = Sprint Review + Sprint Demo (redundant)                  │   │
│  │  0.10 = Waterfall + Agile (incompatible)                         │   │
│  │                                                                   │   │
│  │  Consider: Resource conflict? Purpose overlap? Philosophy?       │   │
│  │                                                                   │   │
│  │  JSON only: {compatibility_score, relationship_type,             │   │
│  │              recommendation, key_concern}                        │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  Parallel Processing:                                                    │
│  • Semaphore limits concurrent LLM calls (default: 25)                  │
│  • Progress bar tracks pair analysis                                    │
│  • Retry logic with exponential backoff                                 │
└────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 4: RESULT COMBINATION AND VALIDATION                             │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  Combine Results:                                                        │
│  • Merge overlap analysis + compatibility scoring                       │
│  • Add method indices for graph construction                            │
│  • Track auto-classified vs LLM-analyzed pairs                         │
│                                                                          │
│  Validation Checks:                                                      │
│  1. Overlap Consistency                                                 │
│     • Flag: has_problematic_overlap=true BUT score > 0.6               │
│     • Indicates potential LLM inconsistency                            │
│                                                                          │
│  2. Score Consistency (if dimension scores present)                     │
│     • Compare dimension average to final compatibility                  │
│     • Flag if difference > 0.3                                         │
│                                                                          │
│  Checkpoint Saving:                                                      │
│  • Save to compatibility_checkpoint.pkl after each batch               │
│  • Enables resume from interruption                                    │
│  • Tracks already-analyzed pairs to avoid duplicates                   │
└────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 5: GRAPH CONSTRUCTION                                            │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  Reclassification (Refined Rules):                                       │
│  • Apply classification rules to all results                            │
│  • Separate into: incompatible, synergistic, nonrelated, compatible    │
│                                                                          │
│  Sparse Graph Construction:                                              │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │  STORED EDGES (actionable relationships):                      │     │
│  │                                                                 │     │
│  │  Incompatible Edges:                                           │     │
│  │  • score < 0.7                                                 │     │
│  │  • Repulsion strength = 1.0 - score                            │     │
│  │  • Use case: Avoid these pairs in toolkits                     │     │
│  │                                                                 │     │
│  │  Synergistic Edges:                                            │     │
│  │  • score >= 0.95 AND overlap_type != conflicting               │     │
│  │  • Attraction strength = score                                  │     │
│  │  • Use case: Prioritize these combinations                     │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │  IMPLICIT (not stored):                                        │     │
│  │                                                                 │     │
│  │  Compatible: 0.7 <= score < 0.95                               │     │
│  │  • These pairs work fine together                              │     │
│  │  • No special handling needed                                  │     │
│  │                                                                 │     │
│  │  Nonrelated: Different problems, no overlap                    │     │
│  │  • No constraints on using together                            │     │
│  │  • Neither synergistic nor problematic                         │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                          │
│  Sparsity Benefit:                                                       │
│  • Typically 80-90% reduction in stored edges                           │
│  • Focus on actionable information only                                 │
│  • Faster graph algorithms and visualization                            │
└────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  OUTPUTS                                                                 │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  compatibility_checkpoint.pkl:                                           │
│  • All raw results for resumable analysis                               │
│  • Set of analyzed pair indices                                         │
│  • ~176,715 pairs when complete                                         │
│                                                                          │
│  compatibility_analysis.json:                                            │
│  {                                                                       │
│    "metadata": {                                                         │
│      "total_methods": 595,                                               │
│      "pairs_analyzed": 176715,                                           │
│      "incompatibilities_found": 30589,                                   │
│      "synergies_found": 7318                                            │
│    },                                                                    │
│    "statistics": {                                                       │
│      "avg_compatibility": 0.78,                                          │
│      "incompatible_count": 30589,                                        │
│      "synergistic_count": 7318                                          │
│    },                                                                    │
│    "top_incompatibilities": [...top 20 with overlap data...],           │
│    "top_synergies": [...top 50 with overlap data...]                    │
│  }                                                                       │
│                                                                          │
│  compatibility_graph_sparse.json:                                        │
│  {                                                                       │
│    "nodes": [{id, name, source}, ...],                                  │
│    "edges": [                                                            │
│      {source, target, type, score, strength, concern},                  │
│      ...                                                                 │
│    ],                                                                    │
│    "metadata": {                                                         │
│      "edges_stored": 37907,                                              │
│      "sparsity_reduction": "78.6%"                                      │
│    }                                                                     │
│  }                                                                       │
└────────────────────────────────────────────────────────────────────────┘
```

### Performance Characteristics

| Metric | Value |
|--------|-------|
| Total pairs analyzed | 176,715 (complete coverage) |
| LLM calls per pair | 2 (overlap + compatibility) |
| Concurrent LLM calls | 25 |
| Analysis time | Multiple runs with checkpointing |
| Checkpoint frequency | After each batch |

**Note:** The full pairwise analysis covers all 176,715 possible pairs, providing complete compatibility data rather than a statistical sample. This enables accurate toolkit generation and comprehensive synergy/conflict detection.

### Incremental Analysis Support

The system supports incremental analysis through checkpointing:

1. **Checkpoint Loading**
   - On startup, loads existing `compatibility_checkpoint.pkl`
   - Extracts set of already-analyzed pair indices
   - Deduplicates any duplicate entries

2. **Pair Exclusion**
   - New pair generation excludes already-analyzed pairs
   - Enables spreading analysis across multiple runs
   - Supports both strategic and exhaustive modes

3. **Result Merging**
   - New results appended to existing results
   - Updated checkpoint saved immediately after analysis
   - Report generation uses complete accumulated data

---

## Integration Between Stages

### Data Flow

```
methods_deduplicated.csv
         │
         ├─────────────────────────────────────────────┐
         │                                             │
         ▼                                             ▼
┌─────────────────────┐                    ┌─────────────────────┐
│  12D Analysis       │                    │  Compatibility      │
│  analyze_12d.py     │                    │  Analysis           │
│                     │                    │  analyze_           │
│  Output:            │                    │  compatibility.py   │
│  method_scores_     │────────────────────┤                     │
│  12d_deduplicated.  │ impact_potential   │  Uses:              │
│  json               │ for strategic      │  - Impact scores    │
│                     │ sampling           │    for sampling     │
└─────────────────────┘                    │  - Methods for      │
                                           │    embedding        │
                                           │    similarity       │
                                           └─────────────────────┘
```

### Key Dependencies

1. **12D Analysis provides:**
   - `impact_potential` score for strategic pair sampling
   - Method metadata for compatibility prompts

2. **Compatibility Analysis provides:**
   - Synergy scores for toolkit generation
   - Incompatibility warnings for toolkit validation
   - Overlap data for understanding method relationships

---

## Configuration Reference

### config.yaml Structure

```yaml
llm:
  base_url: "http://192.168.0.247:9003/v1"
  api_key: "your-api-key"
  model: "gpt-oss-20b"
  temperature: 0.1        # Low for consistent rankings
  timeout: 120           # Seconds per LLM call
  max_concurrent: 25     # Parallel LLM calls

embedding:
  base_url: "http://192.168.0.136:9003/v1"
  api_key: "your-api-key"
  model: "bge-large-en"
  batch_size: 25
  max_concurrent: 50

ranking:
  chunk_size: 18                    # Methods per ranking call
  overlap_size: 4                   # Overlap between chunks
  parallel_chunks: 4                # Concurrent chunk processing
  ranking_rounds: 5                 # Validation passes
  use_calibrated_scoring: true      # Realistic score ranges
  consistency_threshold: 0.8        # Min correlation between passes
  gaussian_std: 17.0                # Score distribution spread
  add_jitter: true                  # Small random variation
  jitter_amount: 0.5                # Max jitter in score points
```

---

## Summary

The method analysis pipeline transforms 595 deduplicated methods into:

1. **12-Dimensional Scores**: Comprehensive characterization across scope, effort, impact, and human factors with derived implementation difficulty metric.

2. **Pairwise Compatibility Data**: 176,715 relationship assessments with structured overlap analysis, enabling identification of synergistic and incompatible method pairs.

These outputs feed into downstream processes:

- Toolkit generation (context-aware method selection)
- Visualization (clustering, dashboards)
- Reporting (insights, recommendations)

The architecture emphasizes quality through multi-pass validation, calibrated scoring, and complete pairwise coverage, while optimizing performance through parallel processing, embedding-based pre-filtering, and incremental checkpointing.
