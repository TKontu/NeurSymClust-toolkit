# Method Portfolio Builder Architecture

## Overview

The Portfolio Builder generates optimized method toolkits tailored to specific organizational contexts. It combines 12-dimensional method scores, pairwise compatibility data, and semantic category information to select methods that:

1. **Fit the context**: Score highly on dimensions relevant to the organization's situation
2. **Work together**: Maintain high pairwise compatibility (no internal conflicts)
3. **Provide diversity**: Cover multiple categories without over-concentration
4. **Create synergy**: Leverage complementary methods that enhance each other

**Key Insight:** The builder uses a greedy selection algorithm that ensures every method added is compatible with ALL existing toolkit members, not just the most recently added one.

---

## Input Data

### Primary Inputs

| File | Description |
|------|-------------|
| `results/method_scores_12d_deduplicated.json` | 12-dimensional scores for 595 methods |
| `results/compatibility_checkpoint.pkl` | Complete pairwise compatibility analysis (176,715 pairs) |

### Supporting Inputs

| File | Description |
|------|-------------|
| `results_semantic_clustering_combined/combined_clusters.json` | Semantic cluster assignments |
| `results_semantic_clustering_combined/dendrogram_categories.json` | Category synergy definitions |

### 12D Scores Structure

```json
{
  "methods": [
    {
      "index": 1,
      "name": "Method Name",
      "scope": 72.5,
      "temporality": 45.3,
      "ease_adoption": 68.2,
      "resources_required": 35.1,
      "technical_complexity": 42.8,
      "change_management_difficulty": 51.3,
      "impact_potential": 78.9,
      "time_to_value": 62.4,
      "applicability": 55.7,
      "people_focus": 48.2,
      "process_focus": 63.1,
      "purpose_orientation": 71.5,
      "implementation_difficulty": 44.6
    }
  ]
}
```

---

## Context Profiles

The builder includes five pre-defined organizational context profiles, each with specific requirements:

### Profile Structure

```python
{
    'name': 'Human-readable name',
    'description': 'Situation description',
    'constraints': {
        'team_size': 'small|medium|large',
        'maturity': 'low|medium|high',
        'primary_challenge': 'speed|quality|innovation|compliance',
        'resource_level': 'minimal|moderate|high'
    },
    'dimension_weights': {
        'time_to_value': 0.35,      # Positive = higher is better
        'resources_required': -0.20  # Negative = lower is better
    },
    'ppp_profile': {'people': 70, 'process': 20, 'purpose': 85},
    'toolkit_size': 15,
    'min_compatibility': 0.7,
    'max_per_category': 3
}
```

### Available Contexts

| Context Key | Name | Primary Focus | Toolkit Size |
|-------------|------|---------------|--------------|
| `startup_mvp` | Startup - MVP Phase | Speed, low resources, fast value | 15 |
| `startup_scaling` | Startup - Scaling Phase | Balance quality and speed | 15 |
| `enterprise_transformation` | Enterprise - Digital Transformation | Strategic scope, high impact | 15 |
| `regulated_industry` | Regulated Industry - Compliance Focus | Process rigor, risk management | 15 |
| `hardware_product` | Hardware Product Development | Long cycles, planning, risk | 15 |

### Dimension Weights by Context

| Dimension | Startup MVP | Startup Scaling | Enterprise | Regulated | Hardware |
|-----------|-------------|-----------------|------------|-----------|----------|
| time_to_value | +0.35 | +0.20 | — | — | -0.10 |
| ease_adoption | +0.25 | — | — | +0.15 | — |
| resources_required | -0.20 | — | — | — | — |
| impact_potential | +0.15 | +0.25 | +0.25 | — | +0.20 |
| scope | — | +0.20 | +0.30 | — | — |
| temporality | — | — | +0.25 | — | — |
| process_focus | — | +0.15 | — | +0.30 | — |
| technical_complexity | — | -0.10 | — | +0.10 | — |
| change_management | — | -0.10 | -0.20 | — | — |
| applicability | +0.05 | — | — | +0.20 | — |

*Positive weights: Higher dimension value is better*
*Negative weights: Lower dimension value is better*

---

## Core Components

### CompatibilityMatrix Class

Provides O(1) lookup for method pair compatibility:

```python
class CompatibilityMatrix:
    """
    Manages pre-computed pairwise compatibility scores.

    Key methods:
    - get_compatibility(method_a, method_b) → float (0-1)
    - get_pair_data(method_a, method_b) → full result dict
    - get_overlap_analysis(method_a, method_b) → overlap dict
    - is_synergistic(method_a, method_b) → bool
    - is_incompatible(method_a, method_b) → bool
    - has_problematic_overlap(method_a, method_b) → bool
    """
```

**Internal Structure:**
- N×N numpy matrix for fast score lookup
- Dictionary mapping (method_a, method_b) → full result data
- Bidirectional name-to-index mapping

### SituationalToolkitBuilder Class

Main builder class that orchestrates toolkit generation:

```python
class SituationalToolkitBuilder:
    """
    Builds optimized toolkits using:
    - 12D scoring data
    - Pairwise compatibility matrices
    - Category synergy patterns
    - Organizational context profiles
    """
```

---

## Toolkit Building Algorithm

### Phase 1: Context Fitness Calculation

Calculate how well each method fits the target context:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  CONTEXT FITNESS SCORING                                                │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  For each method, calculate weighted combination of 12D scores:         │
│                                                                          │
│  fitness = 0                                                             │
│                                                                          │
│  For each (dimension, weight) in context.dimension_weights:             │
│      if weight > 0:                                                      │
│          # Higher dimension value is better                             │
│          fitness += method[dimension] × weight                          │
│      else:                                                               │
│          # Lower dimension value is better (invert scale)               │
│          fitness += (100 - method[dimension]) × |weight|                │
│                                                                          │
│  # Normalize to 0-100 scale                                             │
│  fitness = normalize(fitness)                                           │
│                                                                          │
│  Example (Startup MVP):                                                  │
│  ────────────────────                                                    │
│  dimension_weights = {                                                   │
│      'time_to_value': 0.35,      # Fast value delivery                 │
│      'ease_adoption': 0.25,      # Easy to learn                       │
│      'resources_required': -0.20, # Low resource needs                 │
│      'impact_potential': 0.15,   # Still want impact                   │
│      'applicability': 0.05       # Prefer universal methods            │
│  }                                                                       │
│                                                                          │
│  For method with:                                                        │
│    time_to_value=80, ease_adoption=70, resources=30,                    │
│    impact=60, applicability=55                                          │
│                                                                          │
│  fitness = 80×0.35 + 70×0.25 + (100-30)×0.20 + 60×0.15 + 55×0.05       │
│          = 28 + 17.5 + 14 + 9 + 2.75                                    │
│          = 71.25 (before normalization)                                 │
└────────────────────────────────────────────────────────────────────────┘
```

### Phase 2: Seed Selection

Initialize toolkit with the best-fit starting method:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  SEED SELECTION                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  Option A: User-Provided Seeds                                          │
│  ─────────────────────────────                                           │
│  If use_seeds parameter provided:                                        │
│      • Convert seed method names to dataframe indices                   │
│      • Add all seeds to initial toolkit                                 │
│      • Verify seeds exist in method database                            │
│                                                                          │
│  Option B: Automatic Seed Selection                                      │
│  ──────────────────────────────────                                      │
│  If preferred_categories defined in context:                             │
│      • Filter methods to preferred categories                           │
│      • Select method with highest fitness score                         │
│                                                                          │
│  Fallback:                                                               │
│      • Select method with highest fitness score overall                 │
│                                                                          │
│  Result: toolkit = [seed_method_idx]                                    │
└────────────────────────────────────────────────────────────────────────┘
```

### Phase 3: Greedy Compatible Expansion

Iteratively add methods that are compatible with ALL existing members:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  GREEDY COMPATIBLE EXPANSION                                            │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  while len(toolkit) < target_size:                                      │
│                                                                          │
│      STEP 1: Get Compatible Candidates                                  │
│      ─────────────────────────────────                                   │
│      candidates = all_methods - toolkit_members                         │
│                                                                          │
│      # Filter: Must be compatible with EVERY toolkit member             │
│      for each toolkit_member in toolkit:                                │
│          for each candidate in candidates:                              │
│              score = compatibility(toolkit_member, candidate)           │
│              if score < min_compatibility:                              │
│                  remove candidate                                       │
│                                                                          │
│      # Filter: Category limits                                          │
│      for each category with count >= max_per_category:                  │
│          remove candidates from that category                           │
│                                                                          │
│      # Filter: Avoided categories                                       │
│      remove candidates from avoid_categories                            │
│                                                                          │
│      if no candidates remain:                                           │
│          STOP (toolkit complete at current size)                        │
│                                                                          │
│      STEP 2: Score Candidates                                           │
│      ────────────────────────                                            │
│      For each candidate, calculate composite score:                     │
│                                                                          │
│      score = (                                                           │
│          fitness × 0.40 +                    # Context fit: 40%         │
│          avg_compatibility × 100 × 0.25 +   # Avg compat: 25%          │
│          min_compatibility × 100 × 0.20 +   # No weak links: 20%       │
│          diversity_bonus × 100 +            # New category: +10 additive│
│          synergy_score × 100 × 0.15         # Pairwise synergy: 15%    │
│      )                                                                   │
│                                                                          │
│      STEP 3: Select Best Candidate                                      │
│      ─────────────────────────────                                       │
│      best = candidate with highest composite score                      │
│      toolkit.append(best)                                               │
│                                                                          │
│      Log: method name, score, category, fitness                         │
└────────────────────────────────────────────────────────────────────────┘
```

### Candidate Scoring Components

The composite score combines weighted percentages plus an additive bonus:

- **Weighted components** (40% + 25% + 20% + 15% = 100%): fitness, avg_compat, min_compat, synergy
- **Additive bonus**: +10 points for introducing a new category

#### 1. Fitness Score (40%)

Pre-computed context fitness from Phase 1.

#### 2. Average Compatibility (25%)

```python
compatibilities = []
for toolkit_member in toolkit:
    score = compatibility_matrix.get_compatibility(candidate, toolkit_member)
    compatibilities.append(score)
avg_compat = mean(compatibilities)
```

#### 3. Minimum Compatibility (20%)

```python
min_compat = min(compatibilities)
```

Ensures no "weak links" in the toolkit - every pair must work well together.

#### 4. Diversity Bonus (additive, not weighted)

```python
diversity_bonus = 0.1 if candidate.category not in toolkit_categories else 0.0
# Added as: diversity_bonus * 100 = 10 points if new category, 0 otherwise
```

This is an **additive bonus** (not a percentage weight). When a candidate would add a new category to the toolkit, 10 points are added to the score. This encourages category coverage without being a hard requirement.

#### 5. Pairwise Synergy Score (15%)

Uses actual overlap_analysis data to detect true synergies:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PAIRWISE SYNERGY CALCULATION                                           │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  synergy_score = 0.0                                                     │
│                                                                          │
│  For each toolkit_member:                                                │
│      overlap = get_overlap_analysis(candidate, toolkit_member)          │
│                                                                          │
│      # Synergistic pair (score >= 0.95, not conflicting)               │
│      if is_synergistic(candidate, toolkit_member):                      │
│          synergy_score += 0.30                                          │
│                                                                          │
│      # Complementary (same problem, no problematic overlap)             │
│      elif overlap.same_problem AND                                      │
│           NOT overlap.has_problematic_overlap AND                       │
│           overlap.overlap_type == 'none':                               │
│          synergy_score += 0.15                                          │
│                                                                          │
│      # Penalize problematic overlap                                     │
│      if overlap.has_problematic_overlap:                                │
│          synergy_score -= 0.20                                          │
│                                                                          │
│  # Normalize by toolkit size                                            │
│  synergy_score = synergy_score / len(toolkit)                           │
│  synergy_score = clamp(synergy_score, -0.5, 1.0)                       │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Output Structure

### Single Toolkit Result

```json
{
  "context": "startup_mvp",
  "context_name": "Startup - MVP Phase",
  "context_description": "Small team building initial product...",
  "methods": [
    {
      "name": "Method Name",
      "category": "P12",
      "fitness_score": 78.5,
      "avg_compatibility": 0.847,
      "min_compatibility": 0.723,
      "diversity_score": 10.0,
      "synergy_score": 15.2,
      "impact_potential": 72.3,
      "implementation_difficulty": 38.5,
      "time_to_value": 81.2,
      "ease_adoption": 75.8
    }
  ],
  "method_names": ["Method 1", "Method 2", ...],
  "size": 15,
  "statistics": {
    "avg_fitness": 71.2,
    "avg_compatibility": 0.812,
    "min_compatibility": 0.687,
    "avg_impact": 68.5,
    "avg_difficulty": 42.3,
    "avg_time_to_value": 74.1,
    "categories_covered": 12,
    "category_distribution": {"P12": 2, "S5": 3, ...}
  }
}
```

### Multi-Context Comparison

```json
{
  "toolkits": {
    "startup_mvp": {...},
    "startup_scaling": {...},
    ...
  },
  "overlap_matrix": {
    "startup_mvp": {
      "startup_scaling": {
        "shared_count": 8,
        "shared_methods": ["Method A", "Method B", ...],
        "jaccard_similarity": 0.42
      }
    }
  },
  "unique_methods": {
    "startup_mvp": {
      "unique_methods": ["Method X", "Method Y"],
      "count": 2,
      "percentage": 13.3
    }
  },
  "transition_analysis": {
    "startup_mvp → startup_scaling → enterprise_transformation": {
      "stages": [
        {
          "from": "startup_mvp",
          "to": "startup_scaling",
          "keep": ["Method A", ...],
          "keep_count": 8,
          "add": ["Method P", ...],
          "add_count": 7,
          "remove": ["Method X", ...],
          "remove_count": 7,
          "continuity_percentage": 53.3
        }
      ]
    }
  },
  "dimension_coverage": {
    "startup_mvp": {
      "dimension_averages": {
        "scope": 42.5,
        "impact_potential": 68.3,
        ...
      },
      "balance_score": 78.2
    }
  }
}
```

---

## Visualization Data Generation

The builder generates data for multiple visualization types:

### 1. Network Graph

Shows method relationships within toolkit:

```json
{
  "nodes": [
    {
      "id": "Method Name",
      "label": "Method Name",
      "category": "P12",
      "fitness": 78.5,
      "impact": 72.3,
      "size": 72.3
    }
  ],
  "edges": [
    {
      "source": "Method A",
      "target": "Method B",
      "weight": 0.85,
      "strength": "strong"
    }
  ]
}
```

### 2. Spider/Radar Chart

12D dimension averages for the toolkit:

```json
{
  "dimensions": ["scope", "temporality", ...],
  "values": {"scope": 52.3, "temporality": 48.7, ...},
  "context": "Startup - MVP Phase"
}
```

### 3. Category Distribution

```json
{
  "categories": ["P12", "S5", "P3", ...],
  "counts": [3, 2, 2, ...],
  "total": 15
}
```

### 4. Implementation Timeline

Methods sequenced by difficulty and time-to-value:

```json
{
  "waves": {
    "quick_wins": ["Method A", "Method B"],
    "core_implementations": ["Method C", "Method D", ...],
    "advanced_practices": ["Method X", "Method Y"]
  },
  "sequenced_methods": ["Method A", "Method B", ...],
  "timeline_metadata": {
    "quick_wins_period": "0-3 months",
    "core_period": "3-9 months",
    "advanced_period": "9+ months"
  }
}
```

---

## Category Synergy System

### Loading Category Synergies

Category synergies are loaded from `dendrogram_categories.json`, which is generated by hierarchical clustering of semantic cluster centroids:

```python
CATEGORY_SYNERGIES = {
    'agile_scaling_and_flow': {
        'display_name': 'Agile Scaling & Flow',
        'categories': ['P12', 'S5', 'P19'],
        'cluster_names': ['Agile Scaling', 'Flow Optimization', ...],
        'reason': 'Semantically related clusters (UMAP 5D based)',
        'strength': 'strong',
        'bonus': 1.20
    }
}
```

### Synergy Bonus Calculation (Legacy)

When a candidate would help complete a synergy group:

```python
def calculate_synergy_bonus(candidate_category, current_categories):
    bonus = 0.0

    for synergy_def in CATEGORY_SYNERGIES.values():
        required_cats = set(synergy_def['categories'])

        if candidate_category in required_cats:
            present = current_categories & required_cats
            would_have = present | {candidate_category}

            coverage = len(would_have) / len(required_cats)
            strength_bonus = synergy_def.get('bonus', 1.0)

            if len(would_have) > len(present):
                bonus += coverage * (strength_bonus - 1.0)

    return min(bonus, 1.0)
```

**Note:** The current implementation prioritizes actual pairwise synergy from overlap_analysis over category-based synergy assumptions.

---

## Algorithm Design Rationale

### Why Greedy Selection?

Alternative approaches considered:

- **Optimization (ILP/genetic)**: Computationally expensive for 595 methods
- **Clustering-based**: Doesn't account for pairwise compatibility
- **Random sampling**: No guarantee of quality

Greedy selection advantages:
- Guaranteed compatibility at every step
- Explainable selection process
- Fast execution (polynomial time)
- Produces high-quality toolkits empirically

### Why ALL-Compatible Filtering?

The key constraint is that each new method must be compatible with **every** existing toolkit member, not just the most recently added one:

```
Traditional approach:
  Add Method A (any method)
  Add Method B (compatible with A)
  Add Method C (compatible with B, but might conflict with A!) ❌

Our approach:
  Add Method A (any method)
  Add Method B (compatible with A)
  Add Method C (compatible with BOTH A and B) ✓
```

This prevents "compatibility drift" where later additions conflict with earlier ones.

### Why Weighted Composite Scoring?

The composite score (fitness 40% + avg_compat 25% + min_compat 20% + diversity 10% + synergy 15%) balances multiple objectives:

- **Fitness dominance (40%)**: Primary goal is context fit
- **Compatibility insurance (45% total)**: Both average AND minimum matter
- **Exploration (25%)**: Diversity and synergy encourage balanced toolkits

---

## Command-Line Interface

```bash
# Build toolkit for specific context
python build_method_portfolios.py --context startup_mvp

# Build with seed methods
python build_method_portfolios.py --context startup_mvp --seeds "Daily Standup" "Sprint Planning"

# Compare all contexts
python build_method_portfolios.py --compare-all

# Build all toolkits (default)
python build_method_portfolios.py

# Custom paths
python build_method_portfolios.py \
    --scores results/method_scores_12d_deduplicated.json \
    --compatibility results/compatibility_checkpoint.pkl \
    --output-dir results \
    --output-prefix toolkit
```

---

## Output Files

| File | Description |
|------|-------------|
| `results/toolkit_{context}.json` | Single context toolkit with visualization data |
| `results/toolkit_all_contexts.json` | All five toolkits |
| `results/toolkit_comparison.json` | Multi-context comparison analysis |

---

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Methods evaluated | 595 |
| Context profiles | 5 |
| Toolkit size | 15 methods each |
| Compatibility checks per addition | O(toolkit_size × remaining_candidates) |
| Total build time (single context) | ~5 seconds |
| Total build time (all contexts) | ~25 seconds |
