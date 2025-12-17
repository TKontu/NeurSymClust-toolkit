# Molecular Graph Visualization Architecture

## Overview

The molecular visualization transforms pairwise compatibility data into an intuitive graph representation where methods are grouped into "molecules" based on their compatibility scores. High-compatibility methods cluster tightly together, while low-compatibility methods form loose, spread-out structures.

**Key Insight:** The visualization makes compatibility relationships visually apparent—tight clusters indicate synergistic method groups, while loose clusters reveal potentially problematic combinations.

---

## Input Data

### Primary Input

| File | Description |
|------|-------------|
| `results/compatibility_checkpoint.pkl` | Complete pairwise compatibility analysis (176,715 pairs) |

The checkpoint contains:
```python
{
    'results': [
        {
            'method_a': 'Method Name A',
            'method_a_index': 123,
            'method_b': 'Method Name B',
            'method_b_index': 456,
            'compatibility_score': 0.85,  # 0.0-1.0
            'relationship_type': 'compatible',
            'overlap_type': 'none',
            ...
        },
        ...
    ],
    'analyzed_pairs': {(123, 456), ...}  # Set of analyzed pair indices
}
```

### Supporting Inputs

| File | Description |
|------|-------------|
| `results_semantic_clustering_combined/combined_clusters.json` | Semantic cluster assignments for methods |
| `results_semantic_clustering_combined/dendrogram_categories.json` | Category groupings for clusters |
| `input/methods_deduplicated.csv` | Method sources and descriptions |

---

## Configuration Parameters

```python
# Edge Classification Thresholds
INCOMPATIBLE_THRESHOLD = 0.75   # Edges below this are drawn red
SYNERGISTIC_THRESHOLD = 0.95    # Edges above this are drawn dark green

# Molecule Formation Constraints
MAX_MOLECULE_SIZE = 10          # Maximum methods per molecule
MIN_MOLECULE_SIZE = 3           # Minimum methods per molecule
```

---

## Process Flow

### Phase 1: Data Loading and Graph Construction

```
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 1: Load Compatibility Data                                        │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  Input: compatibility_checkpoint.pkl                                     │
│                                                                          │
│  Extract:                                                                │
│  • 595 unique methods with indices and names                            │
│  • 176,715 pairwise compatibility scores                                │
│  • Method metadata (source, description)                                │
└────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 2: Build NetworkX Graph                                           │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  G = nx.Graph()                                                          │
│                                                                          │
│  Nodes:                                                                  │
│  • 595 nodes, one per method                                            │
│  • Node attributes: name, index                                         │
│                                                                          │
│  Edges:                                                                  │
│  • One edge per analyzed pair                                           │
│  • Edge attribute: compatibility_score (0.0-1.0)                        │
│                                                                          │
│  Edge Score Dictionary:                                                  │
│  • edge_scores[(min_idx, max_idx)] = score                              │
│  • Enables O(1) score lookup for any pair                               │
└────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 3: Load Category Mappings                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  From semantic clustering results:                                       │
│  • cluster_to_synergy: Maps cluster IDs to category keys               │
│  • synergy_display_names: Human-readable category names                 │
│                                                                          │
│  Build:                                                                  │
│  • index_to_category: Method index → category key                       │
│  • category_colors: Category key → {color, display_name}               │
│                                                                          │
│  20-color palette assigned to categories by frequency                   │
└────────────────────────────────────────────────────────────────────────┘
```

### Phase 2: Molecule Formation (Greedy Compatibility Algorithm)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  MOLECULE FORMATION: Greedy Highest-Compatibility-First                 │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  Goal: Group methods into molecules where internal compatibility        │
│        is maximized. High-quality molecules = tight clusters.           │
│        Low-quality molecules = loose clusters.                          │
│                                                                          │
│  Algorithm:                                                              │
│                                                                          │
│  remaining = all_nodes                                                   │
│  molecules = []                                                          │
│                                                                          │
│  while len(remaining) >= MIN_MOLECULE_SIZE:                             │
│                                                                          │
│      1. Find Best Starting Pair                                         │
│         ─────────────────────────                                        │
│         • Search remaining nodes for pair with highest compatibility    │
│         • Sample first 50 neighbors per node for speed                  │
│         • This pair seeds the new molecule                              │
│                                                                          │
│      2. Grow Molecule Greedily                                          │
│         ────────────────────────                                         │
│         while molecule.size < MAX_MOLECULE_SIZE:                        │
│             for each candidate in remaining:                            │
│                 test_group = molecule + candidate                       │
│                 avg_score = average_pairwise_compatibility(test_group)  │
│             add candidate with highest avg_score                        │
│                                                                          │
│      3. Calculate Molecule Quality                                      │
│         ──────────────────────────                                       │
│         avg_compatibility = mean of all internal pairwise scores        │
│                                                                          │
│      4. Record Molecule                                                 │
│         ───────────────                                                  │
│         molecules.append({                                              │
│             'nodes': [node_ids],                                        │
│             'size': N,                                                   │
│             'avg_compatibility': 0.XX,                                  │
│             'center': [0, 0],                                           │
│             'offsets': {},                                              │
│             'radius': 0                                                 │
│         })                                                               │
│                                                                          │
│  Result: ~60 molecules of size 3-10, sorted by quality                  │
└────────────────────────────────────────────────────────────────────────┘
```

**Average Pairwise Compatibility Calculation:**

```python
def calculate_avg_pairwise_score(nodes, edge_scores):
    """
    For a set of N nodes, calculate:
    sum(compatibility(i,j) for all i<j) / (N*(N-1)/2)
    """
    scores = []
    for i, node_a in enumerate(nodes):
        for node_b in nodes[i+1:]:
            edge_key = (min(node_a, node_b), max(node_a, node_b))
            if edge_key in edge_scores:
                scores.append(edge_scores[edge_key])
    return mean(scores) if scores else 0.0
```

### Phase 3: Internal Molecule Shaping (Force-Directed)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  INTERNAL SHAPING: Compatibility-Based Spring Forces                    │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  Key Principle:                                                          │
│  • High compatibility → strong attraction → tight cluster               │
│  • Low compatibility → weak attraction → loose cluster                  │
│                                                                          │
│  For each molecule independently:                                        │
│                                                                          │
│  1. Calculate Internal Spacing Parameters                               │
│     ─────────────────────────────────────                                │
│     avg_compat = molecule.avg_compatibility                             │
│                                                                          │
│     # Ideal distance scales inversely with compatibility                │
│     ideal_dist = 0.3 + (1.0 - avg_compat) × 2.0                        │
│                                                                          │
│     # High compat (0.95-1.0) → ideal_dist = 0.3 (tight)                │
│     # Low compat (0.5-0.7)  → ideal_dist = 1.2 (loose)                 │
│                                                                          │
│     # Attraction strength scales with compatibility                     │
│     attraction_strength = 0.2 + avg_compat × 0.6                       │
│                                                                          │
│  2. Apply Spring Forces (60 iterations)                                 │
│     ──────────────────────────────────                                   │
│     For each pair (node_a, node_b) within molecule:                     │
│                                                                          │
│         edge_score = edge_scores[(node_a, node_b)]                      │
│         edge_strength = attraction_strength × edge_score                │
│                                                                          │
│         # Spring force: F = k × (distance - ideal_distance)            │
│         force_magnitude = edge_strength × (dist - ideal_dist)           │
│         force_magnitude = clamp(force_magnitude, -0.5, 1.5)            │
│                                                                          │
│         # Apply forces symmetrically                                    │
│         forces[node_a] -= force                                         │
│         forces[node_b] += force                                         │
│                                                                          │
│  3. Apply Local Repulsion (prevent overlap)                             │
│     ────────────────────────────────────                                 │
│     For pairs closer than 0.4 units:                                    │
│         repulsion = 0.15 / (dist² + 0.05)                              │
│                                                                          │
│  4. Update Positions                                                    │
│     ────────────────                                                     │
│     positions[node] += learning_rate × forces[node]                     │
│     learning_rate decays: 0.12 → 0.12 × 0.9^(iteration/15)             │
└────────────────────────────────────────────────────────────────────────┘
```

### Phase 4: Molecule Locking

```
┌─────────────────────────────────────────────────────────────────────────┐
│  MOLECULE LOCKING                                                       │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  After internal shaping, lock each molecule's structure:                │
│                                                                          │
│  For each molecule:                                                      │
│      1. Calculate center = mean(node_positions)                         │
│                                                                          │
│      2. Store offsets: offset[node] = position[node] - center           │
│                                                                          │
│      3. Calculate radius = max(|offset|) + 0.5                          │
│         (Used for collision detection)                                  │
│                                                                          │
│      4. Mark molecule as locked                                         │
│                                                                          │
│  Result: Each molecule is now a rigid body with:                        │
│  • A center point that can move                                         │
│  • Fixed relative positions of internal nodes                           │
│  • A bounding radius for repulsion                                      │
└────────────────────────────────────────────────────────────────────────┘
```

### Phase 5: Inter-Molecular Positioning

```
┌─────────────────────────────────────────────────────────────────────────┐
│  INTER-MOLECULAR POSITIONING: Force-Directed Layout                     │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  Initial Placement:                                                      │
│  • Arrange molecule centers in a spiral                                 │
│  • angle = 2π × mol_id / num_molecules                                 │
│  • radius = 2.0 + mol_id × 0.2                                         │
│                                                                          │
│  Force Simulation (300 iterations):                                     │
│                                                                          │
│  FORCE 1: Inter-Molecular Repulsion                                     │
│  ───────────────────────────────────                                     │
│  For each molecule pair (mol_i, mol_j):                                 │
│                                                                          │
│      desired_min_dist = 2.5 × (radius_i + radius_j)                    │
│      actual_dist = |center_i - center_j|                               │
│                                                                          │
│      if actual_dist < desired_min_dist:                                 │
│          # Strong repulsion when too close                              │
│          force = 3.5 × (desired_min_dist - dist) / (dist + 0.1)        │
│          force = min(force, 8.0)                                        │
│          Apply repulsion along center-to-center axis                    │
│                                                                          │
│      elif actual_dist < desired_min_dist × 1.5:                        │
│          # Gentle repulsion in comfort zone                             │
│          force = 1.0 × (desired_min_dist × 1.5 - dist) / dist          │
│          force = min(force, 2.0)                                        │
│                                                                          │
│  FORCE 2: Compatibility-Weighted Attraction                             │
│  ───────────────────────────────────────────                             │
│  For each molecule pair (mol_i, mol_j):                                 │
│                                                                          │
│      # Sample cross-molecule compatibility                              │
│      compat_scores = []                                                 │
│      for node_i in mol_i.nodes[:20]:                                   │
│          for node_j in mol_j.nodes[:20]:                               │
│              if edge_exists(node_i, node_j):                            │
│                  compat_scores.append(edge_score)                       │
│                                                                          │
│      avg_inter_compat = mean(compat_scores)                            │
│                                                                          │
│      # Only attract if compatible and not too close                    │
│      if dist > desired_min × 1.8 and dist < 20 and compat > 0.70:      │
│          force = 0.5 × avg_inter_compat × (dist - desired × 1.8)       │
│          force = min(force, 1.5)                                        │
│          Apply attraction along center-to-center axis                   │
│                                                                          │
│  Update Positions:                                                       │
│  • Move molecule centers by force × learning_rate                       │
│  • Update all node positions: pos = center + offset                     │
│  • learning_rate decays: 0.12 × 0.93^(iteration/40)                    │
└────────────────────────────────────────────────────────────────────────┘
```

### Phase 6: Edge Selection for Visualization

```
┌─────────────────────────────────────────────────────────────────────────┐
│  EDGE SELECTION: Peripheral Edges Only                                  │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  Internal Edges (within molecules):                                      │
│  • Draw ALL internal edges                                              │
│  • Color by compatibility:                                              │
│    - score >= 0.95: dark green (synergistic)                           │
│    - score >= 0.75: light green (compatible)                           │
│    - score < 0.75: red (incompatible)                                  │
│                                                                          │
│  External Edges (between molecules):                                     │
│  ─────────────────────────────────                                       │
│  Problem: Drawing all ~175K external edges creates visual noise         │
│                                                                          │
│  Solution: Smart Selection Algorithm                                     │
│                                                                          │
│  1. For each molecule pair, calculate avg cross-compatibility           │
│     (sample up to 15×15 = 225 pairs for speed)                         │
│                                                                          │
│  2. Skip pairs with avg_compat < 0.75 (not interesting)                │
│                                                                          │
│  3. For each molecule, rank other molecules by distance                 │
│                                                                          │
│  4. Keep only 3 closest compatible connections per molecule             │
│                                                                          │
│  5. For each kept connection:                                           │
│     • Find the two closest nodes between the molecules                  │
│     • Draw edge only if their score >= 0.80                            │
│     • Edge alpha = 0.3 + (score - 0.80) × 2.0                          │
│                                                                          │
│  Result: ~100-200 meaningful external edges instead of ~175K           │
│  Benefit: Edges show which molecules are well-connected                │
└────────────────────────────────────────────────────────────────────────┘
```

### Phase 7: Visualization Rendering

```
┌─────────────────────────────────────────────────────────────────────────┐
│  VISUALIZATION: Static PNG + Interactive HTML                           │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  LAYER 0: Molecule Boundaries (background)                              │
│  ─────────────────────────────────────────                               │
│  For molecules with 3+ nodes:                                           │
│  • Compute convex hull of node positions                                │
│  • Fill with color based on quality:                                    │
│    - avg_compat > 0.90: green fill (α=0.15)                            │
│    - avg_compat > 0.75: yellow fill (α=0.12)                           │
│    - avg_compat <= 0.75: red fill (α=0.15)                             │
│  • Draw dashed boundary line                                            │
│                                                                          │
│  LAYER 1: Internal Edges                                                │
│  ───────────────────────                                                 │
│  • Synergistic (≥0.95): dark green, α=0.4, width=1.5                   │
│  • Compatible (≥0.75): light green, α=0.25, width=1.0                  │
│  • Incompatible (<0.75): red, α=0.3, width=1.2                         │
│                                                                          │
│  LAYER 2: External Edges                                                │
│  ───────────────────────                                                 │
│  • Green lines connecting peripheral nodes                              │
│  • Alpha varies with compatibility (0.3-0.7)                           │
│                                                                          │
│  LAYER 3: Nodes (foreground)                                            │
│  ───────────────────────────                                             │
│  • Color: By semantic category (20-color palette)                       │
│  • Size: log(degree + 2) × 150                                         │
│  • White edge for visibility                                            │
│                                                                          │
│  LEGEND                                                                  │
│  ──────                                                                  │
│  • Categories sorted by frequency (most common first)                  │
│  • Shows count per category                                             │
│                                                                          │
│  OUTPUT                                                                  │
│  ──────                                                                  │
│  • results/viz_molecular_final.png (36×36 inch, 300 DPI)               │
│  • results/viz_molecular_interactive.html (Plotly, 1600×1600)          │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Output Files

### Static Visualization

**File:** `results/viz_molecular_final.png`

- Resolution: 36×36 inches at 300 DPI (10,800×10,800 pixels)
- Format: PNG with white background
- Shows all molecules with boundaries, internal/external edges, and colored nodes

### Interactive Visualization

**File:** `results/viz_molecular_interactive.html`

- Built with Plotly.js
- Hover to see method details:
  - Method name
  - Source
  - Category
  - Molecule quality
  - Connection count
- Pan/zoom enabled
- Legend toggles category visibility

---

## Visual Encoding Summary

| Visual Property | Data Encoded | Mapping |
|-----------------|--------------|---------|
| **Node color** | Semantic category | 20-color palette |
| **Node size** | Connection degree | log(degree + 2) × scale |
| **Molecule boundary color** | Internal compatibility | Green (>0.90), Yellow (>0.75), Red (≤0.75) |
| **Molecule tightness** | Internal compatibility | High compat = tight, Low compat = loose |
| **Internal edge color** | Pairwise compatibility | Green (≥0.95), Light green (≥0.75), Red (<0.75) |
| **External edge alpha** | Cross-molecule compatibility | 0.3 + (score - 0.80) × 2.0 |
| **Molecule proximity** | Cross-molecule compatibility | Compatible molecules attract |

---

## Key Insights from Visualization

1. **Tight Green Clusters**: Highly synergistic method groups—use together for maximum benefit

2. **Loose Red Clusters**: "Bad actor" molecules—methods that don't work well together but were grouped due to lack of better options

3. **External Green Edges**: Show which molecules are complementary—can combine methods from connected molecules

4. **Color Distribution**: Shows how semantic categories (from clustering) align with compatibility (from pairwise analysis)

5. **Isolated Molecules**: Methods with few external connections may be specialized or niche

---

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Input pairs | 176,715 |
| Methods visualized | 595 |
| Molecules formed | ~60 |
| Internal shaping iterations | 60 per molecule |
| Inter-molecular iterations | 300 |
| External edges displayed | ~100-200 (from ~175K) |
| PNG render time | ~30 seconds |
| HTML render time | ~10 seconds |

---

## Algorithm Design Rationale

### Why Greedy Molecule Formation?

Alternative approaches considered:
- **Community detection** (Louvain, modularity): Doesn't guarantee max size limits
- **K-means on embeddings**: Ignores pairwise compatibility structure
- **Hierarchical clustering**: Produces unbalanced sizes

Greedy highest-compatibility-first:
- Guarantees molecule size bounds (3-10)
- Maximizes internal compatibility by construction
- Produces naturally varying quality levels
- Computationally efficient (O(N²) with sampling)

### Why Force-Directed Layout?

- Internal forces create visual metaphor: tight = good, loose = bad
- Lock-then-position allows independent optimization phases
- Inter-molecular forces prevent overlap while showing relationships
- Familiar graph visualization paradigm

### Why Peripheral Edge Selection?

- Full edge drawing (175K edges) creates unreadable hairball
- 3-closest-neighbors-per-molecule balances sparsity and connectivity
- Peripheral-only edges ensure no edge crosses through a molecule
- Compatibility threshold (0.80) ensures only meaningful edges shown
