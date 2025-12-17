# Semantic Clustering of Product Development Methods

This document describes the semantic clustering architecture used to group 595 product development methods into a three-level hierarchy: Methods → Clusters → Categories.

## Overview

The semantic clustering pipeline transforms textual method descriptions into numerical embeddings and applies a two-pass clustering approach to discover natural groupings:

1. **Primary Pass (HDBSCAN)** - Density-based clustering identifies 21 primary clusters (P0-P20)
2. **Secondary Pass (HDBSCAN)** - Re-clusters noise points into 26 secondary clusters (S0-S25)
3. **Category Extraction** - Groups similar clusters into 16 categories using UMAP 5D + Ward linkage

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INPUT LAYER                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  input/methods_deduplicated.csv     input/cluster_names.json                │
│  (595 methods)                      (Source of truth for cluster names)     │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EMBEDDING LAYER                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  semantic_clustering.py                                                      │
│  ├── Text: "{name}. {description}"                                          │
│  ├── Model: bge-large-en (1024 dimensions)                                  │
│  └── Output: embeddings.npy (595 x 1024)                                    │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      TWO-PASS CLUSTERING LAYER                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  semantic_clustering_combined.py                                             │
│                                                                              │
│  PASS 1: Primary Clustering                                                  │
│  ├── UMAP: 1024D → 5D (n_neighbors=15, min_dist=0.1, cosine)               │
│  ├── HDBSCAN: min_cluster_size=10, min_samples=3                            │
│  └── Result: 21 primary clusters (P0-P20) + noise points                    │
│                                                                              │
│  PASS 2: Secondary Clustering (noise re-clustering)                          │
│  ├── Input: ~150 noise points from Pass 1                                   │
│  ├── UMAP: 1024D → 5D (n_neighbors=10, min_dist=0.05)                      │
│  ├── HDBSCAN: min_cluster_size=5, min_samples=2                             │
│  └── Result: 26 secondary clusters (S0-S25) + remaining unclustered (U)     │
│                                                                              │
│  Output: combined_clusters.json (47 clusters: P0-P20, S0-S25, U)            │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      CATEGORY EXTRACTION LAYER                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  extract_dendrogram_synergies.py                                             │
│                                                                              │
│  1. Compute cluster centroids (1024D)                                        │
│  2. UMAP reduce centroids: 1024D → 5D                                       │
│  3. Ward linkage on 5D centroids                                            │
│  4. Cut dendrogram at distance threshold                                     │
│  5. LLM generates category names                                             │
│                                                                              │
│  Output: dendrogram_categories.json (16 categories)                          │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CONSUMER LAYER                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  cluster_utils.py - Unified access to cluster/category mappings              │
│  ├── load_cluster_mappings() → cluster names, cluster→category mapping      │
│  └── get_category_display_name(cluster_id) → "Category Name"                │
│                                                                              │
│  Consumers:                                                                  │
│  ├── build_method_portfolios.py - Toolkit generation with category synergies│
│  ├── create_12d_dashboard.py - 12D visualization with category colors       │
│  ├── create_toolkit_scientific_viz.py - Toolkit visualization               │
│  └── visualize_graph.py - Molecular compatibility visualization             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Three-Level Hierarchy

### Level 1: Methods (595)
Individual product development practices from the deduplicated dataset.

### Level 2: Clusters (47)
Semantic groupings discovered by HDBSCAN:
- **21 Primary Clusters (P0-P20)**: Dense, well-defined clusters from initial pass
- **26 Secondary Clusters (S0-S25)**: Clusters formed from noise re-clustering
- **Unclustered (U)**: Methods that don't fit any cluster

### Level 3: Categories (16)
Higher-level groupings of related clusters, derived from UMAP 5D Ward linkage:

| Category | Clusters | Theme |
|----------|----------|-------|
| Team Collaboration & Feedback Systems | P15, S11, S13, S18 | Communication, sensing, coordination |
| Cross-Functional Organizational Design | P9, S21 | Team structure, Conway's Law |
| Team Empowerment & Alignment | P19, P20, S14, S22 | Culture, autonomy, cognitive load |
| Adaptive Architecture & Planning | P11, S9, S15, S19 | Modularity, risk, scenario planning |
| Design for Integration & Simplicity | P6, P8, S4 | DFX, supplier integration |
| Systems Engineering & Integration | P7, S0, S17 | Requirements, concurrent engineering |
| Lean Culture & Quality | P4, S24, S25 | Leadership, JIT, built-in quality |
| Project Management & Stakeholder Visibility | P0, P3, S7 | Scheduling, metrics, stakeholders |
| Decision Analytics & Strategy | P10, S6 | Trade-offs, financial decisions |
| Process Optimization & Capacity | P5, S2, S5 | Flow, queuing theory, systems thinking |
| Leadership Development & Coaching | P14, S3 | Mentoring, servant leadership |
| Continuous Improvement Cycles | P13, S23 | Retrospectives, Kaizen/Kata |
| Real-Time Adaptive Problem Solving | P1, S20 | Problem solving, process stability |
| Agile Scaling & Transformation | P12, P18, S1, S12 | Scrum, scaling, hybrid models |
| Continuous Delivery & Experimentation | P16, S8, S16 | TDD, CI, simulation |
| Customer-Centric Incremental Delivery | P2, P17, S10 | Customer value, prototyping |

## Key Files

### Input Files
| File | Description |
|------|-------------|
| `input/methods_deduplicated.csv` | 595 methods with Index, Method, Description, Source |
| `input/cluster_names.json` | **Source of truth** for cluster names (manually edited) |

### Generated Files
| File | Description |
|------|-------------|
| `results_semantic_clustering/embeddings.npy` | Cached embeddings (595 x 1024) |
| `results_semantic_clustering_combined/combined_clusters.json` | All 47 clusters with methods |
| `results_semantic_clustering_combined/dendrogram_categories.json` | 16 categories with cluster groupings |
| `results_semantic_clustering_combined/combined_dendrogram_umap5d.png` | Category dendrogram (UMAP 5D based) |

### Utility Files
| File | Description |
|------|-------------|
| `cluster_utils.py` | Unified API for loading cluster/category mappings |
| `config.yaml` | LLM and embedding configuration |

## Configuration

### Embedding (config.yaml)
```yaml
embedding:
  model: bge-large-en
  base_url: http://192.168.0.136:9003/v1
  batch_size: 25
```

### LLM for Category Naming (config.yaml)
```yaml
llm:
  model: gpt-oss-20b
  base_url: http://192.168.0.247:9003/v1
  temperature: 0.1
```

### UMAP Parameters
```python
# For clustering (5D)
umap.UMAP(
    n_components=5,
    n_neighbors=15,      # Primary pass
    min_dist=0.1,
    metric='cosine',
    random_state=42
)

# For noise re-clustering (5D, tighter)
umap.UMAP(
    n_components=5,
    n_neighbors=10,
    min_dist=0.05,
    metric='cosine',
    random_state=42
)
```

### HDBSCAN Parameters
```python
# Primary pass
hdbscan.HDBSCAN(
    min_cluster_size=10,
    min_samples=3,
    metric='euclidean',
    cluster_selection_method='eom'
)

# Secondary pass (noise re-clustering)
hdbscan.HDBSCAN(
    min_cluster_size=5,
    min_samples=2,
    metric='euclidean',
    cluster_selection_method='eom'
)
```

## Running the Pipeline

```bash
# Activate virtual environment
source venv/bin/activate

# Step 1: Generate embeddings (if not cached)
python semantic_clustering.py

# Step 2: Run two-pass clustering
python semantic_clustering_combined.py

# Step 3: Extract categories from dendrogram
python extract_dendrogram_synergies.py

# Step 4: (Optional) Update cluster names manually
# Edit input/cluster_names.json, then re-run Step 2-3
```

## Quality Metrics

### Silhouette Scores
| Space | Score | Interpretation |
|-------|-------|----------------|
| 1024D | 0.029 | Poor (expected for high-dimensional text) |
| UMAP 5D | 0.245 | Good (UMAP creates better separation) |

The low 1024D silhouette score is expected because:
1. Product development methods have high semantic overlap
2. Methods exist on a continuum rather than discrete groups
3. High-dimensional spaces suffer from the curse of dimensionality

UMAP projection improves cluster separation while preserving semantic relationships.

### Cluster Membership Probability

HDBSCAN assigns probability scores (0.0 - 1.0) for each method:

| Probability | Interpretation |
|-------------|----------------|
| 0.90-1.00 | Core member - definitive example of cluster theme |
| 0.70-0.90 | Strong member - clearly belongs to cluster |
| 0.50-0.70 | Borderline - could fit multiple clusters |
| < 0.50 | Weak assignment - on cluster periphery |

## Dependencies

- `numpy`, `pandas` - Data handling
- `umap-learn` - Dimensionality reduction
- `hdbscan` - Density-based clustering
- `scipy` - Hierarchical clustering (Ward linkage)
- `matplotlib`, `plotly` - Visualization
- `aiohttp` - LLM API calls
- `pyyaml` - Configuration
