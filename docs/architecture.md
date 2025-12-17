# Computational Methods Analysis Tool - Architecture

## Overview

This tool analyzes product development methods through a multi-stage pipeline: deduplication → categorization → 12-dimensional analysis → visualization → compatibility analysis → graph clustering visualization → toolkit generation → toolkit visualization.

The system processes 595 deduplicated methods, scoring each across 12 dimensions, analyzing 176,715 pairwise compatibility relationships with rich overlap analysis, creating molecular cluster visualizations showing natural groupings, and generating context-specific toolkits using data-driven synergy detection.

## Pipeline Architecture

```
┌─────────────────┐
│  methods.csv    │  Raw input: 800+ methods with descriptions
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  Stage 1: DEDUPLICATION                             │
│  ─────────────────────────────────────────────      │
│  Scripts:                                           │
│    • find_exact_duplicates.py                       │
│    • find_and_merge_duplicates.py                   │
│    • merge_duplicates.py                            │
│    • review_semantic_duplicates.py                  │
│                                                     │
│  Uses: embeddings.py, duplicate_synthesizer.py     │
│  Method: Cosine similarity (threshold: 0.9)        │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  methods_deduplicated.csv           │  ~595 unique methods
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  Stage 2: CATEGORIZATION                            │
│  ──────────────────────────────────────────────     │
│  Scripts:                                           │
│    • assign_intelligent_categories.py               │
│    • analyze_category_fit.py                        │
│                                                     │
│  Uses: intelligent_categories.py                   │
│  Output: categories.json, abstraction.json         │
│  Dimensions: Domain, abstraction level, etc.       │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  Stage 3: 12-DIMENSIONAL ANALYSIS                   │
│  ──────────────────────────────────────────────     │
│  Scripts:                                           │
│    • analyze_12d.py (main orchestrator)             │
│    • filter_methods_for_analysis.py (optional)      │
│                                                     │
│  Uses: ranking_analyzer.py                         │
│                                                     │
│  13 Dimensions (12 + 1 derived):                    │
│    1. Scope (Tactical ↔ Strategic)                  │
│    2. Temporality (Immediate ↔ Evolutionary)        │
│    3. Ease of Adoption (Hard ↔ Easy)                │
│    4. Resources Required (Low ↔ High)               │
│    5. Technical Complexity (Simple ↔ Complex)       │
│    6. Change Management Difficulty (Easy ↔ Hard)    │
│    7. Impact Potential (Low ↔ High)                 │
│    8. Time to Value (Slow ↔ Fast)                   │
│    9. Applicability (Niche ↔ Universal)             │
│   10. People Focus (Technical/System ↔ Human)       │
│   11. Process Focus (Ad-hoc ↔ Systematic)           │
│   12. Purpose Orientation (Internal ↔ External)     │
│   13. Implementation Difficulty (derived composite) │
│                                                     │
│  Method:                                            │
│    • LLM-based ranking (5-round validation)         │
│    • Chunked comparison (size: 10, overlap: 4)      │
│    • Cross-validation for consistency               │
│    • Calibrated scoring (realistic distributions)   │
│                                                     │
│  Output:                                            │
│    • method_scores_12d_deduplicated.json            │
│    • scope_temporality_ranked_deduplicated.json     │
│                                                     │
│  Performance: ~75-100 min for 595 methods           │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  Stage 4: VISUALIZATION & REPORTING                 │
│  ──────────────────────────────────────────────     │
│  Scripts:                                           │
│    • create_12d_dashboard.py (main visualizer)      │
│    • analyze_overlap.py                             │
│                                                     │
│  Uses: visualize.py, report.py                     │
│                                                     │
│  Output: 7 Interactive Visualizations               │
│                                                     │
│  2D Plots (4):                                      │
│    1. Impact vs Implementation Difficulty           │
│       X: Implementation (Easy ↔ Hard)               │
│       Y: Impact Potential (Low ↔ High)              │
│                                                     │
│    2. Scope vs Temporality                          │
│       X: Scope (Tactical ↔ Strategic)               │
│       Y: Temporality (Immediate ↔ Evolutionary)     │
│                                                     │
│    3. Time to Value vs Impact                       │
│       X: Time to Value (Slow ↔ Fast)                │
│       Y: Impact Potential (Low ↔ High)              │
│                                                     │
│    4. People vs Process Focus                       │
│       X: People (Technical/System ↔ Human)          │
│       Y: Process (Ad-hoc ↔ Systematic)              │
│                                                     │
│  3D Plots (3):                                      │
│    5. The Strategic Cube                            │
│       X: Scope (Tactical ↔ Strategic)               │
│       Y: Impact (Low ↔ High)                        │
│       Z: Implementation (Easy ↔ Hard)               │
│                                                     │
│    6. People × Process × Purpose Space              │
│       X: People (Technical/System ↔ Human)          │
│       Y: Process (Ad-hoc ↔ Systematic)              │
│       Z: Purpose (Internal ↔ External)              │
│                                                     │
│    7. The Adoption Space                            │
│       X: Ease of Adoption (Hard ↔ Easy)             │
│       Y: Change Mgmt Difficulty (Easy ↔ Hard)       │
│       Z: Time to Value (Slow ↔ Fast)                │
│                                                     │
│  Features:                                          │
│    • Category-based color coding (13 categories)    │
│    • Independent dropdown filters per plot          │
│    • Dual-end axis labels for 3D plots              │
│    • Grid-attached labels (rotate with plot)        │
│    • Interactive hover with method details          │
│                                                     │
│  Static Outputs:                                    │
│    • evaluation_dashboard.png                       │
│    • subcriteria_analysis.png                       │
│    • category_heatmap.png                           │
│    • category_4field_matrix.png                     │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  Stage 5: COMPATIBILITY ANALYSIS                    │
│  ──────────────────────────────────────────────     │
│  Scripts:                                           │
│    • analyze_compatibility.py (main analyzer)       │
│                                                     │
│  Method:                                            │
│    • LLM-based pairwise compatibility scoring       │
│    • Smart sampling (reduces comparisons by 90%)    │
│    • Overlap analysis (7 fields):                   │
│      - same_problem, same_role, same_timing         │
│      - same_output, causes_confusion                │
│      - overlap_type (none/partial/conflicting/      │
│        redundant)                                   │
│      - has_problematic_overlap                      │
│                                                     │
│  Classification Rules:                              │
│    • Incompatible: score < 0.7                      │
│    • Synergistic: score >= 0.95 AND not conflicting│
│    • Nonrelated: different problems, no overlap     │
│    • Compatible: everything else                    │
│                                                     │
│  Output:                                            │
│    • compatibility_checkpoint.pkl (~176,715 pairs)  │
│    • compatibility_analysis.json                    │
│      - top_incompatibilities (with overlap data)    │
│      - top_synergies (with overlap data)            │
│      - metadata and statistics                      │
└────────┬────────────────────────────────────────────┘
         │
         ├────────────────────────────────────────────┐
         │                                            │
         ▼                                            ▼
┌─────────────────────────────────────────────────────┐
│  Stage 5A: GRAPH CLUSTERING VISUALIZATION           │
│  (Parallel to Toolkit Generation)                   │
│  ──────────────────────────────────────────────     │
│  Scripts:                                           │
│    • visualize_graph.py (molecular layout engine)   │
│                                                     │
│  Method: Greedy Molecular Clustering                │
│    • Form molecules by highest compatibility first  │
│    • Molecule size: 3-10 methods                    │
│    • Internal forces scale with compatibility:      │
│      - High compat (0.95-1.0) → tight spacing (0.3) │
│      - Low compat (0.5-0.7) → loose spacing (1.2)   │
│    • Bad actors naturally form large loose clusters │
│                                                     │
│  Layout Algorithm (2-Phase):                        │
│    Phase 1: Shape molecules internally              │
│      • Pairwise attraction based on edge scores     │
│      • Spring forces (F = k × (dist - ideal_dist))  │
│      • Local repulsion (prevent overlap)            │
│      • 60 iterations with learning rate decay       │
│                                                     │
│    Phase 2: Position molecules                      │
│      • Inter-molecular repulsion (2.5× sum radii)   │
│      • Compatibility-weighted attraction            │
│      • Only connect 3 closest molecules per cluster │
│      • 300 iterations for convergence               │
│                                                     │
│  Edge Classification:                               │
│    • Incompatible: score < 0.75 (red, internal)     │
│    • Compatible: score ≥ 0.75 (light green)         │
│    • Synergistic: score ≥ 0.95 (dark green)         │
│    • External: peripheral edges only (≥0.80)        │
│                                                     │
│  Visualization Features:                            │
│    • Convex hull boundaries colored by quality:     │
│      - Green (compat > 0.90): tight, high-quality   │
│      - Yellow (compat > 0.75): medium quality       │
│      - Red (compat ≤ 0.75): loose, problematic      │
│    • Node size: log(degree + 2) × 150               │
│    • Node color: category-based (13 categories)     │
│    • Legend: sorted by category frequency           │
│    • Smart edge drawing:                            │
│      - All internal edges within molecules          │
│      - Only closest peripheral edges between        │
│        molecules (reduces visual clutter)           │
│                                                     │
│  Natural Emergence Properties:                      │
│    ✓ High-quality molecules are tight and small     │
│    ✓ Low-quality molecules are loose and large      │
│    ✓ Peripheral edges only between molecules        │
│    ✓ No edge passes through a molecule              │
│                                                     │
│  Output:                                            │
│    • viz_molecular_final.png                        │
│      (36×36 inches, 300 DPI, ~60-70 molecules)      │
│                                                     │
│  Statistics Reported:                               │
│    • Total molecules formed                         │
│    • Highest/lowest molecule quality (avg compat)   │
│    • Average molecule size (nodes)                  │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  Stage 6: TOOLKIT GENERATION                        │
│  ──────────────────────────────────────────────     │
│  Scripts:                                           │
│    • build_method_portfolios.py                     │
│                                                     │
│  Context Profiles (5):                              │
│    1. Startup - MVP Phase                           │
│    2. Startup - Scaling Phase                       │
│    3. Enterprise - Digital Transformation           │
│    4. Regulated Industry - Compliance Focus         │
│    5. Hardware Product Development                  │
│                                                     │
│  Selection Score Formula:                           │
│    Score = Fitness×0.4 +                            │
│            Avg_Compat×100×0.25 +                    │
│            Min_Compat×100×0.20 +                    │
│            Diversity +                              │
│            Synergy×100×0.15                         │
│                                                     │
│  Fitness Calculation:                               │
│    • Context-specific dimension weights             │
│    • MVP (5 dims): time_to_value (0.35),            │
│      ease_adoption (0.25), resources (-0.20),       │
│      impact_potential (0.15), applicability (0.05)  │
│    • Scaling (6 dims): scope, impact, process,      │
│      time_to_value, technical_complexity,           │
│      change_mgmt_difficulty                         │
│    • Enterprise (4 dims): scope, temporality,       │
│      impact, change_mgmt_difficulty                 │
│    • Regulated (4 dims): process, technical,        │
│      applicability, ease_adoption                   │
│      Note: risk_decision_making defined but missing │
│    • Hardware (2 dims): time_to_value, impact       │
│      Note: 3 dimensions defined but missing         │
│    • Normalized to 0-100 scale                      │
│                                                     │
│  Pairwise Synergy Detection:                        │
│    • Uses actual overlap_analysis data              │
│    • +0.3 per synergistic pair (score ≥0.95)        │
│    • +0.15 per complementary pair (same problem,    │
│      different approach)                            │
│    • -0.2 per problematic overlap (conflicts)       │
│    • Can go negative for conflict-prone methods     │
│                                                     │
│  Diversity Scoring:                                 │
│    • +10 points if new category                     │
│    • 0 points if category already in toolkit        │
│                                                     │
│  Output:                                            │
│    • toolkit_comparison.json                        │
│      - Methods with diversity_score, synergy_score  │
│      - Selection metadata and statistics            │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  Stage 7: TOOLKIT VISUALIZATION                     │
│  ──────────────────────────────────────────────     │
│  Scripts:                                           │
│    • create_toolkit_dashboard.py               │
│                                                     │
│  Three Visualizations per Context:                  │
│                                                     │
│  Panel A: Methods Table                             │
│    • Method names with categories                   │
│    • Fitness scores (0-100)                         │
│    • Compatibility scores (0-1 scale)               │
│    • Impact potential (0-100)                       │
│    • Dynamic height (no scrollbars)                 │
│                                                     │
│  Panel B: Implementation Roadmap                    │
│    • Scatter plot visualization                     │
│    • X-axis: Implementation Difficulty (0-100)      │
│    • Y-axis: Time to Value (0-100)                  │
│    • Bubble size: Impact Potential                  │
│    • Color: Impact (RdYlGn colorscale)              │
│    • Text: Method numbers (adaptive contrast)       │
│      - Black text for yellow middle (40-70)         │
│      - White text for red/green ends                │
│    • Strategic quadrant lines (50/50 split)         │
│                                                     │
│  Panel C: Score Composition Table                   │
│    • Breakdown of selection scores                  │
│    • Fitness contribution (40% weight)              │
│    • Avg Compat contribution (25% weight)           │
│    • Min Compat contribution (20% weight)           │
│    • Diversity contribution (10% weight, 0 or 10)   │
│    • Synergy contribution (15% weight, -7.5 to +15) │
│    • Total selection score                          │
│    • Dynamic height (no scrollbars)                 │
│                                                     │
│  Explanation Section:                               │
│    • Detailed breakdown of each score component     │
│    • Context-specific fitness weights explained     │
│    • Compatibility calculation rationale            │
│    • Diversity and synergy scoring logic            │
│                                                     │
│  Output:                                            │
│    • toolkit_{context}_analysis.html (combined)     │
│    • toolkit_{context}_a_methods.html               │
│    • toolkit_{context}_b_roadmap.html               │
│    • toolkit_{context}_c_scores.html                │
└─────────────────────────────────────────────────────┘
```

## Core Modules (`src/`)

### Data Processing
- **data.py**: CSV loading, DataFrame handling
- **embeddings.py**: BGE-large-en embedding generation (batch: 25, concurrent: 50)
- **duplicate_synthesizer.py**: Semantic duplicate merging logic

### Analysis
- **ranking_analyzer.py**: 12D ranking engine with LLM-based comparison
- **intelligent_categories.py**: Category assignment and validation
- **sampler.py**: Smart sampling strategies for pairwise comparisons

### Compatibility & Toolkit Building
- **analyze_compatibility.py**: LLM-based compatibility scoring with overlap analysis
- **build_method_portfolios.py**: Context-aware toolkit generation with:
  - CompatibilityMatrix class (stores scores + full overlap_analysis)
  - SituationalToolkitBuilder (greedy selection with pairwise synergy)
  - 5 context profiles with dimension-specific weights

### Visualization
- **visualize.py**: Chart generation (matplotlib/plotly)
- **report.py**: HTML report generation with interactive elements
- **create_12d_dashboard.py**: 12D interactive dashboard (4 2D + 3 3D plots)
- **visualize_graph.py**: Molecular layout clustering visualization with:
  - Greedy molecule formation by compatibility
  - Two-phase physics-based layout (internal + inter-molecular forces)
  - Smart peripheral edge drawing (3 closest per molecule)
  - Convex hull boundaries colored by molecule quality
- **create_toolkit_dashboard.py**: Three-panel toolkit visualizations with explanation sections

## Configuration (`config.yaml`)

### LLM Setup
- Provider: OpenAI-compatible (local VLLM server)
- Model: gpt-oss-20b
- Endpoint: http://192.168.0.247:9003/v1
- Concurrency: 25 parallel requests
- Temperature: 0.1 (consistent rankings)

### Embedding Setup
- Model: bge-large-en
- Endpoint: http://192.168.0.136:9003/v1
- Batch size: 25
- Max tokens: 450

### Reranking (Optional)
- Model: bge-reranker-v2-m3
- Initial K: 20 → Final K: 5

### Analysis Parameters
- Duplicate threshold: 0.9 cosine similarity
- Ranking: 5-round validation, consistency threshold 0.75
- Sampling: 10% same-source, 5% cross-source, 3000 medium-similarity pairs

## Data Flow

### Input Files
```
input/
├── methods.csv                    # Original 800+ methods
├── method_categories.json         # Category definitions
└── methods_deduplicated.csv       # After deduplication
```

### Output Files
```
results/
├── duplicates.json                          # Detected duplicates
├── categories.json                          # Method categorization
├── abstraction.json                         # Abstraction level analysis
├── method_scores_12d_deduplicated.json      # 13D scores (12 + 1 derived)
├── scope_temporality_ranked_deduplicated.json
├── compatibility_checkpoint.pkl             # Full compatibility data (176,715 pairs)
├── compatibility_analysis.json              # Top incompatibilities & synergies
├── viz_molecular_final.png                  # Graph clustering visualization (~60-70 molecules)
├── toolkit_comparison.json                  # Generated toolkits with scores
├── interactive_dashboard.html               # 12D dashboard (4 2D + 3 3D plots)
├── evaluation_dashboard.png                 # Static 12-panel dashboard
├── subcriteria_analysis.png                 # Implementation difficulty breakdown
├── category_heatmap.png                     # Category distribution
├── category_4field_matrix.png               # 4-field category matrix
├── toolkit_startup_mvp_analysis.html        # Toolkit visualization (MVP)
├── toolkit_startup_scaling_analysis.html    # Toolkit visualization (Scaling)
├── toolkit_enterprise_transformation_analysis.html  # Toolkit viz (Enterprise)
├── toolkit_regulated_industry_analysis.html # Toolkit visualization (Regulated)
├── toolkit_hardware_product_analysis.html   # Toolkit visualization (Hardware)
└── report.html                              # Summary report
```

## Key Features

### 1. Smart Deduplication
- Exact string matching (find_exact_duplicates.py)
- Semantic similarity using embeddings (find_and_merge_duplicates.py)
- Manual review workflow (review_semantic_duplicates.py)
- Synthesized descriptions for merged duplicates

### 2. Multi-Dimensional Ranking
- 12 independent dimensions covering effort, impact, and scope
- LLM-based pairwise comparison (more accurate than direct scoring)
- Multi-pass validation (5 rounds) for consistency
- Chunked processing with overlap for calibration
- Cross-validation to detect ranking inconsistencies

### 3. Intelligent Categorization
- Domain classification (UX, Engineering, Strategy, etc.)
- Abstraction levels (High/Medium/Low)
- Compatibility grouping
- Category fit analysis

### 4. Interactive 12D Visualization Dashboard
- **4 2D plots**: Impact vs Implementation, Scope vs Temporality, Time to Value vs Impact, People vs Process
- **3 3D plots**: Strategic Cube, People×Process×Purpose Space, Adoption Space
- Category-based color coding (13 categories)
- Independent dropdown filters per plot
- Dual-end axis labels (visible from any angle)
- Grid-attached directional labels (rotate with 3D plots)
- Interactive hover with method details
- Static dashboards: 12-panel overview, subcriteria analysis

### 5. Compatibility Analysis with Overlap Detection
- LLM-based pairwise compatibility scoring (0-1 scale)
- Rich overlap analysis with 7 fields per pair
- Refined classification rules:
  - Incompatible: score < 0.7
  - Synergistic: score >= 0.95 AND not conflicting
  - Nonrelated: different problems, no overlap
- Smart sampling reduces O(n²) to ~10% of comparisons
- Detects 7,318 synergistic pairs and 30,589 problematic overlaps

### 5A. Graph Clustering Visualization (Molecular Layout)
- **Greedy molecular clustering** by highest compatibility first
- **Physics-based layout** with compatibility-scaled forces:
  - High compatibility (0.95-1.0) → tight molecules (spacing: 0.3)
  - Low compatibility (0.5-0.7) → loose molecules (spacing: 1.2)
  - Bad actors naturally form large, loose clusters
- **Two-phase algorithm**:
  - Phase 1: Shape each molecule internally (60 iterations)
  - Phase 2: Position molecules with inter-molecular forces (300 iterations)
- **Smart edge visualization**:
  - All internal edges within molecules
  - Only 3 closest peripheral edges per molecule (reduces clutter)
  - Color-coded by compatibility (red/green)
- **Visual quality indicators**:
  - Convex hull boundaries colored by molecule quality (green/yellow/red)
  - Node size scales with connectivity (log(degree + 2) × 150)
  - Node color by category (13 categories, sorted by frequency)
- **Natural emergence**: High-quality methods cluster tightly, problematic methods form loose groups
- **Output**: viz_molecular_final.png (36×36 inches, 300 DPI, ~60-70 molecules)

### 6. Context-Aware Toolkit Generation
- 5 organizational context profiles with specific dimension weights
- Multi-factor selection scoring:
  - Fitness (40%): Context-specific weighted sum of 4-6 dimensions
  - Avg Compatibility (25%): Harmony with entire toolkit
  - Min Compatibility (20%): No weak links/conflicts
  - Diversity (10%): Category coverage (0 or 10 points)
  - Synergy (15%): Pairwise relationship analysis using overlap data
- Data-driven synergy detection:
  - +0.3 per synergistic pair (high compatibility, not conflicting)
  - +0.15 per complementary pair (same problem, different approach)
  - -0.2 per problematic overlap (conflicts detected)
- Saves actual diversity and synergy scores with each selected method

### 7. Scientific Toolkit Visualization
- Three-panel visualization per context:
  - Panel A: Methods table with fitness, compatibility, impact scores
  - Panel B: Implementation roadmap (scatter plot with strategic quadrants)
  - Panel C: Score composition breakdown (showing all 5 components)
- Adaptive text contrast (black on yellow, white on red/green)
- Dynamic heights (no scrollbars)
- Comprehensive explanation section detailing all scoring logic
- Separate and combined HTML outputs for flexibility

## Performance Characteristics

### Deduplication
- Time: ~30-60 minutes for 800 methods
- Bottleneck: Embedding generation
- Optimization: Batch processing (25/batch)

### 12D Analysis
- Time: 75-100 minutes for 595 methods
- Bottleneck: LLM ranking calls
- Optimization: 20 parallel chunks, 5-pass validation
- Total comparisons: ~150,000 for 595 methods

### Compatibility Analysis
- Time: ~2-4 hours for 176,715 pairs
- Bottleneck: Compatibility LLM calls
- Optimization: Smart sampling (90% reduction)
- Comparisons: ~17,500 sampled pairs (vs. 176,715 total possible)
- Output: Rich overlap analysis with 7 fields per pair

### Graph Clustering Visualization
- Time: ~30-60 seconds for 595 methods
- Method: Two-phase physics simulation (60 + 300 iterations)
- Memory: ~50-100 MB for positions and force calculations
- Bottleneck: Molecule formation (greedy search)
- Output: viz_molecular_final.png (~60-70 molecules, 36×36 inches, 300 DPI)

### Toolkit Generation
- Time: ~1-2 minutes (uses pre-computed compatibility data)
- Method: Greedy selection with pairwise synergy scoring
- Contexts: 5 organizational profiles processed
- Output: toolkit_comparison.json with diversity/synergy scores

### Toolkit Visualization
- Time: ~5-10 seconds per context
- Method: Three separate Plotly figures combined in HTML
- Output: 5 context-specific analysis pages (3 panels each)

## Dependencies

### Python Packages
See `requirements.txt` for full list. Key dependencies:
- pandas, numpy (data processing)
- scikit-learn (similarity, clustering)
- openai (LLM/embedding client)
- plotly, matplotlib (visualization)
- pyyaml (configuration)

### External Services
- VLLM server for LLM inference
- BGE embedding server
- Optional: Redis cache for API responses

## Usage Workflow

```bash
# 1. Setup
pip install -r requirements.txt
cp config.example.yaml config.yaml  # Edit with your endpoints

# 2. Deduplication
python find_exact_duplicates.py
python find_and_merge_duplicates.py
python merge_duplicates.py

# 3. Categorization
python assign_intelligent_categories.py

# 4. 12D Analysis
python analyze_12d.py

# 5. Visualization (creates 7 interactive plots: 4 2D + 3 3D)
python create_12d_dashboard.py

# 6. Compatibility Analysis (2-4 hours)
python analyze_compatibility.py

# 7. Graph Clustering Visualization (30-60 seconds) [PARALLEL TO STEP 8]
python visualize_graph.py

# 8. Toolkit Generation (1-2 minutes)
python build_method_portfolios.py --comparison

# 9. Toolkit Visualization (5-10 seconds per context)
python create_toolkit_dashboard.py

# 10. Review outputs
open results/interactive_dashboard.html           # 12D visualizations (7 plots)
open results/viz_molecular_final.png              # Graph clustering visualization
open results/toolkit_startup_mvp_analysis.html    # MVP toolkit analysis
open results/toolkit_startup_scaling_analysis.html # Scaling toolkit analysis
open results/toolkit_regulated_industry_analysis.html  # etc.
```

## Design Principles

1. **Modularity**: Each stage is independent, outputs can be reused
2. **Configurability**: All parameters in config.yaml
3. **Efficiency**: Smart sampling, parallel processing, caching
4. **Quality**: Multi-pass validation, consistency checks
5. **Transparency**: Detailed logging, interactive exploration
6. **Reproducibility**: Seeded randomness, logged parameters

## Known Issues

### Missing Dimensions in Context Profiles
Two context profiles reference dimensions that don't exist in the method data:

- **Regulated Industry**: References `risk_decision_making` (weight: 0.25) - dimension not available, silently ignored
- **Hardware Product**: References `planning_adaptation` (0.25), `risk_decision_making` (0.25), `design_development` (0.20) - none available, silently ignored

**Impact**:
- Regulated context uses only 4 of 5 intended dimensions for fitness calculation
- Hardware context uses only 2 of 5 intended dimensions for fitness calculation
- Fitness scores still computed correctly with available dimensions, but weights don't sum to intended values

**Resolution needed**: Either add these dimensions to the 12D analysis stage, or update context profiles to use only available dimensions with rebalanced weights.

## Future Enhancements

- [ ] Add missing dimensions (risk_decision_making, planning_adaptation, design_development) to 12D analysis
- [ ] Rebalance context profile weights to sum correctly with available dimensions
- [ ] Redis caching integration (currently disabled)
- [ ] GPU acceleration for embeddings
- [ ] Real-time dashboard updates
- [ ] Method recommendation API
- [ ] A/B testing framework for toolkit effectiveness
