# 12D Dashboard Visualization Architecture

## Overview

The 12D Dashboard creates comprehensive visualizations of the 12-dimensional method analysis results. It generates both static (PNG) and interactive (HTML) visualizations that enable exploration of method characteristics across multiple dimensions.

**Key Output:** A 7-plot interactive HTML dashboard with 4 two-dimensional scatter plots and 3 three-dimensional rotation-enabled plots, plus a 12-panel static PNG dashboard.

---

## Input Data

### Primary Input

| File | Description |
|------|-------------|
| `results/method_scores_12d_deduplicated.json` | 12-dimensional scores for 595 methods |

### Supporting Inputs

| File | Description |
|------|-------------|
| `results_semantic_clustering_combined/combined_clusters.json` | Cluster assignments for category coloring |
| `results_semantic_clustering_combined/dendrogram_categories.json` | Category display names (via cluster_utils) |

### Input Data Structure

```json
{
  "methods": [
    {
      "index": 1,
      "name": "Method Name",
      "source": "Source Name",
      "scope": 72.5,
      "temporality": 45.3,
      "ease_adoption": 68.2,
      "resources_required": 35.1,
      "technical_complexity": 42.8,
      "change_management_difficulty": 51.3,
      "implementation_difficulty": 44.6,
      "impact_potential": 78.9,
      "time_to_value": 62.4,
      "applicability": 55.7,
      "people_focus": 48.2,
      "process_focus": 63.1,
      "purpose_orientation": 71.5
    }
  ]
}
```

---

## Output Files

| File | Type | Description |
|------|------|-------------|
| `results/evaluation_dashboard.png` | Static PNG | 12-panel static dashboard (24×20 inches, 150 DPI) |
| `results/interactive_dashboard.html` | Interactive HTML | 7 interactive Plotly plots with dropdowns |
| `results/subcriteria_analysis.png` | Static PNG | Subcriteria histograms (if subcriteria columns present) |

---

## Derived Scores

The dashboard calculates composite scores for categorization and visualization:

### ROI Score

```python
score_roi = (
    impact_potential × 0.40 +
    ease_score × 0.30 +
    speed_score × 0.20 +
    applicability × 0.10
)
```

### Strategic Score

```python
score_strategic = (
    impact_potential × 0.45 +
    applicability × 0.35 +
    scope × 0.20
)
```

### Quick Wins Score

```python
score_quick_wins = (
    speed_score × 0.35 +
    ease_score × 0.35 +
    impact_potential × 0.20 +
    applicability × 0.10
)
```

### Composite Score

```python
score_composite = (
    impact_potential × 0.25 +
    applicability × 0.20 +
    ease_score × 0.20 +
    speed_score × 0.15 +
    scope × 0.10 +
    temporality × 0.10
)
```

Where:
- `ease_score = 100 - implementation_difficulty`
- `speed_score = time_to_value`

---

## Static Dashboard (PNG)

The static dashboard contains 12 panels in a 3×4 grid:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  STATIC DASHBOARD: evaluation_dashboard.png                             │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐              │
│  │ 1. Impact   │ 2. Time to  │ 3. Applica- │ 4. Category │              │
│  │ vs Ease     │ Value Dist  │ bility Dist │ Pie Chart   │              │
│  │ (scatter)   │ (histogram) │ (violin)    │             │              │
│  ├─────────────┼─────────────┼─────────────┼─────────────┤              │
│  │ 5. Portfolio│ 6. Grade    │ 7. Correla- │ 8. Top 5    │              │
│  │ Matrix BCG  │ Distribution│ tion Heatmap│ Radar Chart │              │
│  │ (scatter)   │ (histogram) │             │ (polar)     │              │
│  ├─────────────┼─────────────┼─────────────┼─────────────┤              │
│  │ 9. Score    │ 10. Imple-  │ 11. Quick   │ 12. 3D      │              │
│  │ Distributions│ mentation  │ vs Strategic│ Method      │              │
│  │ (boxplot)   │ Roadmap     │ (scatter)   │ Space       │              │
│  └─────────────┴─────────────┴─────────────┴─────────────┘              │
│                                                                          │
└────────────────────────────────────────────────────────────────────────┘
```

### Panel Descriptions

| # | Panel | Type | X-Axis | Y-Axis | Visual Encoding |
|---|-------|------|--------|--------|-----------------|
| 1 | Impact vs Ease | Scatter | Ease Score | Impact Potential | Color: Time to Value, Size: Applicability |
| 2 | Time to Value | Histogram | Time Categories | Count | 5 bins: Immediate → Very Long |
| 3 | Applicability | Violin | - | Applicability Score | Percentile lines at 25/50/75 |
| 4 | Categories | Pie Chart | - | - | Method category distribution |
| 5 | Portfolio Matrix | Scatter | ROI Score | Strategic Score | BCG-style quadrants |
| 6 | Grade Distribution | Histogram | Letter Grade | Count | A+ through F |
| 7 | Correlation Heatmap | Heatmap | Dimensions | Dimensions | Color: -1 to +1 correlation |
| 8 | Top 5 Methods | Radar | 4 dimensions | - | Ease, Impact, Speed, Applicability |
| 9 | Score Distributions | Boxplot | Score Types | Score Value | ROI, Quick Wins, Strategic, Composite |
| 10 | Implementation Roadmap | Text | - | - | Phase 1-3 method lists |
| 11 | Quick vs Strategic | Scatter | Quick Win Score | Strategic Score | Color: Impact Potential |
| 12 | 3D Method Space | 3D Scatter | Ease | Impact | Z: Applicability, Color: Speed |

### BCG Portfolio Matrix Quadrants

```
                    Strategic Value
                    High (>70)  Low (<70)
ROI Score  ┌──────────────┬──────────────┐
High (>70) │    STARS     │  CASH COWS   │
           │    (gold)    │   (green)    │
           ├──────────────┼──────────────┤
Low (<70)  │  QUESTIONS   │    DOGS      │
           │   (orange)   │    (red)     │
           └──────────────┴──────────────┘
```

---

## Interactive Dashboard (HTML)

The interactive dashboard contains 7 Plotly plots with category filtering:

### 2D Plots (4)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PLOT 1: Impact vs Implementation Difficulty                            │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  X-Axis: ← Easy | Implementation Difficulty | Hard →                    │
│  Y-Axis: ← Low | Impact Potential | High →                              │
│                                                                          │
│  Quadrant interpretation:                                                │
│  • Top-Right: High impact, hard to implement (strategic investments)   │
│  • Top-Left: High impact, easy to implement (quick wins / stars)       │
│  • Bottom-Right: Low impact, hard to implement (avoid)                 │
│  • Bottom-Left: Low impact, easy to implement (low priority)           │
└────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  PLOT 2: Scope vs Temporality                                           │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  X-Axis: ← Tactical | Scope | Strategic →                               │
│  Y-Axis: ← Immediate | Temporality | Evolutionary →                     │
│                                                                          │
│  Quadrant interpretation:                                                │
│  • Top-Right: Strategic scope, long-term change (transformation)       │
│  • Top-Left: Tactical scope, long-term change (sustainable practices) │
│  • Bottom-Right: Strategic scope, immediate results (quick strategic)  │
│  • Bottom-Left: Tactical scope, immediate results (operational)        │
└────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  PLOT 3: Time to Value vs Impact                                        │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  X-Axis: ← Slow | Time to Value | Fast →                                │
│  Y-Axis: ← Low | Impact Potential | High →                              │
│                                                                          │
│  Quadrant interpretation:                                                │
│  • Top-Right: High impact, fast value (ideal methods)                  │
│  • Top-Left: High impact, slow value (long-term investments)           │
│  • Bottom-Right: Low impact, fast value (incremental improvements)     │
│  • Bottom-Left: Low impact, slow value (reconsider necessity)          │
└────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  PLOT 4: People vs Process Focus                                        │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  X-Axis: ← Technical/System | People Focus | Human →                    │
│  Y-Axis: ← Ad-hoc | Process Focus | Systematic →                        │
│                                                                          │
│  Quadrant interpretation:                                                │
│  • Top-Right: Human-centered, systematic (culture + process)           │
│  • Top-Left: Technical, systematic (engineering discipline)            │
│  • Bottom-Right: Human-centered, ad-hoc (informal collaboration)       │
│  • Bottom-Left: Technical, ad-hoc (developer tools/techniques)         │
└────────────────────────────────────────────────────────────────────────┘
```

### 3D Plots (3)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PLOT 5: The Strategic Cube                                             │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  X-Axis: Scope: Tactical (0) ↔ Strategic (100)                         │
│  Y-Axis: Impact: Low (0) ↔ High (100)                                  │
│  Z-Axis: Implementation: Easy (0) ↔ Hard (100)                         │
│                                                                          │
│  Key regions:                                                            │
│  • High X, High Y, Low Z: Strategic stars (ideal)                      │
│  • High X, High Y, High Z: Strategic investments (worth effort)        │
│  • Low X, Low Y, High Z: Avoid (low value, high cost)                  │
└────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  PLOT 6: People × Process × Purpose Space                               │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  X-Axis: People: Technical/System (0) ↔ Human (100)                    │
│  Y-Axis: Process: Ad-hoc (0) ↔ Systematic (100)                        │
│  Z-Axis: Purpose: Internal (0) ↔ External (100)                        │
│                                                                          │
│  Key regions:                                                            │
│  • High X, High Y, High Z: Customer-focused systematic people methods  │
│  • Low X, High Y, Low Z: Internal process automation                   │
│  • High X, Low Y, High Z: Informal customer collaboration              │
└────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  PLOT 7: The Adoption Space                                             │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  X-Axis: Ease of Adoption: Hard (0) ↔ Easy (100)                       │
│  Y-Axis: Change Management: Easy (0) ↔ Hard (100)                      │
│  Z-Axis: Time to Value: Slow (0) ↔ Fast (100)                          │
│                                                                          │
│  Key regions:                                                            │
│  • High X, Low Y, High Z: Easy to adopt, fast value (quick wins)       │
│  • Low X, High Y, Low Z: Hard to adopt, slow value (challenging)       │
│  • High X, High Y, High Z: Easy but needs change management            │
└────────────────────────────────────────────────────────────────────────┘
```

### Interactive Features

Each plot includes:

1. **Category Dropdown Filter**
   - "Show All Categories" (default)
   - Individual category selection with method count
   - Located at top-left of each plot

2. **Legend**
   - Shows all semantic categories with colors
   - Click to toggle visibility
   - Located at right side of each plot

3. **Hover Information**
   - Method name (bold)
   - Category display name
   - Source
   - X/Y/Z dimension values with labels

4. **Interactivity**
   - Pan and zoom (2D plots)
   - Rotate and zoom (3D plots)
   - Click legend to toggle categories

---

## Category Coloring System

### Color Assignment

Categories are loaded from semantic clustering results and assigned colors from a 20-color palette:

```python
color_palette = [
    '#E41A1C',  # Red
    '#377EB8',  # Blue
    '#4DAF4A',  # Green
    '#984EA3',  # Purple
    '#FF7F00',  # Orange
    '#FFFF33',  # Yellow
    '#A65628',  # Brown
    '#F781BF',  # Pink
    '#999999',  # Gray
    '#66C2A5',  # Teal
    '#FC8D62',  # Coral
    '#8DA0CB',  # Periwinkle
    '#E78AC3',  # Orchid
    '#A6D854',  # Lime
    '#FFD92F',  # Gold
    '#1B9E77',  # Dark Teal
    '#D95F02',  # Dark Orange
    '#7570B3',  # Slate
    '#E7298A',  # Magenta
    '#66A61E',  # Olive
]
```

### Category Mapping Process

```
┌─────────────────────────────────────────────────────────────────────────┐
│  CATEGORY MAPPING                                                        │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  1. Load cluster_mappings from cluster_utils.py                         │
│     • cluster_to_synergy: cluster ID → category key                    │
│     • synergy_display_names: category key → display name               │
│                                                                          │
│  2. Load combined_clusters.json                                          │
│     • Build method name → category key mapping                         │
│     • Match by normalized method name (lowercase, stripped)            │
│                                                                          │
│  3. Apply to DataFrame                                                   │
│     • df['method_category'] = lookup by method name                    │
│     • Fallback to 'uncategorized' if not found                         │
│                                                                          │
│  4. Assign colors                                                        │
│     • Iterate synergy_display_names                                     │
│     • Assign color_palette[i % 20]                                      │
│     • Store in category_info dict                                       │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Subcriteria Analysis

If implementation difficulty subcriteria columns are present, generates additional histograms:

| Subcriteria | Description |
|-------------|-------------|
| ease_adoption | How easy to learn and start using |
| resources_required | Budget, tools, infrastructure needed |
| technical_complexity | Technical skill requirements |
| change_management_difficulty | Organizational resistance |

Each histogram shows:
- Distribution of scores (20 bins)
- Mean value (red dashed line)
- Median value (green dashed line)
- Grid lines for readability

---

## HTML Structure

The interactive dashboard HTML structure:

```html
<!DOCTYPE html>
<html>
<head>
    <title>12D Interactive Dashboard - 595 Methods</title>
    <style>
        /* Plot containers with shadow and rounded corners */
        /* Section headers for 2D and 3D groupings */
    </style>
</head>
<body>
    <h1>Method analysis result visualizations - 595 Methods</h1>

    <!-- 2D Plots Section -->
    <div class="section-header">📊 2D Visualizations</div>

    <div class="plot-container">
        <h2>1. Impact vs Implementation Difficulty</h2>
        <!-- Plotly div with dropdown and legend -->
    </div>

    <!-- ... plots 2-4 ... -->

    <!-- 3D Plots Section -->
    <div class="section-header">🎲 3D Visualizations</div>

    <div class="plot-container">
        <h2>5. The Strategic Cube</h2>
        <!-- Plotly 3D div with dropdown and legend -->
    </div>

    <!-- ... plots 6-7 ... -->
</body>
</html>
```

---

## Process Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│  DASHBOARD GENERATION PROCESS                                           │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  1. Load Data                                                            │
│     ─────────                                                            │
│     • Load method_scores_12d_deduplicated.json                          │
│     • Convert to pandas DataFrame                                       │
│     • Calculate derived scores (ROI, Strategic, Quick Wins, Composite) │
│                                                                          │
│  2. Prepare Category Mappings                                            │
│     ──────────────────────────                                           │
│     • Load cluster_mappings via cluster_utils                           │
│     • Load combined_clusters.json                                       │
│     • Build method_name → category mapping                              │
│     • Assign colors from 20-color palette                               │
│                                                                          │
│  3. Create Static Dashboard                                              │
│     ────────────────────────                                             │
│     • Create 3×4 matplotlib figure (24×20 inches)                       │
│     • Generate 12 panel visualizations                                  │
│     • Save as evaluation_dashboard.png (150 DPI)                        │
│                                                                          │
│  4. Create Interactive Dashboard                                         │
│     ───────────────────────────                                          │
│     • Generate 4 2D Plotly scatter plots                                │
│     • Generate 3 3D Plotly scatter plots                                │
│     • Add category dropdowns to each plot                               │
│     • Combine into single HTML page                                     │
│     • Save as interactive_dashboard.html                                │
│                                                                          │
│  5. Create Subcriteria Analysis (Optional)                               │
│     ──────────────────────────────────────                               │
│     • Check for subcriteria columns                                     │
│     • Generate histograms with mean/median lines                        │
│     • Save as subcriteria_analysis.png                                  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Command-Line Interface

```bash
# Auto-detect input file
python create_12d_dashboard.py

# Specify input file
python create_12d_dashboard.py --input results/method_scores_12d_deduplicated.json
```

Auto-detection order:
1. `results/method_scores_12d_deduplicated.json` (preferred)
2. `results/method_scores_12d.json` (fallback)

---

## Visual Design Principles

### 2D Plots

- Quadrant lines at x=50 and y=50 (dashed black, 40% opacity)
- Axis ranges: -5 to 105 (slight padding)
- Marker size: 8 pixels
- Marker opacity: 0.8
- Black edge lines on markers (0.5 width)

### 3D Plots

- Tick labels at 0, 20, 40, 60, 80, 100
- Semantic labels at endpoints (e.g., "0: Tactical", "100: Strategic")
- Light gray grid
- Marker size: 5 pixels (smaller for clarity)
- Marker opacity: 0.8

### Hover Templates

```
<b>Method Name</b>
Category: Display Name
Source: Source Name
Dimension 1: XX.X/100
Dimension 2: XX.X/100
[Dimension 3: XX.X/100]  (3D only)
```

---

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Methods visualized | 595 |
| 2D plots generated | 4 |
| 3D plots generated | 3 |
| Static panels | 12 |
| Category dropdown entries | ~15-20 per plot |
| HTML file size | ~3-5 MB |
| PNG file size (static) | ~2-3 MB |
| Generation time | ~10-15 seconds |
