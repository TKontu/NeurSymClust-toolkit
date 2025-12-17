# Prompt Documentation

This folder contains detailed documentation of all prompts used in the method analysis pipeline.

## 12-Dimensional Ranking Prompts

These prompts are used to rank methods relative to each other on each dimension. The LLM receives a chunk of 18 methods and must assign ranks 1-18 with no ties.

| File | Dimension | Scale |
|------|-----------|-------|
| [01_rank_scope.md](01_rank_scope.md) | Scope | Tactical ↔ Strategic |
| [02_rank_temporality.md](02_rank_temporality.md) | Temporality | Immediate ↔ Evolutionary |
| [03_rank_ease_adoption.md](03_rank_ease_adoption.md) | Ease of Adoption | Hard ↔ Easy |
| [04_rank_resources.md](04_rank_resources.md) | Resources Required | Low ↔ High |
| [05_rank_complexity.md](05_rank_complexity.md) | Technical Complexity | Simple ↔ Complex |
| [06_rank_change_mgmt.md](06_rank_change_mgmt.md) | Change Management Difficulty | Easy ↔ Hard |
| [07_rank_impact.md](07_rank_impact.md) | Impact Potential | Low ↔ High |
| [08_rank_time_to_value.md](08_rank_time_to_value.md) | Time to Value | Slow ↔ Fast |
| [09_rank_applicability.md](09_rank_applicability.md) | Applicability | Niche ↔ Universal |
| [10_rank_people_focus.md](10_rank_people_focus.md) | People Focus | Technical ↔ Human |
| [11_rank_process_focus.md](11_rank_process_focus.md) | Process Focus | Ad-hoc ↔ Systematic |
| [12_rank_purpose_orientation.md](12_rank_purpose_orientation.md) | Purpose Orientation | Internal ↔ External |

### Ranking Prompt Structure

Each ranking prompt follows a consistent structure:

1. **Task Definition**: Clear statement of the ranking dimension
2. **Scale Endpoints**: Definition of rank 1 and rank N meanings
3. **Comparison Anchors**: 3 well-known methods as reference points (low, middle, high)
4. **Methods List**: The actual methods to rank (injected at runtime)
5. **Rules**: Constraints on output (no ties, use all ranks, spread across range)
6. **Output Format**: JSON array specification

### Why Ranking Instead of Scoring?

Direct scoring (e.g., "rate this method 0-100") leads to clustering issues where LLMs tend to give similar scores to many methods. Ranking forces differentiation:

- Every method must have a unique rank
- The full scale is always used (1 to N)
- Relative comparisons are more reliable than absolute judgments

## Compatibility Analysis Prompts

These prompts analyze pairwise relationships between methods.

| File | Purpose | Output |
|------|---------|--------|
| [13_overlap_detection.md](13_overlap_detection.md) | Detect functional overlap | 5 boolean fields + overlap_type |
| [14_compatibility_scoring.md](14_compatibility_scoring.md) | Score compatibility | 0-1 score + relationship type |

### Two-Phase Analysis

Each method pair receives two LLM calls:

1. **Overlap Detection**: Structured analysis of whether methods address the same problem, role, timing, or output
2. **Compatibility Scoring**: Calibrated numerical score with qualitative assessment

The results are combined to classify pairs as:
- **Incompatible** (score < 0.7)
- **Synergistic** (score >= 0.95 and not conflicting)
- **Nonrelated** (different purposes, no overlap)
- **Compatible** (everything else)

## Template Variables

All prompts use these template variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `{count}` | Number of methods in chunk | 18 |
| `{methods_list}` | Formatted list of methods with descriptions | 1. Method Name\n   Description... |
| `{method_a}` / `{method_b}` | Method names for pairwise comparison | "Daily Standup" |
| `{description_a}` / `{description_b}` | Method descriptions (truncated to 500 chars) | "A short meeting..." |

## Output Format Requirements

All prompts require:
- Pure JSON output only
- No comments, explanations, or markdown formatting
- No `//` comments inside JSON
- Strict adherence to specified structure

This enables reliable parsing and validation of LLM responses.
