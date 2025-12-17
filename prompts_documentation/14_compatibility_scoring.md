# Analysis Prompt: Compatibility Scoring

## Purpose

The compatibility scoring prompt evaluates how well two methods can coexist and be used together in an organization. It produces a numerical score and qualitative assessment with practical recommendations.

## Output Fields

| Field | Type | Description |
|-------|------|-------------|
| compatibility_score | float | 0.0-1.0 score indicating how well methods work together |
| relationship_type | enum | synergistic, compatible, neutral, problematic, incompatible |
| recommendation | enum | combine, choose_one, sequence, avoid |
| key_concern | string | Brief description of the main issue (or "none") |

## Calibration Scale

The prompt provides calibration examples to ensure consistent scoring:

| Score | Example Pair | Relationship |
|-------|--------------|--------------|
| 0.95 | Daily Standup + Sprint Planning | Highly synergistic |
| 0.85 | TDD + CI | Strong compatibility |
| 0.60 | Scrum + Kanban | Moderate tension |
| 0.35 | User Stories + Use Cases | Significant overlap |
| 0.20 | Sprint Review + Sprint Demo | Redundant |
| 0.10 | Waterfall + Agile | Incompatible |

## Prompt Template

```
Rate compatibility (0.0-1.0):

Method A: {method_a}
{description_a}

Method B: {method_b}
{description_b}

Calibration scale:
0.95 = Daily Standup + Sprint Planning
0.85 = TDD + CI
0.60 = Scrum + Kanban
0.35 = User Stories + Use Cases
0.20 = Sprint Review + Sprint Demo (redundant)
0.10 = Waterfall + Agile (incompatible)

Consider:
- Resource conflict?
- Purpose overlap?
- Philosophy alignment?
- Can coexist?

Use full 0-1 scale. Redundant methods <0.3, incompatible <0.15.

JSON only:
{
  "compatibility_score": [0.0-1.0],
  "relationship_type": "synergistic|compatible|neutral|problematic|incompatible",
  "recommendation": "combine|choose_one|sequence|avoid",
  "key_concern": "[brief issue or 'none']"
}
```

## Expected Output Format

```json
{
  "compatibility_score": 0.82,
  "relationship_type": "compatible",
  "recommendation": "combine",
  "key_concern": "none"
}
```

## Classification Thresholds

The system applies these rules to the compatibility score:

| Classification | Rule |
|---------------|------|
| Incompatible | score < 0.7 |
| Synergistic | score >= 0.95 AND overlap_type != conflicting |
| Nonrelated | same_problem=false AND same_output=false AND overlap_type=none |
| Compatible | Everything else (0.7 <= score < 0.95) |
