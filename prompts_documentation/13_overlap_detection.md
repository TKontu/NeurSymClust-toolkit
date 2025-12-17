# Analysis Prompt: Overlap Detection

## Purpose

The overlap detection prompt analyzes whether two methods address the same problem space, are executed by the same roles, occur at the same time, or produce the same outputs. This structured analysis helps identify redundant or conflicting methods.

## Output Fields

| Field | Type | Description |
|-------|------|-------------|
| same_problem | boolean | Do both methods address the same core problem? |
| same_role | boolean | Does the same role/person execute both methods? |
| same_timing | boolean | Are they used at the same time/ceremony? |
| same_output | boolean | Do they produce the same artifact or output? |
| causes_confusion | boolean | Would using both cause team confusion? |
| overlap_type | enum | Classification: none, partial, conflicting, redundant |

## Classification Rules

| YES Count | Classification |
|-----------|----------------|
| 4-5 | redundant |
| 2-3 | conflicting |
| 1 | partial |
| 0 | none |

## Prompt Template

```
Analyze overlap between these methods:

Method A: {method_a_name}
{method_a_description}

Method B: {method_b_name}
{method_b_description}

YES/NO for each:
1. Same core problem?
2. Same role executes both?
3. Same timing/ceremony?
4. Same output/artifact?
5. Confusing to use both?

Classification:
- 4-5 YES = redundant
- 2-3 YES = conflicting
- 1 YES = partial
- 0 YES = none

JSON only:
{
  "same_problem": true/false,
  "same_role": true/false,
  "same_timing": true/false,
  "same_output": true/false,
  "causes_confusion": true/false,
  "overlap_type": "none|partial|conflicting|redundant"
}
```

## Expected Output Format

```json
{
  "same_problem": true,
  "same_role": false,
  "same_timing": true,
  "same_output": false,
  "causes_confusion": true,
  "overlap_type": "conflicting"
}
```

## Derived Field

The system derives `has_problematic_overlap` from the response:
- `true` if `overlap_type` is "redundant" or "conflicting"
- `false` otherwise
