# Ranking Prompt: Scope

## Dimension Description

**Scope** measures the organizational reach of a method, from tactical (individual/team level) to strategic (enterprise-wide).

## Scale

| Rank | Description |
|------|-------------|
| 1 | Most tactical (individual/team level, specific tasks) |
| {count} | Most strategic (enterprise-wide, organizational transformation) |

## Comparison Anchors

| Example Method | Position | Rationale |
|----------------|----------|-----------|
| Daily standup meeting | Near 1 (tactical) | Team-level, specific task |
| Digital transformation | Near {count} (strategic) | Enterprise-wide change |
| Release management | Middle | Multi-team coordination |

## Prompt Template

```
RANKING TASK: Order these {count} methods by SCOPE

SCALE:
Rank 1 = Most tactical (individual/team level, specific tasks)
Rank {count} = Most strategic (enterprise-wide, organizational transformation)

COMPARISON ANCHORS:
- "Daily standup meeting" → near 1 (tactical)
- "Digital transformation" → near {count} (strategic)
- "Release management" → middle (multi-team coordination)

METHODS:
{methods_list}

RULES:
1. Use ALL ranks 1-{count} exactly once (no ties)
2. Spread across full range - don't cluster
3. Compare each method to anchors and to each other

OUTPUT FORMAT:
- Return ONLY the JSON array: [[method_number, rank],[method_number, rank], ...]
- NO comments, explanations, reasoning, or markdown formatting
- NO `//` comments inside the JSON
- Just the pure JSON array

JSON Response:
```

## Expected Output Format

```json
[[1, 5], [2, 12], [3, 1], [4, 18], ...]
```

Each entry is `[method_number, assigned_rank]` where method_number corresponds to the position in the input list.
