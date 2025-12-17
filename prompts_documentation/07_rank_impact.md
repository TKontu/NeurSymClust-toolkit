# Ranking Prompt: Impact Potential

## Dimension Description

**Impact Potential** measures the potential business value and transformational effect of a method when successfully implemented.

## Scale

| Rank | Description |
|------|-------------|
| 1 | Lowest impact (marginal improvements, minor optimizations) |
| {count} | Highest impact (transformational, game-changing results) |

## Comparison Anchors

| Example Method | Position | Rationale |
|----------------|----------|-----------|
| Naming conventions | Near 1 (low impact) | Minor optimization |
| DevOps culture | Near {count} (high impact) | Transformational |
| Code reviews | Middle | Moderate impact |

## Prompt Template

```
RANKING TASK: Order these {count} methods by IMPACT POTENTIAL

SCALE:
Rank 1 = Lowest impact (marginal improvements, minor optimizations)
Rank {count} = Highest impact (transformational, game-changing results)

COMPARISON ANCHORS:
- "DevOps culture" → near {count} (high impact)
- "Naming conventions" → near 1 (low impact)
- "Code reviews" → middle (moderate impact)

METHODS:
{methods_list}

RULES:
1. Use ALL ranks 1-{count} exactly once (no ties)
2. Spread across full range
3. Think: business value, transformational effect

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
