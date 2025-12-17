# Ranking Prompt: Temporality

## Dimension Description

**Temporality** measures the time horizon of a method's impact, from immediate results to long-term evolutionary change.

## Scale

| Rank | Description |
|------|-------------|
| 1 | Most immediate (hours/days impact, quick results) |
| {count} | Most evolutionary (years impact, long-term cultural change) |

## Comparison Anchors

| Example Method | Position | Rationale |
|----------------|----------|-----------|
| Bug fix | Near 1 (immediate) | Hours to complete |
| Learning organization culture | Near {count} (evolutionary) | Years to develop |
| Sprint cycle | Middle | Weeks/months |

## Prompt Template

```
RANKING TASK: Order these {count} methods by TEMPORALITY

SCALE:
Rank 1 = Most immediate (hours/days impact, quick results)
Rank {count} = Most evolutionary (years impact, long-term cultural change)

COMPARISON ANCHORS:
- "Bug fix" → near 1 (immediate)
- "Learning organization culture" → near {count} (evolutionary)
- "Sprint cycle" → middle (weeks/months)

METHODS:
{methods_list}

RULES:
1. Use ALL ranks 1-{count} exactly once (no ties)
2. Spread across full range
3. Think: how long until meaningful results?

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
