# Ranking Prompt: Time to Value

## Dimension Description

**Time to Value** measures how quickly a method delivers meaningful results after implementation begins.

## Scale

| Rank | Description |
|------|-------------|
| 1 | Slowest (takes years to see meaningful results) |
| {count} | Fastest (immediate or near-immediate results) |

## Comparison Anchors

| Example Method | Position | Rationale |
|----------------|----------|-----------|
| Culture transformation | Near 1 (slow) | Years to show results |
| Daily standup | Near {count} (fast) | Immediate visibility |
| Team coaching | Middle | Moderate time to value |

## Prompt Template

```
RANKING TASK: Order these {count} methods by TIME TO VALUE

SCALE:
Rank 1 = Slowest (takes years to see meaningful results)
Rank {count} = Fastest (immediate or near-immediate results)

COMPARISON ANCHORS:
- "Culture transformation" → near 1 (very slow)
- "Daily standup" → near {count} (very fast)
- "Team coaching" → middle (moderate)

METHODS:
{methods_list}

RULES:
1. Use ALL ranks 1-{count} exactly once (no ties)
2. Spread across full range
3. Think: how quickly does it deliver value after implementation?

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
