# Ranking Prompt: Resources Required

## Dimension Description

**Resources Required** measures the investment needed to implement a method, including budget, tools, people, and infrastructure.

## Scale

| Rank | Description |
|------|-------------|
| 1 | Least resource-intensive (can do with existing resources) |
| {count} | Most resource-intensive (significant budget, tools, external help) |

## Comparison Anchors

| Example Method | Position | Rationale |
|----------------|----------|-----------|
| Daily standup | Near 1 (minimal) | No additional resources needed |
| Enterprise architecture overhaul | Near {count} (intensive) | Significant investment |
| Team coaching | Middle | Moderate resources |

## Prompt Template

```
RANKING TASK: Order these {count} methods by RESOURCE REQUIREMENTS

SCALE:
Rank 1 = Least resource-intensive (can do with existing resources)
Rank {count} = Most resource-intensive (significant budget, tools, external help)

COMPARISON ANCHORS:
- "Daily standup" → near 1 (minimal resources)
- "Enterprise architecture overhaul" → near {count} (very resource-intensive)
- "Team coaching" → middle (moderate resources)

METHODS:
{methods_list}

RULES:
1. Use ALL ranks 1-{count} exactly once (no ties)
2. Spread across full range
3. Think: budget, tools, time, infrastructure needed

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
