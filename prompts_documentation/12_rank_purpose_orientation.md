# Ranking Prompt: Purpose Orientation

## Dimension Description

**Purpose Orientation** measures whether a method primarily serves internal organizational needs (efficiency, cost) or external stakeholder value (customers, market).

## Scale

| Rank | Description |
|------|-------------|
| 1 | Most internal (efficiency, cost-cutting, internal processes) |
| {count} | Most external (customer value, innovation, market outcomes) |

## Comparison Anchors

| Example Method | Position | Rationale |
|----------------|----------|-----------|
| Code refactoring | Near 1 (internal) | Internal efficiency focus |
| Customer discovery | Near {count} (external) | External value creation |
| Continuous deployment | Middle | Balanced internal/external |

## Prompt Template

```
RANKING TASK: Order these {count} methods by PURPOSE ORIENTATION

SCALE:
Rank 1 = Most internal (efficiency, cost-cutting, internal processes)
Rank {count} = Most external (customer value, innovation, market outcomes)

COMPARISON ANCHORS:
- "Customer discovery" → near {count} (external value)
- "Code refactoring" → near 1 (internal efficiency)
- "Continuous deployment" → middle (balanced)

METHODS:
{methods_list}

RULES:
1. Use ALL ranks 1-{count} exactly once (no ties)
2. Spread across full range
3. Think: who benefits? organization or customers?

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
