# Ranking Prompt: Change Management Difficulty

## Dimension Description

**Change Management Difficulty** measures the organizational resistance and cultural transformation required to implement a method.

## Scale

| Rank | Description |
|------|-------------|
| 1 | Least difficult (minimal organizational disruption, easy buy-in) |
| {count} | Most difficult (major cultural shift, organizational resistance, power structure changes) |

## Comparison Anchors

| Example Method | Position | Rationale |
|----------------|----------|-----------|
| Adding a task board | Near 1 (easy) | Minimal disruption |
| Self-organizing teams | Near {count} (difficult) | Power shift, cultural transformation |
| Cross-functional teams | Middle | Moderate organizational change |

## Prompt Template

```
RANKING TASK: Order these {count} methods by CHANGE MANAGEMENT DIFFICULTY

SCALE:
Rank 1 = Least difficult (minimal organizational disruption, easy buy-in)
Rank {count} = Most difficult (major cultural shift, organizational resistance, power structure changes)

COMPARISON ANCHORS:
- "Adding a task board" → near 1 (minimal disruption)
- "Self-organizing teams" → near {count} (power shift, cultural transformation)
- "Cross-functional teams" → middle (moderate organizational change)

METHODS:
{methods_list}

RULES:
1. Use ALL ranks 1-{count} exactly once (no ties)
2. Spread across full range
3. Think: organizational inertia, stakeholder buy-in, workflow disruption, cultural transformation

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
