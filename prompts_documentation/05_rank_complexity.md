# Ranking Prompt: Technical Complexity

## Dimension Description

**Technical Complexity** measures the level of technical sophistication and expertise required to implement a method.

## Scale

| Rank | Description |
|------|-------------|
| 1 | Least complex (non-technical, simple, conceptual) |
| {count} | Most complex (deep technical expertise, sophisticated systems required) |

## Comparison Anchors

| Example Method | Position | Rationale |
|----------------|----------|-----------|
| Team vision statement | Near 1 (simple) | Conceptual, no technical skills |
| Microservices architecture | Near {count} (complex) | Deep technical expertise |
| Automated testing | Middle | Moderate technical needs |

## Prompt Template

```
RANKING TASK: Order these {count} methods by TECHNICAL COMPLEXITY

SCALE:
Rank 1 = Least complex (non-technical, simple, conceptual)
Rank {count} = Most complex (deep technical expertise, sophisticated systems required)

COMPARISON ANCHORS:
- "Team vision statement" → near 1 (simple/conceptual)
- "Microservices architecture" → near {count} (very complex)
- "Automated testing" → middle (moderate technical needs)

METHODS:
{methods_list}

RULES:
1. Use ALL ranks 1-{count} exactly once (no ties)
2. Spread across full range
3. Think: technical skills & system sophistication needed

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
