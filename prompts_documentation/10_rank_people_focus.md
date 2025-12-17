# Ranking Prompt: People Focus

## Dimension Description

**People Focus** measures the degree to which a method centers on human elements (relationships, behavior, culture) versus technical/system elements.

## Scale

| Rank | Description |
|------|-------------|
| 1 | Least people-focused (pure technical/systems, automation-driven) |
| {count} | Most people-focused (entirely about people, culture, leadership) |

## Comparison Anchors

| Example Method | Position | Rationale |
|----------------|----------|-----------|
| Automated testing | Near 1 (technical) | Pure technical, no human element |
| Servant leadership | Near {count} (people) | Entirely about people |
| Pair programming | Middle | Balanced technical and human |

## Prompt Template

```
RANKING TASK: Order these {count} methods by PEOPLE FOCUS

SCALE:
Rank 1 = Least people-focused (pure technical/systems, automation-driven)
Rank {count} = Most people-focused (entirely about people, culture, leadership)

COMPARISON ANCHORS:
- "Servant leadership" → near {count} (entirely people)
- "Automated testing" → near 1 (pure technical)
- "Pair programming" → middle (balanced)

METHODS:
{methods_list}

RULES:
1. Use ALL ranks 1-{count} exactly once (no ties)
2. Spread across full range
3. Think: human interaction vs technical/system elements

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
