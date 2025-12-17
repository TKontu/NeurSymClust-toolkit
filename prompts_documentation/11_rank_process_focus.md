# Ranking Prompt: Process Focus

## Dimension Description

**Process Focus** measures the degree of procedural rigor, standardization, and documentation inherent in a method.

## Scale

| Rank | Description |
|------|-------------|
| 1 | Least process-focused (ad-hoc, informal, flexible) |
| {count} | Most process-focused (rigid procedures, standardized, documented) |

## Comparison Anchors

| Example Method | Position | Rationale |
|----------------|----------|-----------|
| Open innovation | Near 1 (ad-hoc) | Informal, flexible |
| Six Sigma | Near {count} (rigid) | Highly standardized process |
| Definition of Done | Middle | Semi-structured |

## Prompt Template

```
RANKING TASK: Order these {count} methods by PROCESS FOCUS

SCALE:
Rank 1 = Least process-focused (ad-hoc, informal, flexible)
Rank {count} = Most process-focused (rigid procedures, standardized, documented)

COMPARISON ANCHORS:
- "Six Sigma" → near {count} (rigid process)
- "Open innovation" → near 1 (ad-hoc)
- "Definition of Done" → middle (semi-structured)

METHODS:
{methods_list}

RULES:
1. Use ALL ranks 1-{count} exactly once (no ties)
2. Spread across full range
3. Think: standardization, documentation, procedural rigor

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
