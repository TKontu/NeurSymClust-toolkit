# Ranking Prompt: Ease of Adoption

## Dimension Description

**Ease of Adoption** measures how easily a method can be learned and implemented by individuals or teams.

## Scale

| Rank | Description |
|------|-------------|
| 1 | Hardest to adopt (steep learning curve, extensive training needed) |
| {count} | Easiest to adopt (intuitive, minimal training, start immediately) |

## Comparison Anchors

| Example Method | Position | Rationale |
|----------------|----------|-----------|
| Test-driven development | Near 1 (hard) | Requires new skills, practice |
| Daily standup meeting | Near {count} (easy) | Learn in 5 minutes |
| Sprint planning | Middle | Some practice needed |

## Prompt Template

```
RANKING TASK: Order these {count} methods by EASE OF ADOPTION

SCALE:
Rank 1 = Hardest to adopt (steep learning curve, extensive training needed)
Rank {count} = Easiest to adopt (intuitive, minimal training, start immediately)

COMPARISON ANCHORS:
- "Test-driven development" → near 1 (requires new skills, practice)
- "Daily standup meeting" → near {count} (learn in 5 minutes)
- "Sprint planning" → middle (some practice needed)

METHODS:
{methods_list}

RULES:
1. Use ALL ranks 1-{count} exactly once (no ties)
2. Spread across full range
3. Think: individual/team learning curve, training time, technical skills, intuitiveness

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
